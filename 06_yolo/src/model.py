import torch
import torch.nn as nn

class YOLO(nn.Module):

    """
    YOLO v1-style object detection model.

    Applies sigmoid to bounding box predictions and softmax to class scores.

    Output shape:
        [batch_size, grid_size, grid_size, num_boxes * 5 + num_classes]

    Each bounding box prediction contains:
        - x, y: Bounding box center coordinates.
        - w, h: Bounding box width and height.
        - confidence: Predicted confidence score.

    Class scores represent probabilities across classes and sum to 1
    for each grid cell.
    """

    def __init__(self, grid_size, num_boxes, num_classes):
        super().__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.conv1 = self.block(3, 64, 7, 2, 3)
        self.conv2 = self.block(64, 192, 3, 1, 1)
        self.conv3 = nn.Sequential(
            self.block(192, 128, 1),
            self.block(128, 256, 3, 1, 1),
            self.block(256, 256, 1),
            self.block(256, 512, 3, 1, 1),
        )   

        self.conv4 = nn.Sequential(
            self.block(512, 256, 1, 1),
            self.block(256, 512, 3, 1, 1),
            self.block(512, 256, 1, 1),
            self.block(256, 512, 3, 1, 1),
            self.block(512, 256, 1, 1),
            self.block(256, 512, 3, 1, 1),
            self.block(512, 256, 1, 1),
            self.block(256, 512, 3, 1, 1),
            self.block(512, 512, 1, 1),
            self.block(512, 1024, 3, 1, 1),
        )

        self.conv5 = nn.Sequential(
            self.block(1024, 512, 1),
            self.block(512, 1024, 3, 1, 1),
            self.block(1024, 512, 1),
            self.block(512, 1024, 3, 1, 1),
            self.block(1024, 1024, 3, 1, 1),
            self.block(1024, 1024, 3, 2, 1),
        )

        self.conv6 = nn.Sequential(
            self.block(1024, 1024, 3, 1, 1),
            self.block(1024, 1024, 3, 1, 1),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(0.1),
        )

        self.fc2 = nn.Linear(
            4096,
            self.grid_size 
            * self.grid_size 
            * (self.num_boxes * 5 + self.num_classes),
        )

    @staticmethod
    def block(in_channel, out_channel, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
            nn.LeakyReLU(0.1)
        )
    
    def forward(self, img):
        out = self.conv1(img)
        out = self.maxpool(out)

        out = self.conv2(out)
        out = self.maxpool(out)

        out = self.conv3(out)
        out = self.maxpool(out)

        out = self.conv4(out)
        out = self.maxpool(out)

        out = self.conv5(out)
        out = self.conv6(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)

        out = self.fc2(out)

        out = out.view(
            out.size(0),
            self.grid_size,
            self.grid_size,
            self.num_boxes * 5 + self.num_classes,
        )

        # Apply sigmoid to bounding box coordinates and confidence
        boxes = torch.sigmoid(out[..., :self.num_boxes * 5])

        # Apply softmax to class scores
        classes  = torch.softmax(
            out[..., self.num_boxes * 5:],
            dim = -1,
        )

        return torch.cat((boxes, classes), dim = -1)