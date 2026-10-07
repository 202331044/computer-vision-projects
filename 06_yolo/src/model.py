import torch.nn as nn

class YOLO(nn.Module):

    def __init__(self, grid_size, num_boxes, num_classes):
        super().__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.conv1 = self.block(3, 64, 7, 2, 3)
        self.conv2 = self.block(64, 192, 3, 1, 1)
        self.conv3 = nn.Sequential(self.block(192, 128, 1),
                                   self.block(128, 256, 3, 1, 1),
                                   self.block(256, 256, 1),
                                   self.block(256, 512, 3, 1, 1)
                                   )   


        self.conv4 = nn.Sequential(self.block(512, 256, 1, 1),
                                   self.block(256, 512, 3, 1, 1),
                                   self.block(512, 256, 1, 1),
                                   self.block(256, 512, 3, 1, 1),
                                   self.block(512, 256, 1, 1),
                                   self.block(256, 512, 3, 1, 1),
                                   self.block(512, 256, 1, 1),
                                   self.block(256, 512, 3, 1, 1),
                                   self.block(512, 512, 1, 1),
                                   self.block(512, 1024, 3, 1, 1)
                                   )

        self.conv5 = nn.Sequential(self.block(1024, 512, 1),
                                   self.block(512, 1024, 3, 1, 1),
                                   self.block(1024, 512, 1),
                                   self.block(512, 1024, 3, 1, 1),
                                   self.block(1024, 1024, 3, 1, 1),
                                   self.block(1024, 1024, 3, 2, 1)
                                   )


        self.conv6 = nn.Sequential(self.block(1024, 1024, 3, 1, 1),
                                   self.block(1024, 1024, 3, 1, 1)
                                   )

        self.fc1 = nn.Sequential(nn.Linear(1024 * 7 * 7, 4096),
                                 nn.LeakyReLU(0.1)
                                )

        self.fc2 = nn.Linear(4096, 
                             self.grid_size * self.grid_size * \
                             (self.num_boxes * 5 + self.num_classes))

    @staticmethod
    def block(in_channel, out_channel, kernel_size, stride=1, padding=0):
        return nn.Sequential(nn.Conv2d(in_channel, 
                                       out_channel, 
                                       kernel_size,
                                       stride, 
                                       padding),
                             nn.LeakyReLU(0.1))
    
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

        return out