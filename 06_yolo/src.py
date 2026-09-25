import torch.nn as nn
import torch

class YOLO(nn.Module):
    def __init__(self, s, b, c):
        super().__init__()
        self.s = s
        self.b = b
        self.c = c

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
        self.fc2 = nn.Linear(4096, self.s * self.s *(self.b * 5 + self.c))

    @staticmethod
    def block(in_channel, out_channel, kernel_size, stride=1, padding=0):
        return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size,
                                       stride, padding),
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

def iou(pred_box, target_box):

    pred_x, pred_y, pred_w, pred_h = pred_box
    target_x, target_y, target_w, target_h = target_box

    pred_x1 = pred_x - pred_w/2
    pred_y1 = pred_y - pred_h/2
    pred_x2 = pred_x + pred_w/2
    pred_y2 = pred_y + pred_h/2

    target_x1 = target_x - target_w/2
    target_y1 = target_y - target_h/2
    target_x2 = target_x + target_w/2
    target_y2 = target_y + target_h/2
    
    x1 = torch.maximum(pred_x1, target_x1)
    x2 = torch.minimum(pred_x2, target_x2)
    y1 = torch.maximum(pred_y1, target_y1)
    y2 = torch.minimum(pred_y2, target_y2)

    inter_w = torch.clamp(x2 - x1, min = 0)
    inter_h = torch.clamp(y2 - y1, min = 0)
    intersection = inter_w * inter_h

    union = (pred_w * pred_h + target_w * target_h) - intersection
    union = torch.clamp(union, min = 1e-6)
    
    return intersection / union

def loss(targets, pred, predictors, grid_size, num_boxes, num_classes):
    coord = 5
    noobj = 0.5

    loss1 = 0.0
    loss2 = 0.0
    loss3 = 0.0
    loss4 = 0.0
    loss5 = 0.0

    batch_size = targets.shape[0]
    target_size = targets.shape[1]

    tgt_sets = []

    for batch_idx in range(batch_size):
        tgt_set = set()

        for obj_idx in range(target_size):
            (tgt_r, tgt_c, 
             tgt_x, tgt_y, 
             tgt_w, tgt_h, 
             _, tgt_class_idx) = targets[batch_idx][obj_idx]

            tgt_r = int(tgt_r)
            tgt_c = int(tgt_c)
            tgt_class_idx = int(tgt_class_idx)
            predictor = int(predictors[batch_idx][obj_idx])

            pred_idx = predictor * 5

            pred_x, pred_y, pred_w, pred_h, pred_conf =\
                  pred[batch_idx, tgt_r, tgt_c, pred_idx : pred_idx + 5]

            tgt_conf = iou((pred_x, pred_y, pred_w, pred_h ),
                           (tgt_x, tgt_y, tgt_w, tgt_h))
    
            loss1 += ((tgt_x - pred_x) ** 2 + (tgt_y - pred_y) ** 2)

            loss2 += ((torch.sqrt(tgt_w) - torch.sqrt(pred_w)) ** 2 +
                      (torch.sqrt(tgt_h) - torch.sqrt(pred_h)) ** 2)
            
            loss3 += (tgt_conf - pred_conf) ** 2

            tgt_set.add((tgt_r, tgt_c, predictor))

            for class_idx in range(num_classes):
                tgt_prob = 1 if tgt_class_idx == class_idx else 0
                pred_prob = pred[batch_idx, tgt_r, tgt_c, 10 + class_idx]

                loss5 += (tgt_prob - pred_prob) ** 2


        tgt_sets.append(tgt_set)

    for batch_idx in range(batch_size):
        for row in range(grid_size):
            for col in range(grid_size):
                for box_idx in range(num_boxes):
                    if (row, col, box_idx) not in tgt_sets[batch_idx]:
                        idx = box_idx * 5
                        conf = pred[batch_idx, row, col, idx + 4]
                        loss4 += conf ** 2
                
    loss1 *= coord
    loss2 *= coord
    loss4 *= noobj

    return loss1, loss2, loss3, loss4, loss5

if __name__ == '__main__':
    grid_size = 7
    num_boxes = 2
    num_classes = 20

    model = YOLO(grid_size, num_boxes, num_classes)
    img = torch.randn(2, 3, 448, 448)
    out = model(img)
    out = out.view(-1, grid_size, grid_size, num_boxes * 5 + num_classes)
    batch_idx = 0

    targets = torch.tensor([
        [
            [2, 3, 0.5, 0.5, 0.2, 0.3, 1.0, 5]
        ]
    ])

    predictors = torch.tensor([
        [0]
    ])

    pred = torch.zeros(1, grid_size, grid_size, num_boxes * 5 + num_classes)

    pred[0, 2, 3, 0] = 0.6
    pred[0, 2, 3, 1] = 0.4
    pred[0, 2, 3, 2] = 0.25
    pred[0, 2, 3, 3] = 0.35
    pred[0, 2, 3, 4] = 0.8 
    pred[0, 2, 3, 15] = 0.7

    print(loss(targets, pred, predictors, grid_size, num_boxes, num_classes))