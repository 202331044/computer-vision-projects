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
    
    x1 = max(pred_x1, target_x1)
    x2 = min(pred_x2, target_x2)
    y1 = max(pred_y1, target_y1)
    y2 = min(pred_y2, target_y2)

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    intersection = inter_w * inter_h

    union = (pred_w * pred_h + target_w * target_h) - intersection

    if union <= 0: 
        return 0
    
    return intersection / union

if __name__ == '__main__':
    s = 7
    b = 2
    c = 20

    model = YOLO(s, b, c)
    img = torch.randn(2, 3, 448, 448)
    out = model(img)
    out = out.view(-1, s, s, b * 5 + c)

    #target tensor, predictor = 1
    batch_idx = 0

    target = torch.zeros(s, s, b * 5 + c)
    tgt_r = 2
    tgt_c = 3
    tgt_x = 0.906
    tgt_y = 0.813
    tgt_w = 0.223
    tgt_h = 0.268
    tgt_conf = 1
    tgt_class_idx = 5

    target[tgt_r, tgt_c, :5] = torch.tensor([tgt_x, tgt_y, tgt_w, tgt_h, tgt_conf])
    target[tgt_r, tgt_c, 10 + tgt_class_idx] = 1

    #IOU

    pred_x1, pred_y1, pred_w1, pred_h1, pred_conf1 = out[batch_idx, tgt_r, tgt_c, :5]
    pred_x2, pred_y2, pred_w2, pred_h2, pred_conf2 = out[batch_idx, tgt_r, tgt_c, 5:10]

    iou1 = iou((pred_x1, pred_y1, pred_w1, pred_h1), (tgt_x, tgt_y, tgt_w, tgt_h))
    iou2 = iou((pred_x2, pred_y2, pred_w2, pred_h2), (tgt_x, tgt_y, tgt_w, tgt_h))

    print(iou1.item())
    print(iou2.item())

    if iou1.item() >= iou2.item():
        predictor = 0
    else:
        predictor = 1

