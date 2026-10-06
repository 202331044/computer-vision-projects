import cv2
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
import torch
import numpy as np

class VOCDataset(Dataset):
    def __init__(self, 
                 image_set, 
                 target_width, 
                 target_height, 
                 grid_size,
                 transform=None):

        self.grid_size = grid_size
        self.target_width = target_width
        self.target_height = target_height
        self.transform = transform

        self.dataset = VOCDetection(
                                    root = './data',
                                    year = '2007',
                                    image_set = image_set,
                                    download = True
                                    )

        self.VOC_CLASSES = [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]

        self.class_id_dict = { name: idx for idx, name in enumerate(self.VOC_CLASSES)}
        
        self.targets = self._make_targets()

    def _make_targets(self):
        
        """
        new_targets: [new_target, ... ]

        new_target: [[[x_center, y_center, width, height, class_idx], ] ... ]

        """
        new_targets = []

        for _, targets in self.dataset:
            new_target = []
            target = targets['annotation']
            
            for obj in target['object']:
                name = obj['name']
                class_id = self.class_id_dict[name]
                
                bbox = obj['bndbox']
                xmax = float(bbox['xmax'])
                xmin = float(bbox['xmin'])
                ymax = float(bbox['ymax'])
                ymin = float(bbox['ymin'])

                width = xmax - xmin
                height = ymax - ymin
                x_center = xmin + (width / 2)
                y_center = ymin + (height / 2)

                new_target.append([x_center, y_center, width, height, class_id])

            new_targets.append(new_target)

        return new_targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        target_bbox[row, col]:
            [x_cell, y_cell, bbox_w_ratio, bbox_h_ratio, class_id, objectness]

            x_cell, y_cell:
                Bounding box center coordinates within the grid cell.
                Range: [0, 1)

            bbox_w_ratio, bbox_h_ratio:
                Bounding box width/height normalized by image size.
                Range: [0, 1]

            class_id:
                Class index.
                Range: [0, num_classes)

            objectness:
                1 if an object exists in this grid cell, otherwise 0.
        """
        
        img, _ = self.dataset[idx]
        targets = self.targets[idx]

        ##img resize

        img = np.array(img)

        img_h, img_w = img.shape[:2]
        ratio = min(self.target_width / img_w, self.target_height / img_h)

        resized_img_w = int(img_w * ratio)
        resized_img_h = int(img_h * ratio)
        
        resized_img = cv2.resize(img, (resized_img_w, resized_img_h))

        ##add img padding

        img_padding_w = self.target_width - resized_img_w
        img_padding_h = self.target_height - resized_img_h

        left = img_padding_w // 2
        right = img_padding_w - left

        top = img_padding_h // 2
        bottom = img_padding_h - top

        padded_img = cv2.copyMakeBorder(resized_img,
                                        top,
                                        bottom,
                                        left,
                                        right,
                                        cv2.BORDER_CONSTANT,
                                        value = (114, 114, 114))

        if self.transform is not None:
            padded_img = self.transform(padded_img)
        
        #resize bbox
        
        target_bbox = torch.zeros(self.grid_size, 
                                  self.grid_size,
                                  6, 
                                  dtype=torch.float32)
        for target in targets:
            bbox_x_cen, bbox_y_cen, bbox_w, bbox_h, class_id = target

            resized_bbox_x = bbox_x_cen * ratio + left
            resized_bbox_y = bbox_y_cen * ratio + top

            resized_bbox_w = bbox_w * ratio
            resized_bbox_h = bbox_h * ratio

            bbox_x_ratio = resized_bbox_x / self.target_width
            bbox_y_ratio = resized_bbox_y / self.target_height

            bbox_w_ratio = resized_bbox_w / self.target_width
            bbox_h_ratio = resized_bbox_h / self.target_height

            x_cell = self.grid_size * bbox_x_ratio
            y_cell = self.grid_size * bbox_y_ratio

            row = min(int(y_cell), self.grid_size - 1)
            col = min(int(x_cell), self.grid_size - 1)

            x_cell = x_cell - col
            y_cell = y_cell - row

            if target_bbox[row, col][5] == 1:
                continue

            target_bbox[row, col] = torch.tensor([
                                                x_cell, 
                                                y_cell, 
                                                bbox_w_ratio, 
                                                bbox_h_ratio, 
                                                class_id, 
                                                1],
                                                dtype=torch.float32)
        return padded_img, target_bbox

# def resize_img_targets(img, target, target_h, target_w, grid_size):

#     img = np.array(img)

#     width, height = img.size
#     ratio = min(target_h/height, target_w/width)

#     new_w = int(width * ratio)
#     new_h = int(height * ratio)

#     resized_img = cv2.resize(img, (new_w, new_h))
#     pad_w = target_w - new_w
#     pad_h = target_h - new_h

#     left = pad_w // 2
#     right = pad_w - left
#     top = pad_h // 2
#     bottom = pad_h - top

#     new_img = cv2.copyMakeBorder(resized_img,
#                         top,
#                         bottom,
#                         left,
#                         right,
#                         cv2.BORDER_CONSTANT,
#                         value = (114, 114, 114))
    
#     new_target = torch.zeros(grid_size, grid_size, 6)
    
#     for row in range(grid_size):
#         for col in range(grid_size):
#             if target[row, col][-1] == 0:
#                 continue

#             x, y, w, h, class_id, objectness = target[row, col]
#             x = (x + col) / grid_size
#             y = (y + row) / grid_size

#             x = x * width * ratio
#             y = y * height * ratio
#             w = w * width * ratio
#             h = h * height * ratio

#             x = x + left
#             y = y + top

#             new_x = grid_size * (x / target_w)
#             new_y = grid_size * (y / target_h)

#             new_row = int(new_y)
#             new_col = int(new_x)
#             new_x = new_x - new_col
#             new_y = new_y - new_row

#             w = w / target_w
#             h = h / target_h

#             new_target[new_row, new_col] = new_x, new_y, w, h, class_id, objectness
                
#     return new_img, new_target

# def resize_images_labels(img_dir, all_labels, target_h, target_w):
#     """
#     resized_imgs = {
#         file_name: img
#     }

#     all_labels = {
#         file_name: file_labels
#     }

#     file_labels = [
#         [class_id, x, y, w, h],
#         ...
#     ]
    
#     """
#     resized_imgs = {}

#     for img_file in img_dir.glob("*.jpg"):
#         img = cv2.imread(img_file)
#         height, width = img.shape[:2]

#         ratio = min(target_h/height, target_w/width)

#         new_w = int(width * ratio)
#         new_h = int(height * ratio)

#         resized_img = cv2.resize(img, (new_w, new_h))
#         pad_w = target_w - new_w
#         pad_h = target_h - new_h

#         left = pad_w // 2
#         right = pad_w - left
#         top = pad_h // 2
#         bottom = pad_h - top

#         padded_img = cv2.copyMakeBorder(resized_img,
#                            top,
#                            bottom,
#                            left,
#                            right,
#                            cv2.BORDER_CONSTANT,
#                            value = (114, 114, 114))
        
#         resized_imgs[img_file.stem] = padded_img

#         for label in all_labels[img_file.stem]:
#             class_id, x, y, w, h = label

#             x = x * width  * ratio
#             y = y * height * ratio
#             w = w * width * ratio
#             h = h * height * ratio

#             x = x + left
#             y = y + top

#             x = x / target_w
#             y = y / target_h
#             w = w / target_w
#             h = h / target_h

#             label[:] = [class_id, x, y, w, h]

# def load_labels(label_dir):
#     """
#     all_labels = {
#         file_name: file_labels
#     }

#     file_labels = [
#         [class_id, x, y, w, h],
#         ...
#     ]

#     """
     
#     all_labels = {}

#     for label_file in label_dir.glob("*.txt"):  
#         with open(label_file, "r") as f:
#             file_labels = []
#             for line in f:
#                 class_id, x, y, w, h = map(float, line.split())
#                 file_labels.append([int(class_id), x, y, w, h])

#             all_labels[label_file.stem] = file_labels
    
#     return all_labels



#     return resized_imgs, all_labels

# def convert_labels_to_grid(all_labels, grid_size):
#     """
#     grid_labels = {
#         file_name: file_labels
#     }

#     file_labels = [
#         [class_id, row, col, x, y, w, h],
#         ...
#     ]
    
#     """
#     grid_labels = {}

#     for img_name, file_labels in all_labels.items():
#         file_grid_labels = []

#         for label in file_labels:
#             class_id, x, y, w, h = label

#             x = x * grid_size
#             y = y * grid_size

#             col = int(x)
#             row = int(y)
#             x = x - col
#             y = y - row

#             file_grid_labels.append([class_id, row, col, x, y, w, h])
        
#         grid_labels[img_name] = file_grid_labels
    
#     return grid_labels