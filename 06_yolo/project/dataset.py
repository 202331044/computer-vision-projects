import os
import shutil
from tqdm import tqdm
from utils import pascal_to_yolo

def creat_yolo_dataset(
    files, 
    image_dir, 
    save_img_dir, 
    xml_dir, 
    label_dir, 
    class_names,
    year
):

    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for filename in tqdm(files):
        basename = os.path.splitext(filename)[0]
        new_basename = f'{year}_{basename}'

        shutil.copy(
            os.path.join(image_dir, basename + ".jpg"),
            os.path.join(save_img_dir, new_basename+ ".jpg")
        )

        pascal_to_yolo(
            os.path.join(xml_dir, basename + ".xml"), 
            os.path.join(label_dir, new_basename + ".txt"),
            class_names
        )