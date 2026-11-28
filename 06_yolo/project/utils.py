import os
import shutil
import tarfile
import xml.etree.ElementTree as ET

def make_data_yaml(config):
    data_config = {
        'path': str(config.YOLO_DIR),
        'train':"images/train",
        'val':'images/val',
        'test':'images/test',
        'nc':config.NUM_CLASSES,
        'names': config.CLASS_NAMES
    }

    config.DATA_YAML.parent.mkdir(
        parents=True, 
        exist_ok=True
    )

    with open(config.DATA_YAML, 'w', encoding='utf-8') as f:
        yaml.dump(
            data_config, 
            f, 
            sort_keys=False, 
            allow_unicode=True
        )
    
    return data_config

def get_xml_files(xml_dir):
    files = []
    for file in os.listdir(xml_dir):
        if not file.endswith(".xml"):
            continue

        files.append(file)
    
    return files

def extract_voc_dataset(src, local_tar, extract_path):
    shutil.copy2(src, local_tar)

    os.makedirs(extract_path, exist_ok = True)

    with tarfile.open(local_tar, 'r') as tar:
        tar.extractall(path = extract_path)

def pascal_to_yolo(xml_path, label_path, class_names):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    depth = int(size.find('depth').text)

    annotations = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text

        if class_name not in class_names:
            continue

        class_id = class_names.index(class_name)

        bbox = obj.find('bndbox')

        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2

        width = xmax - xmin
        height = ymax - ymin

        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height

        annotations.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )


    with open(label_path, "w") as f:
        f.write("\n".join(annotations))
    
    return annotations

def pascal_folder_to_yolo(xml_dir, label_dir, class_names):
    os.makedirs(label_dir, exist_ok=True)

    for filename in os.listdir(xml_dir):
        if not filename.endswith('.xml'):
            continue
        
        xml_path = os.path.join(xml_dir, filename)

        txt_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(label_dir, txt_filename)

        pascal_to_yolo(xml_path, label_path, class_names)