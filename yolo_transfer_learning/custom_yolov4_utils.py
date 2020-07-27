import re
import subprocess
from glob import glob
from os.path import join
from typing import Dict, List, Optional

import cv2
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure


def run_shell_script(file_name: str) -> None:
    """
    A function run *.sh script. It assumes the file is in the current directory.
    Args:
        file_name: str

    Returns:None

    """
    cmd = ["./" + file_name]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    for line in iter(p.stdout.readline, b""):
        print(">>> " + line.decode().rstrip())


def get_image_names(file_path: str) -> List:
    """
    A function to read file that contain all the images and return the names in a list.
    Args:
        file_path: str

    Returns: A List

    """
    name_list = []
    f = open(file_path, "r")
    lines = f.readlines()
    f.close()
    for line in lines:
        name = line.split("/")[-1].replace("\n", "")
        name_list.append(name)
    return name_list


def get_img_index(pred_result: List) -> List:
    """
    A function to get the index of each image, so downstream function could split information accordingly.
    Args:
        pred_result: List

    Returns: A list.

    """
    img_index = []

    for i, line in enumerate(pred_result):
        match = re.search("\./test_pics/", line)
        if match:
            img_index.append(i)
        if i == len(pred_result) - 1:
            img_index.append(i)
    return img_index


def find_bbox(pred_file_path: str, train_file_path: str) -> Dict:
    """
    A function to
    - Check if the number of testing images are equal to the number of prediction results.
    - Loop through the output of Yolo prediction .txt file
    - Create a dictionary for each image, use the image name as key and bbox information as value.

    Args:
        pred_file_path: str
        train_file_path: str

    Returns: A dictionary

    """

    f_pred = open(pred_file_path, "r")
    pred_result = f_pred.readlines()
    f_pred.close()

    img_index = get_img_index(pred_result)

    img_names = get_image_names(train_file_path)

    if len(img_index) - 1 != len(img_names):
        return "There is mismatch between the number of predictions and the number of images."

    # Create dictionary with the img name as the key and the bbox information as values.
    target_labels = ["TableCaption", "TableBody", "TableFootnote", "Paragraph", "Table"]
    result = {}
    for i, name in enumerate(img_names):
        key = name
        start = img_index[i] + 1
        end = img_index[i + 1]
        unfiltered_value = pred_result[start:end]
        filtered_value = [
            v for v in unfiltered_value if v.split(":")[0] in target_labels
        ]
        result[key] = filtered_value

    return result


def create_train_file(img_folder_path: str, train_file_path: str) -> None:
    """
    A function to find all the images in a directory and create a .txt file that contains all the images path in the
    target directory (train_file_path).
    Args:
        img_folder_path: str
        train_file_path: str

    Returns: None

    """
    files = []
    for ext in ("*.gif", "*.png", "*.jpg", "*.bmp"):
        img_path = glob(join(img_folder_path, ext))
        if img_path:
            files.extend(img_path)

    write_to_train_file(files, train_file_path)

    print("Training files are created in " + img_folder_path)


def write_to_train_file(files: List, train_file_path: str) -> None:
    """
    A function to create files in the targe directory.
    Args:
        files:List
        train_file_path:str

    Returns: None

    """
    f = open(train_file_path, "w")
    text_to_save = ""
    for i, img_path in enumerate(files):
        img_path_stripped = img_path.replace("/darknet", "")
        if i == len(files) - 1:
            text_to_save += img_path_stripped
        else:
            text_to_save += img_path_stripped + "\n"

    f.write(text_to_save)
    f.close()


def draw_pic(img_path: str, bbox: List) -> None:
    """
    A function to visualize bbox in an image.
    Args:
        img_path: str
        bbox: list

    Returns: None

    """
    img = cv2.imread(img_path)

    for v in bbox:
        x, y, w, h = v[2]
        cv2.rectangle(
            img, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), (255, 0, 0), 2
        )

    figure(num=None, figsize=(20, 15))
    plt.imshow(img)
    plt.show()


def process_pred_list(pred_list: List) -> List:
    """
    A function to convert the list of predictions to a list that contains specific data format for bbox visualization.
    Args:
        pred_list: list

    Returns: list [label, confidence, bbox]

    """
    result = []
    for item in pred_list:
        item_replace_special_char = item.replace("\n", "")
        meta_data = item_replace_special_char.split(":")
        label = meta_data[0]
        confidence = float(meta_data[1].split("\t")[0].strip(" ").strip("%")) / 100
        bbox_info = meta_data[2:]
        bbox = get_bbox(bbox_info)
        result.append([label, confidence, bbox])
    return result


def get_bbox(meta_data: List) -> List:
    """
    A function to get the coordinates of the bbox.
    Args:
        meta_data: A list

    Returns: A list of [x, y, w, h]

    """

    x = [v for v in meta_data[0].split(" ") if v][0]
    y = [v for v in meta_data[1].split(" ") if v][0]
    w = [v for v in meta_data[2].split(" ") if v][0]
    h = meta_data[3].replace(")", "").strip(" ")

    return [x, y, w, h]


def show_pred(
    img_name: str,
    pred_file_path: str,
    train_file_path: str,
    img_folder_path: str,
    confidence: Optional[float] = None,
    label: Optional[str] = None,
) -> None:
    """
    A function to visualize predicted bbox in the original images.
    Args:
        img_name: str
        pred_file_path: str
        train_file_path: str
        img_folder_path: str
        confidence: float
        label: str

    Returns: None. Print images in notebook

    """
    pred_result = find_bbox(pred_file_path, train_file_path)
    pred_list = pred_result[img_name]
    result = process_pred_list(pred_list)

    img_path = img_folder_path + img_name

    if confidence:
        result = [v for v in result if v[1] > confidence]

    if label:
        result = [v for v in result if v[0].lower() == label.lower()]
        draw_pic(img_path, result)
    else:
        draw_pic(img_path, result)
