import glob
import xml.etree.ElementTree as ET
from struct import unpack
from typing import List, Tuple

import cv2
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from PIL import Image


def hex_to_float(t: str) -> float:
    """To get coordinate in pixel unit, divided each value in pound by 72 and multiply with 96

    Args:
        t: str

    Returns: float

    """
    return (unpack("!d", bytes.fromhex(t))[0] / 72) * 96


def get_iamge(image_path: str) -> Tuple:
    """ A function to read image and get height and width.

    Args:
        image_path: str

    Returns: img read in cv2, height, width

    """
    img = cv2.imread(image_path)
    h, w, c = img.shape
    return img, h, w


def convert_label_to_str_number(label: str) -> str:
    """
    A function to transform label to number.
    Args:
        label: str

    Returns: str

    """
    label = label.lower()
    if label == "table":
        obj_class = 0
    elif label == "tablebody":
        obj_class = 1
    elif label == "tablecaption":
        obj_class = 2
    elif label == "tablefootnote":
        obj_class = 3
    elif label == "paragraph":
        obj_class = 4
    else:
        return None
    return str(obj_class)


def transform_MarmotBBox_to_YOLO(obj_class, x, y, w, h, imgw, imgh) -> Tuple:
    """A method to transform Marmotbbox to Yolo format.

    Args:
        obj_class: str
        x: str
        y: str
        w: str
        h: str
        imgw: float
        imgh: float

    Returns: obj_class and coordinates for bounding box

    """

    obj_class = obj_class
    x = float(x)
    y = float(y)
    w = float(w)
    h = float(h)
    x_center = (x + (w - x) / 2) / imgw
    y_center = (y + (h - y) / 2) / imgh
    width = (w - x) / imgw
    height = (h - y) / imgh
    return obj_class, x_center, y_center, width, height


def create_training_folder(
    file_in_pos_path: str, file_in_neg_path, xml_in_path, file_out_path
) -> Tuple:
    """
    Search all the files in marmot dataset directory , call function to convert all the files and output them to a new
    directory called img.
    Args:
        file_in_pos_path:
        file_in_neg_path:
        xml_in_path:
        file_out_path:

    Returns: A tuple.
        image_list, text_list, xml_list

    """
    image_list = []
    xml_list = []
    text_list = []

    for filename in glob.glob(file_in_pos_path + "*.bmp"):
        # Convert bmp to jpg.
        img_name = filename.split("/")[-1].strip(".bmp")
        img_path = file_out_path + img_name + ".jpg"
        img = Image.open(filename).save(img_path)
        image_list.append(img_path)

        # Create txt files.
        text_path = file_out_path + img_name + ".txt"
        f = open(file_out_path + img_name + ".txt", "x")
        f.close()
        text_list.append(text_path)

        # Generate xml paths.
        xml_path = xml_in_path + img_name + ".xml"
        xml_list.append(xml_path)

    for filename in glob.glob(file_in_neg_path + "*.bmp"):
        # Convert bmp to jpg.
        img_name = filename.split("/")[-1].strip(".bmp")
        img_path = file_out_path + img_name + ".jpg"
        img = Image.open(filename).save(img_path)

    return image_list, text_list, xml_list


def write_yolo_file(image_list, text_list, xml_list) -> Tuple:
    """ A function to create a txt file that contain class data and bbox information, a specific format of Yolo model.
    It also split dataset into training and testing and return the file names in two different list.
    Assign 100 for test purposes

    Args:
        image_list: list
        text_list: list
        xml_list: list

    Returns:
        train_list: A list of training image file name
        test_list: A list of testing image file name
        data_stats: A stats of the data set

    """
    train_list = []
    test_list = []
    count = 0

    # check the total number of training data available
    data_stats = {}

    for l, img_path in enumerate(image_list):

        count += 1  # TODO
        if count <= 408:
            train_list.append(img_path)
        else:
            test_list.append(img_path)

        img, imgh, imgw = get_iamge(img_path)

        # create element tree object
        tree = ET.parse(xml_list[l])
        # get root element
        root = tree.getroot()

        target_labels = [
            "TableCaption",
            "TableBody",
            "TableFootnote",
            "Paragraph",
            "Table",
        ]

        text_to_save = ""

        for i, label in enumerate(target_labels):

            obj_class = convert_label_to_str_number(label)
            bbox_list = return_bbox(root, label, imgh)

            if bbox_list:  # If a label exists.

                if label in data_stats:
                    data_stats[label] += 1
                else:
                    data_stats[label] = 1

                for j, bbox in enumerate(bbox_list):
                    x, y, w, h = bbox
                    line_data = transform_MarmotBBox_to_YOLO(
                        obj_class, x, y, w, h, imgw, imgh
                    )

                    is_last_line = (j == len(bbox_list) - 1) and (
                        i == len(target_labels) - 1
                    )

                    if is_last_line:
                        line_text = " ".join([str(d) for d in line_data])
                    else:
                        line_text = " ".join([str(d) for d in line_data]) + "\n"
                    text_to_save = text_to_save + line_text

        # Loop through all the labels in an image
        f = open(text_list[l], "w")
        f.write(text_to_save)

    return train_list, test_list, data_stats


def write_train_test(train_list, test_list, path) -> None:
    """
    A function to write training and testing images names to txt files. It create one train.txt and one test.txt.
    Args:
        train_list: A list of training image file name
        test_list: A list of testing image file name
        path: directory to store the txt files.

    Returns: None

    """
    f = open(path + "train.txt", "w")
    f.write("\n".join(["data/" + str(d).strip("./") for d in train_list]))
    f.close()

    f = open(path + "test.txt", "w")
    f.write("\n".join(["data/" + str(d).strip("./") for d in test_list]))
    f.close()


def return_bbox(root, label, imgh) -> List:
    """
    A function to loop through xml files and find the bbox information of target labels.
    Args:
        root: xml root
        label: str
        imgh: float

    Returns: List

    """
    bbox_list = []
    for table in root.findall(".//Composite[@Label='{}']".format(label)):
        x, y, w, h = list(map(hex_to_double, table.get("BBox").split()))
        y = imgh - y
        h = imgh - h
        bbox_list.append([x, y, w, h])
    return bbox_list


def draw_pic(img_path, xml_path, label) -> None:
    """
    A function to visualize the bbox in an image.
    Args:
        img_path: str
        xml_path: str
        label: str

    Returns: None

    """
    img, imgh, imgw = get_iamge(img_path)

    # create element tree object
    tree = ET.parse(xml_path)
    # get root element
    root = tree.getroot()

    if label:
        target_labels = [label]
    else:
        target_labels = [
            "Table",
            "TableCaption",
            "TableBody",
            "TableFootnote",
            "Paragraph",
        ]

    for i, label in enumerate(target_labels):
        bbox_list = return_bbox(root, label, imgh)
        for bbox in bbox_list:
            x, y, w, h = bbox
            cv2.rectangle(img, (int(x), int(y)), (int(w), int(h)), (255, 0, 0), 2)

    figure(num=None, figsize=(20, 15), dpi=80, facecolor="w", edgecolor="k")
    plt.imshow(img)
    plt.show()


"""
Sample code on how to use the utils.

file_in_pos_path = "./marmot_dataset/Positive/Raw/"
file_in_neg_path = "./marmot_dataset/Negative/Raw/"
file_out_path = "./obj/"
xml_in_path = "./marmot_dataset/Positive/Labeled/"

image_list, text_list, xml_list = create_training_folder(
    file_in_pos_path, file_in_neg_path, xml_in_path, file_out_path
)

train_list, test_list, data_stats = write_yolo_file(image_list, text_list, xml_list)

out_path = "./data/"
write_train_test(train_list, test_list, out_path)
"""
