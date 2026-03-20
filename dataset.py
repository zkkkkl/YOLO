#数据读取


import numpy as np
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET


class MaskDataset(Dataset):
    def __init__(self, data_root, image_list_file, class_names, transform=True):
        self.class_names = class_names
        self.class_to_idx = {name : i for i, name in enumerate(class_names)}

    """
    <annotation>
    <folder>VOC_ROOT</folder>
    <filename>aaaa.jpg</filename>
    <size>
        <width>500</width>
        <height>332</height>
        <depth>3</depth>
    </size>
        <object>
            <name>horse</name>
            <bndbox>
                <xmin>100</xmin>
                <ymin>96</ymin>
                <xmax>355</xmax>
                <ymax>324</ymax>
            </bndbox>
        </object>
    </annotation>
    """
    def parse_voc_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        labels = []
        for obj in root.findall('object'):
            name = obj.find('name')
            if name not in self.class_to_idx:
                continue
            label = self.class_to_idx[name]
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin
            boxes.append([x_center, y_center, width, height])
            labels.append(label)
        return boxes, labels
