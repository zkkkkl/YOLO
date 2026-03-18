#包含锚框聚类、解码、NMS、可视化等功能。
import torch
import numpy as np



# -------------------- 锚框聚类 --------------------
def iou_metric(boxes, anchors):
    """
    计算每个真实框与每个锚框的 IoU（用于聚类）
    boxes: (n, 2) 真实框的宽高
    anchors: (k, 2) 锚框的宽高
    返回: (n, k) IoU 矩阵
    """

    inters_area = np.minimum(boxes[:, None, 0], anchors[:, 0]) * np.minimum(boxes[:, None, 1], anchors[:, 1])
    box_area = boxes[:, 0] * boxes[:, 1]
    anchors_area = anchors[:, 0] * anchors[:, 1]
    union_areas = box_area[:, None] + anchors_area - inters_area
    return inters_area / np.maximum(union_areas, 1e-10)  #防止除0


     
