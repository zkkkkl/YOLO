#包含锚框聚类、解码、NMS、可视化等功能。
import cv2
import torch
import numpy as np
from config import IMAGE_SIZE, ANCHOR_MASK, ANCHORS, STRIDES, NUM_CLASSES


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

def kmeans_plusplus_init(data, k, random_seed=None):
    """
    使用K-means++聚类初始化
    data: (n, 2) 所有真实框宽高
    k: 锚框数量
    return: (k, 2)
    """
    data = np.array(data)
    n = len(data)
    anchors = []
    if random_seed is not None:
        np.random.seed(random_seed)
    first_idx = np.random.choice(n)
    anchors.append(data[first_idx])

    for _ in range(1, k):
        #计算所有点至中心点距离
        iou = iou_metric(data, np.array(anchors))
        distance = 1 - np.argmax(iou, axis=1)
        dist_sum = np.sum(distance)
        if dist_sum == 0:
            prob = np.ones(n) / n
        else:
            prob = distance / dist_sum
        next_idx = np.random.choice(n, p=prob)
        anchors.append(data[next_idx])
    
    return anchors
        



def kmeans_anchors(dataset, num_anchors=9, iterations=100):
    """
    使用k-means聚类生成锚框
    dataset: 返回 (image, boxes, labels) 的数据集 (需能遍历获得所有框的宽高)
    num_anchors: 锚框数量
    iterations: 迭代次数
    return: 聚类后的锚框(num_anchors, 2)，按面积排序
    """
    #获取所有宽高
    all_wh = []
    for i in range(len(dataset)):
        _, boxes, _ = dataset[i]
        #把归一化宽高转为像素
        w = boxes[2] * IMAGE_SIZE
        h = boxes[3] * IMAGE_SIZE
        all_wh.append([w, h])
    all_wh = np.array(all_wh)
    #K-means++选择anchors
    #indices = np.random.choice(len(all_wh), num_anchors, replace=False)
    #anchors = all_wh[indices]
    anchors = kmeans_plusplus_init(all_wh, num_anchors)

    for _ in range(iterations):
        #计算距离
        iou = iou_metric(all_wh, np.array(anchors))
        distance = 1 - iou
        #分配每个框到最近的锚框
        labels = np.argmin(distance, axis=1)
        new_anchors = []
        for i in range(num_anchors):
            cluster = all_wh[labels == i]
            if len(cluster) > 0:
                new_anchors.append(np.median(cluster, axis=0))
            else:
                new_anchors.append(anchors[i])
        anchors = np.array(new_anchors)

    #按面积排序
    areas = anchors[:, 0] * anchors[:, 1]
    anchors = anchors[np.argsort(areas)]
    return anchors.tolist()

def decode_outputs(outputs, conf_thresh=0.5, nms_thresh=0.5):
    """
    将网络输出解码为边界框
    outputs：三个尺度的输出列表，每个形状(batch, num_anchors*(5+num_classes), grid, grid)
    return: 每张图像的框列表，每个框格式[x1, y1, x2, y2, score, class_id] （坐标归一化）
    """

    batch = outputs[0].size[0]
    all_boxes = [[] for _ in range(batch)]

    for i, output in enumerate(outputs):
        stride = STRIDES[i]
        anchors = torch.tensor(ANCHORS[ANCHOR_MASK[i][0] : ANCHORS[i][-1] + 1]).to(device=output.device)
        num_anchors = len(anchors)
        batch, _, grid_h, grid_w = output.shape
        output = output.view(batch, num_anchors, 5+NUM_CLASSES, grid_h, grid_w)
        output = output.permute(0, 1, 3, 4, 2).contiguous()

        for b in range(batch):
            for a in range(num_anchors):
                for iy in range(grid_h):
                    for ix in range(grid_w):
                        pred = output[b, a, iy, ix] #(5+num_classes) x,y,w,h,含有目标概率,num_classes
                        conf = torch.sigmoid(pred[4]).item()
                        if conf < conf_thresh:
                            continue
                        cls_probs = torch.softmax(pred[5:], dim=0)
                        cls_id = torch.argmax(cls_probs).item()
                        score = conf * cls_probs[cls_id]

                        #解码坐标
                        x = (torch.sigmoid(pred[0]) * ix) * stride
                        y = (torch.sigmoid(pred[1]) * iy) * stride
                        w = torch.exp(pred[2]) * anchors[a][0]
                        h = torch.exp(pred[3]) * anchors[a][1]

                        #转换为归一化坐标
                        x1 = (x - w/2) / IMAGE_SIZE
                        y1 = (y - w/2) / IMAGE_SIZE
                        x2 = (x + w/2) / IMAGE_SIZE
                        y2 = (y + w/2) / IMAGE_SIZE

                        all_boxes[b].append([x1, y1, x2, y2, score, cls_id])
    #NMS
    for b in range(batch):
        all_boxes[b] = nms(all_boxes[b], nms_thresh)

    return all_boxes

def nms(boxes, iou_thresh):
    if len(boxes) == 0:
        return []
    keep = []
    #按类别分组
    unique_cls = np.unique(boxes[:, 5])
    for cls in unique_cls:
        cls_mask = boxes[:, 4] == cls
        cls_boxes = boxes[cls_mask]
        #排序
        cls_boxes = sorted(cls_boxes, key=lambda x : x[4], reverse=True)
        
        idx = 0
        while idx < len(cls_boxes):
            best = cls_boxes[idx]
            keep.append(best)
            if idx == len(cls_boxes) -1:
                break
            ious = iou_vectorized(cls_boxes[idx:, :4], best)
            iou_mask = ious <= iou_thresh
            cls_boxes = cls_boxes[iou_mask]
            idx = 0
        
    return np.array(keep)

def iou_vectorized(box1, box2):
    x1 = max(box1[:, 0], box2[0])
    y1 = max(box1[:, 1], box2[1])
    x2 = min(box1[:, 2], box2[2])
    y2 = min(box1[:, 3], box2[3])

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area + 10e-16
    return inter_area / union

def visualize(image, boxes, class_names):
    """绘制检测结果，image为BGR格式的numpy数组"""
    img = image.copy()
    h, w, _ = image.shpe
    for box in boxes:
        x1, y1, x2, y2, score, cls_id = box
        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f'{class_names[cls_id]}:{score:.2f}', (x1 - 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


