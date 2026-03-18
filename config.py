import torch

# -------------------- 数据集路径 --------------------
DATA_ROOT = './data'             #包含 images/ 和 annotations/
TRAIN_TXT = './data/train.txt'   #训练集图像名称列表（每行一个文件名）
VAL_TXT = './data/val.txt'       #验证集图像名称列表
CLASS_NAMES = ['with_mask', 'without_mask']    #类别 0 with_mask, 1 without_mask
NUM_CLASSES = len(CLASS_NAMES)            

# -------------------- 输入尺寸 --------------------
IMAGE_SIZE = 416     #输入图像尺寸
BATCH_SIZE = 16
NUM_WORKS = 4

# -------------------- 模型参数 --------------------
BACKBONE = 'resnet50'       #骨干网络
PRETRAINED = True           #是否使用ImageNet预训练
PREEZE_BACKBONE_BN = True   #是否冻结BN层

# -------------------- 锚框配置 --------------------
#初始锚框（COCO 默认值，可在训练前重新聚类）
ANCHORS = [
    [10, 13], [16, 30], [33, 23],       # 小尺度 stride 8（此需根据数据集调整）
    [30, 61], [62, 45], [59, 119],      # 中尺度 stride 16
    [116, 90], [156, 198], [373, 326]   # 小尺度 stride 32
]
ANCHOR_MASK = [[0, 1, 2], [3, 4, 5], [6, 7, 8]] #每个尺度的锚框索引
STRIDES = [8, 16, 32]   #三个尺度下采样倍数
NUM_ANCHORS_PER_SCALE = 3

# -------------------- 损失权重 --------------------
LAMBDA_COORD = 1.0
LAMBDA_NOOBJ = 0.5
LAMBDA_CLS = 1.0

# -------------------- 训练超参数 --------------------
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
LR_SCHEDULER_STEPS = [60, 80]          # 学习率衰减 epoch
LR_SCHEDULER_GAMMA = 0.1

# -------------------- 设备 --------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------- 保存路径 --------------------
SAVE_DIR = './checkpoints'
LOG_DIR = './logs'
