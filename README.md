mask_yolo/
├── config.py          # 配置文件（超参数、路径等）
├── dataset.py         # 数据集类（VOC 格式标注解析）
├── model.py           # 基于 ResNet 的 YOLO 检测模型
├── loss.py            # YOLO 损失函数（含锚框匹配）
├── utils.py           # 工具函数（锚框生成、NMS、可视化）
├── train.py           # 训练脚本
├── predict.py         # 推理脚本
└── requirements.txt   # 依赖包