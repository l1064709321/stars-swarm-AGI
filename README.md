# 群星 A.I. OS — 多神经调度总线 v0.0.0.1

## 文件清单

```
star_os_v0.0.0.1/
├── bus/
│   ├── message_bus.py          # MessageBus v0.0.0.1 (MoE Router + Mamba)
│   └── model_scheduler.py      # AidLux 模型调度器
├── models/
│   ├── message_bus_v0.0.0.1.pt  # 总线权重 (348KB)
│   ├── textcnn.pt               # TextCNN 攻击识别 (167KB, 100% acc)
│   ├── encoder.pt               # 语义编码器 (440KB, pos_sim=0.91)
│   └── snn.pt                   # SNN 脉冲网络 (39KB, 85.5% acc)
├── training/
│   ├── train_textcnn.py         # TextCNN 训练脚本
│   ├── train_encoder.py         # Encoder 训练脚本
│   └── train_snn_convert.py     # ANN-to-SNN 转换脚本
└── README.md                    # 本文件
```

## 模型统计

| 模型 | 参数量 | 文件大小 | 准确率 | 用途 |
|------|--------|----------|--------|------|
| MoE Router | ~5万 | 348KB | — | 消息路由 |
| Mamba (SSM) | ~50万 | (含在总线中) | — | 时序融合 |
| TextCNN | 4.2万 | 167KB | 100% | 字符级攻击识别 |
| Encoder | 10.6万 | 440KB | 0.91正相似 | 语义向量化 |
| SNN | 0.9万 | 39KB | 85.5% | 脉冲感知 |
| **合计** | **~20万** | **~1MB** | | |

## 运行方式

### 1. 测试 MessageBus

```bash
cd bus
python message_bus.py
```

### 2. 测试 ModelScheduler

```bash
cd bus
python model_scheduler.py
```

### 3. 重新训练模型

```bash
cd training
python train_textcnn.py        # 训练 TextCNN
python train_encoder.py        # 训练 Encoder
python train_snn_convert.py    # 转换 SNN
```

## 在 AidLux 上部署

1. 将整个 `star_os_v0.0.0.1/` 文件夹传到手机 AidLux
2. 确保 AidLux 已安装 PyTorch: `pip install torch`
3. 运行测试: `python bus/message_bus.py`

## 架构

```
消息流入 → 三层校验 → MoE Router 路由
                        ├→ TextCNN (攻击模式)
                        ├→ SNN (脉冲感知)
                        ├→ Encoder (语义编码)
                        ├→ Transformer (语言理解) [待训练]
                        ├→ GNN (图推理) [待训练]
                        └→ VAE (记忆巩固) [待训练]
                     → Mamba 融合 → 输出决策
```

## 版本说明

**v0.0.0.1** — 首版基础实现
- ✅ MoE Router 路由层
- ✅ Mamba SSM 融合层
- ✅ 三层校验 (物理+逻辑+伦理)
- ✅ TextCNN 训练完成
- ✅ Encoder 训练完成
- ✅ SNN 转换完成
- ✅ ModelScheduler
- ⬜ Transformer 训练
- ⬜ GNN 训练
- ⬜ VAE 训练
- ⬜ 集成到 stars.py

## 版权

全部自训练，零版权风险。训练数据基于公开攻击模式知识合成。
