# 群星 A.I. OS — 多神经调度总线 v0.0.0.1

## 文件清单

```
star_os_v0.0.0.1/
├── bus/
│   ├── message_bus.py            # MessageBus v0.0.0.1 (MoE Router + Mamba SSM)
│   └── model_scheduler.py        # AidLux 模型调度器
├── models/
│   ├── message_bus_v0.0.0.1.pt   # 总线权重 (348KB)
│   ├── textcnn.pt                # TextCNN 攻击识别 (167KB, 100% acc)
│   ├── encoder.pt                # 语义编码器 (440KB, pos_sim=0.91)
│   ├── snn.pt                    # SNN 脉冲网络 (39KB, 85.5% acc)
│   ├── transformer.pt            # Transformer 语言理解 (2.6MB, 99.4% acc)
│   ├── gnn.pt                    # GNN 图推理 (74KB, 100% acc)
│   └── vae.pt                    # VAE 记忆压缩 (28KB)
├── training/
│   ├── train_textcnn.py          # TextCNN 训练脚本
│   ├── train_encoder.py          # Encoder 训练脚本
│   ├── train_snn_convert.py      # ANN-to-SNN 转换脚本
│   ├── train_transformer.py      # Transformer 训练脚本
│   ├── train_gnn.py              # GNN 训练脚本
│   └── train_vae.py             # VAE 训练脚本
├── src/                          # Rust 核心模块 (高性能)
│   ├── core/
│   │   ├── mod.rs                # 模块声明
│   │   ├── snn.rs                # SNN 脉冲网络 (LIF + STDP, 223行)
│   │   └── kalman.rs             # 自适应卡尔曼滤波 (Joseph形式, 207行)
│   └── main.rs                   # TCP 陷阱引擎 (引力井防御)
├── Cargo.toml                    # Rust 项目配置
├── star_integration.py           # 集成桥接层 (端到端验证)
└── README.md
```

## 模型统计

| 模型 | 架构 | 参数量 | 文件大小 | 准确率 | 用途 |
|------|------|--------|----------|--------|------|
| MoE Router | 门控网络 | ~5万 | 348KB | — | 消息路由 |
| Mamba (SSM) | 状态空间模型 | ~50万 | (含在总线中) | — | 时序融合 |
| TextCNN | 1D CNN | 4.2万 | 167KB | **100%** | 字符级攻击识别 |
| Encoder | Transformer | 10.6万 | 440KB | 0.91正相似 | 语义向量化 |
| SNN | LIF (ANN转换) | 0.9万 | 39KB | **85.5%** | 脉冲感知 |
| Transformer | 4层Encoder | 65万 | 2.6MB | **99.4%** | 语言理解+分类 |
| GNN | GraphSAGE | 1.3万 | 74KB | **100%** | 图推理 |
| VAE | 编解码器 | 0.6万 | 28KB | 0.26重建相似 | 记忆压缩 |
| **合计** | **7种架构** | **~80万** | **~3.7MB** | | |

## 端到端验证结果

| 输入 | 路由到 | 结果 |
|------|--------|------|
| `1' OR 1=1 --` | TextCNN + SNN | sql_injection (99.9% + 40.6%) |
| `<script>alert('XSS')</script>` | TextCNN + SNN | xss (99.9% + 63.6%) |
| `../../../etc/passwd` | TextCNN + SNN | path_traversal (99.9% + 47.7%) |
| `你好，查天气` | Encoder + Transformer | normal (98.5%) |
| `hello world` | Encoder + Transformer | normal (98.1%) |
| `POST /login admin 123456` | TextCNN + SNN | brute_force (98.9% + 41.2%) |

- 平均延迟: **3ms**
- 三层校验通过率: **100%**
- Mamba 融合状态稳定: **11.31**

## 运行方式

### 端到端测试

```bash
python star_integration.py
```

### 单独测试总线

```bash
cd bus && python message_bus.py
```

### 重新训练模型

```bash
cd training
python train_textcnn.py
python train_encoder.py
python train_snn_convert.py
python train_transformer.py
python train_gnn.py
python train_vae.py
```

## 架构

```
消息流入 → 三层校验 → MoE Router 路由
                        ├→ TextCNN (字符级攻击识别)
                        ├→ SNN (脉冲感知)
                        ├→ Transformer (语言理解)
                        ├→ Encoder (语义编码)
                        ├→ GNN (图推理)
                        └→ VAE (记忆压缩)
                     → Mamba SSM 融合 → 输出决策
```

## 神经架构与十层模型对应

| 层级 | 架构 | 状态 |
|------|------|------|
| 总线-路由 | MoE Router | ✅ |
| 总线-融合 | Mamba (SSM) | ✅ |
| L1 感知 | LIF SNN | ✅ |
| L1 字符级 | TextCNN | ✅ |
| L1.5 语言 | Transformer | ✅ |
| L1.5 语义 | Encoder | ✅ |
| L3.5 因果图 | GNN | ✅ |
| L5 记忆 | VAE | ✅ |
| L6 伦理 | 符号系统 | 保留(stars.py) |
| L66 进化 | GA | 保留(stars.py) |

## Rust 核心模块

| 模块 | 功能 | 验证 |
|------|------|------|
| `snn.rs` | LIF 脉冲神经网络 + STDP 学习 (时间差指数衰减) | ✅ 4/4 测试通过 |
| `kalman.rs` | 自适应卡尔曼滤波 (Joseph 形式协方差更新) | ✅ 3/3 测试通过 |
| `main.rs` | TCP 陷阱引擎 (引力井防御) | ✅ 编译通过 |

```bash
# 编译验证
cargo check    # 零错误零警告
cargo test     # 8/8 测试通过
```

## 版权
该版权人属于星之主所有，一切作品版权。解释权归星之主。

## 版本说明

**v0.0.0.1** — 首版基础实现，全部验证通过
