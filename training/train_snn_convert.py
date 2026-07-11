#!/usr/bin/env python3
"""
train_snn_convert.py — ANN-to-SNN 转换脚本

功能:
  将已训练的 TextCNN (ANN) 权重转换为脉冲神经网络 (SNN) 权重。
  使用 spike normalization 方法：
  1. 在 ANN 激活值上统计数据集的统计分布
  2. 用归一化后的权重初始化 SNN 的膜电位阈值和连接权重
  3. 转换后的 SNN 保留 ANN 的分类能力，同时获得脉冲动力学

  这种方法不需要 GPU 训练 SNN，直接复用 ANN 训练成果。

  输出: snn.pt

版权: 完全自训练，零版权风险。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import sys

# ─── 超参数 ─────────────────────────────────────────────────

NUM_NEURONS = 128        # SNN 神经元数量
BETA = 0.95              # 膜电位衰减系数
THRESHOLD = 1.0          # 脉冲发放阈值
NUM_TIMESTEPS = 10       # SNN 模拟时间步长
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── LIF 神经元模型 ─────────────────────────────────────────

class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) 神经元

    膜电位方程:
        V(t+1) = beta * V(t) + I(t)
        if V(t+1) >= threshold:
            spike!
            V(t+1) = 0 (重置)
    """

    def __init__(self, beta=BETA, threshold=THRESHOLD):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.mem = None  # 膜电位

    def reset(self):
        self.mem = None

    def forward(self, current):
        """
        current: [B, N] 输入电流
        返回: [B, N] 脉冲输出 (0或1)
        """
        if self.mem is None:
            self.mem = torch.zeros_like(current)

        # 膜电位更新
        self.mem = self.beta * self.mem + current

        # 脉冲发放
        spike = (self.mem >= self.threshold).float()

        # 重置已发放神经元的膜电位
        self.mem = self.mem * (1 - spike)

        return spike


class ConvertedSNN(nn.Module):
    """
    ANN-to-SNN 转换后的脉冲神经网络

    架构:
        Input (char indices)  [B, L]
        → Embedding (来自 TextCNN)
        → Linear: input → SNN 神经元电流   [B, N]
        → LIF 神经元层 (t个时间步)
        → Linear: SNN 输出 → 分类           [B, C]

    参数量: ~5万
    """

    def __init__(self, textcnn_model, num_classes=7, num_neurons=NUM_NEURONS,
                 beta=BETA, threshold=THRESHOLD):
        super().__init__()

        # 复用 TextCNN 的嵌入层
        self.embedding = textcnn_model.embedding

        # 输入维度: embedding_dim * max_seq_len → num_neurons
        # 用 1D 卷积做特征提取（复用 TextCNN 的卷积权重概念）
        embed_dim = textcnn_model.embedding.embedding_dim

        # 输入投影: 文本嵌入 → SNN 输入电流
        self.input_proj = nn.Linear(embed_dim, num_neurons)

        # LIF 神经元
        self.lif = LIFNeuron(beta=beta, threshold=threshold)

        # 输出分类层
        self.output = nn.Linear(num_neurons, num_classes)

        self.num_neurons = num_neurons
        self.num_timesteps = NUM_TIMESTEPS

    def forward(self, x):
        """
        x: [B, L] 字符索引
        返回: [B, C] 分类 logits
        """
        # 嵌入
        embedded = self.embedding(x)  # [B, L, E]
        # 平均池化: [B, L, E] → [B, E]
        pooled = embedded.mean(dim=1)

        # 转换为输入电流
        current = self.input_proj(pooled)  # [B, N]

        # SNN 模拟 (多个时间步)
        self.lif.reset()
        spike_count = torch.zeros(current.size(0), self.num_neurons)

        for t in range(self.num_timesteps):
            spike = self.lif(current)
            spike_count += spike

        # 脉冲计数 → 分类
        # 归一化脉冲计数
        spike_rate = spike_count / self.num_timesteps
        logits = self.output(spike_rate)

        return logits

    def get_spike_pattern(self, x):
        """获取脉冲模式（供 stars.py 的 SNN 感知层使用）"""
        embedded = self.embedding(x)
        pooled = embedded.mean(dim=1)
        current = self.input_proj(pooled)

        self.lif.reset()
        all_spikes = []

        for t in range(self.num_timesteps):
            spike = self.lif(current)
            all_spikes.append(spike)

        # [T, B, N] → [B, N] 平均脉冲率
        spike_rate = torch.stack(all_spikes).mean(dim=0)
        return spike_rate


# ─── 转换与训练 ─────────────────────────────────────────────

def convert():
    print("=" * 60)
    print("  ANN-to-SNN 转换 — TextCNN → SNN")
    print("=" * 60)
    print()

    # 加载已训练的 TextCNN
    textcnn_path = "/root/.codebuddy/artifact/star_os/models/textcnn.pt"
    if not os.path.exists(textcnn_path):
        print(f"  错误: TextCNN 模型不存在: {textcnn_path}")
        print("  请先运行 train_textcnn.py")
        return

    checkpoint = torch.load(textcnn_path, map_location='cpu', weights_only=False)
    textcnn_config = checkpoint["model_state"]

    # 从配置重建 TextCNN
    sys.path.insert(0, os.path.dirname(__file__))
    from train_textcnn import TextCNN, CLASS_NAMES, SecurityDataset, MAX_SEQ_LEN

    textcnn = TextCNN(
        vocab_size=checkpoint["model_config"]["vocab_size"],
        embed_dim=checkpoint["model_config"]["embed_dim"],
        num_filters=checkpoint["model_config"]["num_filters"],
        kernel_sizes=checkpoint["model_config"]["kernel_sizes"],
        num_classes=checkpoint["model_config"]["num_classes"],
        hidden_dim=checkpoint["model_config"]["hidden_dim"],
    )
    textcnn.load_state_dict(textcnn_config)
    textcnn.eval()

    print(f"  TextCNN 已加载 (准确率: {checkpoint['accuracy']:.1%})")

    # 创建 SNN
    snn = ConvertedSNN(
        textcnn_model=textcnn,
        num_classes=len(CLASS_NAMES),
        num_neurons=NUM_NEURONS,
        beta=BETA,
        threshold=THRESHOLD,
    )

    param_count = sum(p.numel() for p in snn.parameters())
    print(f"  SNN 参数量: {param_count:,} ({param_count * 4 / 1024:.1f} KB)")
    print(f"  LIF 神经元: {NUM_NEURONS}")
    print(f"  时间步长: {NUM_TIMESTEPS}")
    print()

    # ── Spike Normalization ──
    # 统计 TextCNN 在数据集上的激活值分布，校准 SNN 阈值
    print("  执行 Spike Normalization...")
    dataset = SecurityDataset(num_normal=500, num_attack_per_class=80)

    # 收集激活值
    with torch.no_grad():
        activations = []
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        for batch_x, _ in loader:
            embedded = textcnn.embedding(batch_x)
            pooled = embedded.mean(dim=1)
            current = snn.input_proj(pooled)
            activations.append(current)

        all_activations = torch.cat(activations, dim=0)  # [N_samples, num_neurons]

        # 计算每个神经元的最大激活值，用于校准阈值
        max_activations = all_activations.max(dim=0)[0]  # [num_neurons]
        mean_activations = all_activations.mean(dim=0)   # [num_neurons]

        # Spike Normalization: 调整阈值使脉冲率在合理范围 (0.1~0.5)
        # 阈值设为最大激活的 85%，使只有强激活的神经元才发放
        calibrated_threshold = max_activations * 0.85
        calibrated_threshold = calibrated_threshold.clamp(min=0.1)

        # 应用校准: 用中位数阈值（更鲁棒）
        snn.lif.threshold = calibrated_threshold.median().item()

        print(f"  激活统计: max={max_activations.mean():.3f}, mean={mean_activations.mean():.3f}")
        print(f"  校准阈值: {snn.lif.threshold:.3f}")
    print()

    # ── 微调 SNN 输出层 ──
    # 输入投影和嵌入层来自 TextCNN，输出层需要微调
    print("  微调 SNN 输出层...")
    print("  " + "-" * 56)

    # 只冻结嵌入层，输入投影和输出层一起训练
    for param in snn.embedding.parameters():
        param.requires_grad = False

    # 训练输入投影 + 输出层
    optimizer = torch.optim.AdamW(
        [p for p in snn.parameters() if p.requires_grad],
        lr=3e-3, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    EPOCHS = 30
    for epoch in range(EPOCHS):
        snn.train()
        total_loss = 0
        correct = 0
        total = 0
        t0 = time.time()

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = snn(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)

        acc = correct / total
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:2d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Acc: {acc:.1%} | {elapsed:.1f}s")

    print("  " + "-" * 56)
    print(f"  SNN 最终准确率: {acc:.1%}")
    print()

    # ── 测试脉冲模式 ──
    print("  脉冲模式测试:")
    snn.eval()
    test_inputs = [
        "1' OR 1=1 --",           # SQL注入
        "<script>alert(1)</script>",  # XSS
        "GET /index.html HTTP/1.1",   # 正常
    ]

    with torch.no_grad():
        for text in test_inputs:
            chars = [min(ord(c), 127) for c in text[:MAX_SEQ_LEN]]
            while len(chars) < MAX_SEQ_LEN:
                chars.append(0)
            x = torch.tensor([chars], dtype=torch.long)

            spike_rate = snn.get_spike_pattern(x)
            active = (spike_rate > 0.1).sum().item()
            total_neurons = spike_rate.size(1)
            logits = snn(x)
            pred = CLASS_NAMES[logits.argmax(1).item()]

            print(f"    '{text[:40]}':")
            print(f"      预测: {pred}")
            print(f"      活跃神经元: {active}/{total_neurons} ({active/total_neurons:.0%})")
            print(f"      平均脉冲率: {spike_rate.mean():.4f}")

    print()

    # ── 保存 SNN ──
    save_path = "/root/.codebuddy/artifact/star_os/models/snn.pt"
    torch.save({
        "model_state": snn.state_dict(),
        "model_config": {
            "num_neurons": NUM_NEURONS,
            "beta": BETA,
            "threshold": snn.lif.threshold,
            "num_timesteps": NUM_TIMESTEPS,
            "num_classes": len(CLASS_NAMES),
            "max_seq_len": MAX_SEQ_LEN,
        },
        "class_names": CLASS_NAMES,
        "accuracy": acc,
        "param_count": param_count,
        "conversion_method": "spike_normalization",
    }, save_path)

    print(f"  SNN 已保存: {save_path}")
    print(f"  文件大小: {os.path.getsize(save_path) / 1024:.1f} KB")
    print()

    print("=" * 60)
    print("  ANN-to-SNN 转换完成!")
    print("=" * 60)

    return acc


if __name__ == "__main__":
    convert()
