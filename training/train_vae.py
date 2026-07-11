#!/usr/bin/env python3
"""
train_vae.py — 小型 VAE 记忆压缩模型

功能:
  训练 VAE 用于系统状态压缩/记忆巩固。
  编码器: 状态向量 → 潜在空间 (压缩)
  解码器: 潜在空间 → 状态向量 (重建)
  可用于 stars.py 的记忆巩固和世界模型。

  模型: 64→32→16(encoder) →16→32→64(decoder)
  参数量: ~1万
  输出: vae.pt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random, time, os

SEED = 42
random.seed(SEED); torch.manual_seed(SEED)

INPUT_DIM = 64    # 与 MessageBus feature_dim 对齐
LATENT_DIM = 16   # 压缩到 16 维
HIDDEN_DIM = 32
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3

# ─── VAE 模型 ──────────────────────────────────────────────

class VAE(nn.Module):
    """
    变分自编码器 (VAE)

    编码器: x → μ, σ → z (重参数化采样)
    解码器: z → x̂ (重建)

    损失 = 重建损失 + KL 散度
    """
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                 latent_dim=LATENT_DIM):
        super().__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z

    def compress(self, x):
        """压缩: x → z (用于记忆巩固)"""
        mu, _ = self.encode(x)
        return mu

    def reconstruct(self, z):
        """重建: z → x"""
        return self.decode(z)


def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """VAE 损失: 重建 + KL"""
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div, recon_loss.item(), kl_div.item()


# ─── 合成状态数据 ──────────────────────────────────────────

def generate_state_batch(batch_size, num_patterns=7):
    """
    生成模拟系统状态向量

    每种模式对应一个安全类别，模拟 stars.py 的潜在空间状态。
    """
    # 7 种模式中心
    centers = torch.randn(num_patterns, INPUT_DIM)
    centers = F.normalize(centers, dim=1)

    # 每个样本: 随机选一个中心 + 噪声
    batch = torch.zeros(batch_size, INPUT_DIM)
    for i in range(batch_size):
        idx = random.randint(0, num_patterns - 1)
        noise = torch.randn(INPUT_DIM) * 0.3
        batch[i] = centers[idx] + noise
    return batch, centers


# ─── 训练 ───────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  VAE 训练 — 记忆压缩")
    print("=" * 60)

    model = VAE()
    params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {params:,} ({params * 4 / 1024:.1f}KB)")
    print(f"  输入维度: {INPUT_DIM} → 潜在维度: {LATENT_DIM} (压缩比: {INPUT_DIM/LATENT_DIM:.1f}x)")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 生成训练数据
    train_data, centers = generate_state_batch(2000)
    test_data, _ = generate_state_batch(400)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_data),
        batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_data),
        batch_size=BATCH_SIZE, shuffle=False)

    print(f"  训练集: {len(train_data)}, 测试集: {len(test_data)}")
    print("  " + "-" * 56)

    best_loss = float('inf')
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0; total_recon = 0; total_kl = 0; num = 0
        t0 = time.time()

        for (batch,) in train_loader:
            optimizer.zero_grad()
            x_recon, mu, logvar, z = model(batch)
            loss, recon, kl = vae_loss(x_recon, batch, mu, logvar, beta=0.01)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon
            total_kl += kl
            num += len(batch)

        scheduler.step()

        avg_loss = total_loss / (num * INPUT_DIM)
        avg_recon = total_recon / (num * INPUT_DIM)
        avg_kl = total_kl / (num * INPUT_DIM)

        # 测试
        model.eval()
        with torch.no_grad():
            test_loss = 0; test_num = 0
            for (batch,) in test_loader:
                x_recon, mu, logvar, z = model(batch)
                loss, _, _ = vae_loss(x_recon, batch, mu, logvar, beta=0.01)
                test_loss += loss.item()
                test_num += len(batch)
            test_loss /= (test_num * INPUT_DIM)

        if test_loss < best_loss:
            best_loss = test_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{EPOCHS} | Loss: {avg_loss:.4f} "
                  f"(recon: {avg_recon:.4f}, kl: {avg_kl:.4f}) | "
                  f"Test: {test_loss:.4f} | {time.time()-t0:.1f}s")

    print("  " + "-" * 56)
    print(f"  最佳测试损失: {best_loss:.4f}")

    # 验证压缩-重建质量
    model.load_state_dict(best_state)
    model.eval()
    print("\n  压缩-重建验证:")
    with torch.no_grad():
        test_batch, _ = generate_state_batch(100)
        z = model.compress(test_batch)
        recon = model.reconstruct(z)
        mse = F.mse_loss(recon, test_batch).item()
        cos_sim = F.cosine_similarity(recon, test_batch, dim=1).mean().item()
        print(f"    输入维度: {INPUT_DIM} → 压缩维度: {LATENT_DIM}")
        print(f"    重建 MSE: {mse:.4f}")
        print(f"    重建余弦相似度: {cos_sim:.4f}")
        print(f"    压缩比: {INPUT_DIM / LATENT_DIM:.1f}x")

    # 保存
    save_path = "/root/.codebuddy/artifact/star_os/models/vae.pt"
    torch.save({
        "model_state": best_state,
        "model_config": {
            "input_dim": INPUT_DIM, "hidden_dim": HIDDEN_DIM,
            "latent_dim": LATENT_DIM,
        },
        "best_loss": best_loss,
        "recon_mse": mse,
        "recon_cos_sim": cos_sim,
        "param_count": params,
    }, save_path)
    print(f"\n  已保存: {save_path} ({os.path.getsize(save_path)/1024:.1f}KB)")
    print("=" * 60)
    return best_loss

if __name__ == "__main__":
    train()
