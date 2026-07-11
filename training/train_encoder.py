#!/usr/bin/env python3
"""
train_encoder.py — 小型语义编码器训练脚本

功能:
  训练一个轻量 Sentence Encoder，将文本映射为 64 维语义向量。
  替代 stars.py 中 UnifiedLatentSpace 的随机投影层。

  方法: 对比学习 (Contrastive Learning)
  - 正样本对: 同一文本的两种增强（同义词替换/删除/交换）
  - 负样本对: 不同文本
  - 损失函数: InfoNCE (对比损失)

  训练数据: 内置中文短语库（覆盖安全、日常、技术领域）
  模型: 2层 Transformer Encoder, hidden=64
  参数量: ~10万
  预计 CPU 训练时间: 5-10分钟

  输出: encoder.pt

版权: 完全自训练，零版权风险。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import time
import os
import math

# ─── 超参数 ─────────────────────────────────────────────────

VOCAB_SIZE = 2000       # 词汇表大小
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
FFN_DIM = 128
MAX_SEQ_LEN = 32
OUTPUT_DIM = 64         # 输出向量维度（与 MessageBus feature_dim 对齐）
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 5e-4
TEMPERATURE = 0.07      # 对比学习温度
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 模型 ───────────────────────────────────────────────────

class TinyTransformerEncoder(nn.Module):
    """
    轻量 Transformer Encoder

    架构:
        Input (token indices)  [B, L]
        → Token Embedding       [B, L, D]
        + Positional Encoding   [B, L, D]
        → Transformer Layer x2  [B, L, D]
        → Mean Pooling          [B, D]
        → Linear Projection     [B, OUTPUT_DIM]
        → L2 Normalize

    参数量: ~10万
    """

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                 num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
                 ffn_dim=FFN_DIM, max_seq_len=MAX_SEQ_LEN,
                 output_dim=OUTPUT_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Token 嵌入
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 位置编码
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len)

        # Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=0.1,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出投影
        self.output_proj = nn.Linear(embed_dim, output_dim)

    def forward(self, x, mask=None):
        """
        x: [B, L] token indices
        返回: [B, output_dim] L2 归一化的语义向量
        """
        # 嵌入 + 位置编码
        embedded = self.token_embedding(x)  # [B, L, D]
        embedded = self.pos_encoding(embedded)

        # Transformer 编码
        encoded = self.transformer(embedded, src_key_padding_mask=mask)  # [B, L, D]

        # Mean Pooling
        if mask is not None:
            mask_expanded = (~mask).unsqueeze(-1).float()  # [B, L, 1]
            pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = encoded.mean(dim=1)  # [B, D]

        # 投影 + L2 归一化
        output = self.output_proj(pooled)
        output = F.normalize(output, p=2, dim=-1)

        return output


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ─── 词汇表与数据集 ─────────────────────────────────────────

# 中文短语库（按领域分组）
PHRASE_BANK = {
    "security": [
        "SQL注入攻击", "XSS跨站脚本", "路径遍历漏洞", "命令注入", "SSRF服务端请求伪造",
        "暴力破解密码", "DDoS分布式拒绝服务", "中间人攻击", "权限提升", "后门木马",
        "勒索软件", "钓鱼网站", "社会工程学", "零日漏洞", "缓冲区溢出",
        "SQL injection", "cross site scripting", "path traversal", "command injection",
        "brute force attack", "denial of service", "man in the middle", "privilege escalation",
        "backdoor trojan", "ransomware", "phishing", "social engineering", "zero day exploit",
        "buffer overflow", "remote code execution", "credential stuffing", "session hijacking",
        "insecure deserialization", "broken authentication", "sensitive data exposure",
        "XML external entity", "broken access control", "security misconfiguration",
        "cross site request forgery", "injection flaw", "malware detection",
        "intrusion detection", "firewall rule", "penetration testing", "vulnerability scan",
        "security audit", "threat modeling", "risk assessment", "incident response",
        "digital forensics", "security operations center", "SIEM log analysis",
        "WAF web application firewall", "IDS intrusion detection system",
        "IPS intrusion prevention system", "honeypot trap", "sandbox analysis",
        "reverse engineering", "binary analysis", "memory dump", "packet capture",
        "network traffic analysis", "anomaly detection", "behavioral analysis",
        "signature based detection", "heuristic analysis", "machine learning security",
        "adversarial attack", "model poisoning", "data poisoning", "model evasion",
        " UNION SELECT ", " OR 1=1 ", " DROP TABLE ", " <script>alert", " ../etc/passwd",
        " exec cmd ", " ; cat /etc ", " | whoami ", " admin password ", "127.0.0.1 admin",
        "169.254.169.254 metadata", " file:///etc/ ", " gopher:// ", " dict:// ",
    ],
    "normal": [
        "你好世界", "今天天气怎么样", "帮我查一下", "请问有什么可以帮您", "感谢您的帮助",
        "功能介绍", "用户指南", "常见问题", "联系我们", "关于我们",
        "首页导航", "搜索结果", "商品列表", "订单详情", "个人中心",
        "系统设置", "消息通知", "数据导出", "报表生成", "权限管理",
        "hello world", "how are you", "thank you", "good morning", "nice to meet you",
        "welcome to our website", "please sign in", "create account", "forgot password",
        "search products", "view cart", "checkout", "payment method", "shipping address",
        "order confirmation", "tracking number", "customer service", "return policy",
        "terms of service", "privacy policy", "cookie settings", "language preference",
        "dark mode", "light mode", "auto theme", "font size", "accessibility",
        "dashboard overview", "recent activity", "performance metrics", "usage statistics",
        "project management", "task assignment", "deadline reminder", "team collaboration",
        "file upload", "image preview", "document editor", "spreadsheet formula",
        "calendar event", "meeting schedule", "email notification", "chat message",
        "video player", "audio recording", "screen sharing", "remote desktop",
        "cloud storage", "backup restore", "sync settings", "offline mode",
        "API documentation", "developer tools", "code editor", "version control",
        "unit testing", "debug logging", "performance profiling", "memory usage",
        "database query", "data structure", "algorithm optimization", "time complexity",
    ],
    "emotion": [
        "我很开心", "感到沮丧", "非常激动", "有点紧张", "心情不错",
        "我很生气", "感到失望", "非常惊讶", "有点害怕", "心情平静",
        "我很难过", "感到满足", "非常自豪", "有点尴尬", "心情复杂",
        "i am happy", "feeling sad", "very excited", "a bit nervous", "feeling good",
        "i am angry", "feeling disappointed", "very surprised", "a bit scared", "feeling calm",
        "i am grateful", "feeling lonely", "very confident", "a bit confused", "feeling hopeful",
    ],
    "technical": [
        "Python编程", "JavaScript开发", "数据库设计", "系统架构", "网络安全",
        "机器学习", "深度学习", "自然语言处理", "计算机视觉", "数据挖掘",
        "python programming", "javascript development", "database design", "system architecture",
        "network security", "machine learning", "deep learning", "natural language processing",
        "computer vision", "data mining", "cloud computing", "container orchestration",
        "microservices", "REST API", "GraphQL schema", "WebSocket connection",
        "authentication flow", "authorization middleware", "data validation", "error handling",
        "unit test coverage", "integration testing", "end to end testing", "load testing",
        "CI CD pipeline", "Docker container", "Kubernetes cluster", "Terraform infrastructure",
        "monitoring alerting", "log aggregation", "distributed tracing", "circuit breaker",
        "rate limiting", "load balancing", "auto scaling", "horizontal pod autoscaler",
        "neural network", "convolutional layer", "attention mechanism", "transformer model",
        "gradient descent", "backpropagation", "loss function", "optimizer learning rate",
        "training loop", "validation accuracy", "overfitting prevention", "data augmentation",
        "feature engineering", "dimensionality reduction", "principal component analysis",
        "clustering algorithm", "classification model", "regression analysis", "ensemble method",
    ],
}


class Vocabulary:
    """简单词汇表"""

    def __init__(self):
        self.token2id = {"<PAD>": 0, "<UNK>": 1}
        self.id2token = ["<PAD>", "<UNK>"]
        self._build()

    def _build(self):
        for phrases in PHRASE_BANK.values():
            for phrase in phrases:
                tokens = self._tokenize(phrase)
                for token in tokens:
                    if token not in self.token2id:
                        self.token2id[token] = len(self.id2token)
                        self.id2token.append(token)

        print(f"[词汇表] 大小: {len(self.token2id)}")

    def _tokenize(self, text):
        """简单分词: 中文按字，英文按词"""
        tokens = []
        current_en = ""
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                if current_en:
                    tokens.extend(current_en.lower().split())
                    current_en = ""
                tokens.append(char)
            elif char.isalnum():
                current_en += char
            else:
                if current_en:
                    tokens.extend(current_en.lower().split())
                    current_en = ""
                if char.strip():
                    tokens.append(char)
        if current_en:
            tokens.extend(current_en.lower().split())
        return tokens

    def encode(self, text, max_len=MAX_SEQ_LEN):
        tokens = self._tokenize(text)
        ids = [self.token2id.get(t, 1) for t in tokens][:max_len]
        while len(ids) < max_len:
            ids.append(0)  # pad
        return ids

    def __len__(self):
        return len(self.token2id)


class ContrastiveDataset(Dataset):
    """
    对比学习数据集

    每个样本生成两个增强版本（正样本对）。
    同一批次内的其他样本作为负样本。
    """

    def __init__(self, vocab: Vocabulary, num_samples=3000):
        self.vocab = vocab
        self.samples = []

        all_phrases = []
        for phrases in PHRASE_BANK.values():
            all_phrases.extend(phrases)

        # 生成样本
        for _ in range(num_samples):
            phrase = random.choice(all_phrases)
            aug1 = self._augment(phrase)
            aug2 = self._augment(phrase)
            self.samples.append((aug1, aug2))

        print(f"[数据集] 生成 {len(self.samples)} 个对比对")

    def _augment(self, text):
        """文本增强: 同义词替换/删除/交换"""
        result = text
        # 随机选择增强方式
        aug_type = random.choice(["synonym", "delete", "swap", "noise"])

        if aug_type == "synonym":
            # 简单同义词替换
            synonyms = {
                "你好": "您好", "查询": "查找", "帮助": "协助", "问题": "疑问",
                "hello": "hi", "search": "find", "help": "assist", "problem": "issue",
                "attack": "intrusion", "detection": "identification", "security": "protection",
            }
            for old, new in synonyms.items():
                if old in result.lower():
                    result = result.replace(old, new)
                    break

        elif aug_type == "delete":
            # 随机删除一个字符/词
            if len(result) > 5:
                pos = random.randint(2, len(result) - 3)
                result = result[:pos] + result[pos+1:]

        elif aug_type == "swap":
            # 交换两个相邻字符
            if len(result) > 4:
                pos = random.randint(1, len(result) - 3)
                result = result[:pos] + result[pos+1] + result[pos] + result[pos+2:]

        elif aug_type == "noise":
            # 添加轻微噪声（空格或标点）
            if len(result) > 3:
                pos = random.randint(1, len(result) - 2)
                result = result[:pos] + " " + result[pos:]

        return result

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text1, text2 = self.samples[idx]
        ids1 = self.vocab.encode(text1)
        ids2 = self.vocab.encode(text2)
        return (torch.tensor(ids1, dtype=torch.long),
                torch.tensor(ids2, dtype=torch.long))


# ─── 训练 ───────────────────────────────────────────────────

def info_nce_loss(anchors, positives, temperature=TEMPERATURE):
    """
    InfoNCE 对比损失

    anchors: [B, D]
    positives: [B, D]
    同一 batch 内的其他样本作为负样本
    """
    batch_size = anchors.size(0)

    # 相似度矩阵 [B, B]
    sim = torch.matmul(anchors, positives.T) / temperature

    # 正样本在对角线
    labels = torch.arange(batch_size, device=anchors.device)

    # 对称损失
    loss_a = F.cross_entropy(sim, labels)
    loss_b = F.cross_entropy(sim.T, labels)

    return (loss_a + loss_b) / 2


def train():
    print("=" * 60)
    print("  语义编码器训练 — 对比学习")
    print("=" * 60)
    print()

    # 词汇表
    vocab = Vocabulary()

    # 数据集
    dataset = ContrastiveDataset(vocab, num_samples=3000)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"  训练集: {train_size} 对")
    print(f"  测试集: {test_size} 对")
    print()

    # 模型
    model = TinyTransformerEncoder(vocab_size=len(vocab))
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {param_count:,} ({param_count * 4 / 1024:.1f} KB)")
    print()

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 训练循环
    print("  开始训练:")
    print("  " + "-" * 56)

    best_loss = float('inf')
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_sim_pos = 0  # 正样本对平均相似度
        total_sim_neg = 0  # 负样本对平均相似度
        num_batches = 0
        t0 = time.time()

        for batch_a, batch_b in train_loader:
            optimizer.zero_grad()

            # 编码两个增强版本
            vec_a = model(batch_a)  # [B, D]
            vec_b = model(batch_b)  # [B, D]

            # InfoNCE 损失
            loss = info_nce_loss(vec_a, vec_b)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # 计算正/负样本相似度
            with torch.no_grad():
                pos_sim = F.cosine_similarity(vec_a, vec_b).mean().item()
                # 负样本: 错位配对
                neg_sim = F.cosine_similarity(vec_a, vec_b.roll(1, 0)).mean().item()
                total_sim_pos += pos_sim
                total_sim_neg += neg_sim

        scheduler.step()

        avg_loss = total_loss / num_batches
        avg_pos = total_sim_pos / num_batches
        avg_neg = total_sim_neg / num_batches
        elapsed = time.time() - t0

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"  Epoch {epoch+1:2d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
              f"Pos Sim: {avg_pos:.3f} | Neg Sim: {avg_neg:.3f} | {elapsed:.1f}s")

    print("  " + "-" * 56)
    print(f"  最佳损失: {best_loss:.4f}")
    print(f"  正样本相似度: {avg_pos:.3f} (应接近 1.0)")
    print(f"  负样本相似度: {avg_neg:.3f} (应接近 0.0)")
    print()

    # 保存模型
    os.makedirs("/root/.codebuddy/artifact/star_os/models", exist_ok=True)
    save_path = "/root/.codebuddy/artifact/star_os/models/encoder.pt"

    torch.save({
        "model_state": best_state,
        "model_config": {
            "vocab_size": len(vocab),
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "ffn_dim": FFN_DIM,
            "max_seq_len": MAX_SEQ_LEN,
            "output_dim": OUTPUT_DIM,
        },
        "vocab": vocab.token2id,
        "best_loss": best_loss,
        "param_count": param_count,
    }, save_path)

    print(f"  模型已保存: {save_path}")
    print(f"  文件大小: {os.path.getsize(save_path) / 1024:.1f} KB")
    print()

    # 验证: 语义相似度测试
    print("  语义相似度测试:")
    print("  " + "-" * 56)
    model.load_state_dict(best_state)
    model.eval()

    test_pairs = [
        ("SQL注入攻击", "SQL injection", True),   # 应相似
        ("你好世界", "hello world", True),          # 应相似
        ("SQL注入攻击", "你好世界", False),          # 应不相似
        ("机器学习", "deep learning", True),        # 应相似
        ("暴力破解密码", "brute force attack", True),
        ("暴力破解密码", "功能介绍", False),
        ("你好世界", "功能介绍", False),
        ("系统架构", "system architecture", True),
    ]

    with torch.no_grad():
        for text_a, text_b, expected_similar in test_pairs:
            ids_a = torch.tensor([vocab.encode(text_a)], dtype=torch.long)
            ids_b = torch.tensor([vocab.encode(text_b)], dtype=torch.long)
            vec_a = model(ids_a).squeeze(0)
            vec_b = model(ids_b).squeeze(0)
            sim = F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item()

            status = "✓" if (sim > 0.5) == expected_similar else "✗"
            print(f"    {status} '{text_a}' vs '{text_b}': sim={sim:.3f} "
                  f"(期望{'相似' if expected_similar else '不相似'})")

    print()
    print("=" * 60)
    print("  语义编码器训练完成!")
    print("=" * 60)

    return best_loss


if __name__ == "__main__":
    train()
