#!/usr/bin/env python3
"""
train_gnn.py — 小型 GNN 图神经网络训练

功能:
  训练 GNN 用于语义网络/因果图推理。
  使用 GraphSAGE 架构，支持节点分类和图级分类。

  模型: 2层 GraphSAGE, hidden=64
  参数量: ~5万
  输出: gnn.pt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random, time, os, math
from collections import defaultdict

SEED = 42
random.seed(SEED); torch.manual_seed(SEED)

HIDDEN_DIM = 64
NUM_LAYERS = 2
OUTPUT_DIM = 7   # 与安全类别对齐
NODE_FEATURE_DIM = 16
EPOCHS = 30
LR = 1e-3

# ─── GraphSAGE 模型 ─────────────────────────────────────────

class GraphSAGELayer(nn.Module):
    """
    GraphSAGE 单层
    h_v = relu(W · CONCAT(h_v, AGGREGATE(h_u for u in N(v))))
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, node_feats, adj):
        """
        node_feats: [N, in_dim]
        adj: [N, N] 邻接矩阵 (0/1)
        """
        # 邻居均值聚合
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
        neigh_agg = torch.matmul(adj, node_feats) / deg  # [N, in_dim]

        # 拼接自身特征 + 邻居聚合
        combined = torch.cat([node_feats, neigh_agg], dim=1)  # [N, in_dim*2]
        out = F.relu(self.linear(combined))
        return out

class GNNModel(nn.Module):
    """
    2层 GraphSAGE + 分类头
    """
    def __init__(self, input_dim=NODE_FEATURE_DIM, hidden_dim=HIDDEN_DIM,
                 output_dim=OUTPUT_DIM, num_layers=NUM_LAYERS):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphSAGELayer(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GraphSAGELayer(hidden_dim, hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim))

    def forward(self, node_feats, adj):
        h = node_feats
        for layer in self.layers:
            h = layer(h, adj)
        return self.classifier(h)  # [N, output_dim]

    def get_node_embedding(self, node_feats, adj):
        """提取节点嵌入（64维，供 MessageBus）"""
        h = node_feats
        for layer in self.layers:
            h = layer(h, adj)
        return h  # [N, hidden_dim]

# ─── 合成图数据 ─────────────────────────────────────────────

CLASS_NAMES = ["normal","sql_injection","xss","path_traversal",
               "command_injection","ssrf","brute_force"]

# 概念节点 (name → category)
NODE_DEFS = [
    # 安全概念
    ("sql_injection", 1), ("xss", 2), ("path_traversal", 3),
    ("command_injection", 4), ("ssrf", 5), ("brute_force", 6),
    ("normal_request", 0), ("http_request", 0), ("parameter", 0),
    ("cookie", 0), ("session", 0), ("token", 0), ("auth", 0),
    ("admin", 0), ("login", 0), ("password", 6),
    ("script", 2), ("alert", 2), ("javascript", 2),
    ("document", 2), ("onload", 2), ("onerror", 2),
    ("union", 1), ("select", 1), ("drop", 1), ("table", 1),
    ("information_schema", 1), ("benchmark", 1), ("sleep", 1),
    ("passwd", 3), ("etc", 3), ("shadow", 3), ("traversal", 3),
    ("exec", 4), ("cmd", 4), ("shell", 4), ("bash", 4),
    ("127.0.0.1", 5), ("localhost", 5), ("metadata", 5),
    ("169.254.169.254", 5), ("file_protocol", 5), ("gopher", 5),
    ("hydra", 6), ("dictionary", 6), ("credential", 6),
    # 通用概念
    ("request", 0), ("response", 0), ("url", 0), ("path", 0),
    ("header", 0), ("body", 0), ("get", 0), ("post", 0),
    ("api", 0), ("json", 0), ("html", 0), ("css", 0),
    ("firewall", 0), ("detection", 0), ("alert_system", 0),
    ("block", 0), ("allow", 0), ("monitor", 0),
]

# 边定义 (concept_a, concept_b) — 语义关联
EDGE_DEFS = [
    ("sql_injection","union"), ("sql_injection","select"), ("sql_injection","drop"),
    ("sql_injection","table"), ("sql_injection","information_schema"),
    ("sql_injection","benchmark"), ("sql_injection","sleep"),
    ("sql_injection","parameter"), ("sql_injection","http_request"),
    ("xss","script"), ("xss","alert"), ("xss","javascript"),
    ("xss","document"), ("xss","onload"), ("xss","onerror"),
    ("xss","html"), ("xss","http_request"),
    ("path_traversal","passwd"), ("path_traversal","etc"),
    ("path_traversal","shadow"), ("path_traversal","traversal"),
    ("path_traversal","path"), ("path_traversal","url"),
    ("command_injection","exec"), ("command_injection","cmd"),
    ("command_injection","shell"), ("command_injection","bash"),
    ("command_injection","parameter"),
    ("ssrf","127.0.0.1"), ("ssrf","localhost"), ("ssrf","metadata"),
    ("ssrf","169.254.169.254"), ("ssrf","file_protocol"), ("ssrf","gopher"),
    ("ssrf","url"),
    ("brute_force","password"), ("brute_force","hydra"),
    ("brute_force","dictionary"), ("brute_force","credential"),
    ("brute_force","login"), ("brute_force","admin"),
    # 通用关联
    ("http_request","request"), ("http_request","response"),
    ("http_request","url"), ("http_request","header"), ("http_request","body"),
    ("http_request","get"), ("http_request","post"),
    ("request","parameter"), ("request","api"),
    ("api","json"), ("api","token"), ("api","auth"),
    ("auth","session"), ("auth","cookie"), ("auth","login"),
    ("login","password"), ("login","admin"),
    ("session","cookie"), ("session","token"),
    ("normal_request","request"), ("normal_request","api"),
    ("normal_request","get"), ("normal_request","post"),
    # 安全系统关联
    ("firewall","block"), ("firewall","allow"),
    ("detection","alert_system"), ("detection","monitor"),
    ("alert_system","block"), ("alert_system","monitor"),
    ("block","sql_injection"), ("block","xss"), ("block","path_traversal"),
    ("block","command_injection"), ("block","ssrf"), ("block","brute_force"),
    ("monitor","sql_injection"), ("monitor","xss"), ("monitor","command_injection"),
]

class GraphDataset:
    """图数据集"""
    def __init__(self):
        self.node_names = [n for n, _ in NODE_DEFS]
        self.node_labels = torch.tensor([l for _, l in NODE_DEFS])
        self.name_to_idx = {n: i for i, n in enumerate(self.node_names)}
        self.num_nodes = len(self.node_names)

        # 邻接矩阵
        self.adj = torch.zeros(self.num_nodes, self.num_nodes)
        for a, b in EDGE_DEFS:
            if a in self.name_to_idx and b in self.name_to_idx:
                i, j = self.name_to_idx[a], self.name_to_idx[b]
                self.adj[i, j] = 1
                self.adj[j, i] = 1  # 无向图

        # 节点特征: one-hot + 随机噪声
        self.node_feats = torch.zeros(self.num_nodes, NODE_FEATURE_DIM)
        for i in range(self.num_nodes):
            # 用节点名的 hash 做特征
            h = hash(self.node_names[i]) % NODE_FEATURE_DIM
            self.node_feats[i, h] = 1.0
            # 加少量噪声
            self.node_feats[i] += torch.randn(NODE_FEATURE_DIM) * 0.1

        # 度数统计
        self.degrees = self.adj.sum(dim=1)
        print(f"[图数据] 节点: {self.num_nodes}, 边: {int(self.adj.sum().item()/2)}, 类别: {len(CLASS_NAMES)}")

    def augment(self):
        """数据增强: 随机添加/删除边"""
        adj = self.adj.clone()
        # 随机添加 10% 的新边
        num_add = max(1, int(self.adj.sum().item() * 0.1))
        for _ in range(num_add):
            i, j = random.randint(0, self.num_nodes-1), random.randint(0, self.num_nodes-1)
            if i != j:
                adj[i, j] = 1; adj[j, i] = 1
        # 随机删除 5% 的边
        edges = (adj > 0).nonzero()
        num_del = max(1, len(edges) // 20)
        for idx in random.sample(range(len(edges)), min(num_del, len(edges))):
            i, j = edges[idx]
            adj[i, j] = 0; adj[j, i] = 0
        return adj

# ─── 训练 ───────────────────────────────────────────────────
def train():
    print("="*60)
    print("  GNN 训练 — GraphSAGE 图推理")
    print("="*60)

    data = GraphDataset()

    # 多次增强作为训练样本
    num_aug = 50
    all_adj = [data.augment() for _ in range(num_aug)]

    model = GNNModel()
    params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {params:,} ({params*4/1024:.1f}KB)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"  增强图: {num_aug} 个")
    print("  "+"-"*56)

    best_acc = 0; best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0; correct = 0; total = 0
        t0 = time.time()

        for adj in all_adj:
            optimizer.zero_grad()
            logits = model(data.node_feats, adj)  # [N, 7]
            loss = criterion(logits, data.node_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(1)
            correct += (pred == data.node_labels).sum().item()
            total += data.num_nodes

        acc = correct / total

        # 测试: 用原始图
        model.eval()
        with torch.no_grad():
            logits = model(data.node_feats, data.adj)
            test_pred = logits.argmax(1)
            test_acc = (test_pred == data.node_labels).float().mean().item()

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"  Epoch {epoch+1:2d}/{EPOCHS} | Loss: {total_loss/num_aug:.4f} | "
              f"Train: {acc:.1%} | Test: {test_acc:.1%} | {time.time()-t0:.1f}s")

    print("  "+"-"*56)
    print(f"  最佳: {best_acc:.1%}")

    # 分类别
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        logits = model(data.node_feats, data.adj)
        pred = logits.argmax(1)
    print("\n  分类别:")
    for i, name in enumerate(CLASS_NAMES):
        mask = data.node_labels == i
        if mask.sum() > 0:
            acc = (pred[mask] == data.node_labels[mask]).float().mean().item()
            print(f"    {name:20s} {acc:.1%} {'█'*int(acc*30)}")

    # 保存
    save_path = "/root/.codebuddy/artifact/star_os/models/gnn.pt"
    torch.save({
        "model_state": best_state,
        "model_config": {
            "input_dim": NODE_FEATURE_DIM, "hidden_dim": HIDDEN_DIM,
            "output_dim": OUTPUT_DIM, "num_layers": NUM_LAYERS,
        },
        "node_names": data.node_names,
        "name_to_idx": data.name_to_idx,
        "adj_matrix": data.adj,
        "node_feats": data.node_feats,
        "class_names": CLASS_NAMES,
        "accuracy": best_acc,
        "param_count": params,
    }, save_path)
    print(f"\n  已保存: {save_path} ({os.path.getsize(save_path)/1024:.1f}KB)")
    print("="*60)
    return best_acc

if __name__ == "__main__":
    train()
