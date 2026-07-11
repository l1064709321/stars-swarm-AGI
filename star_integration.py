#!/usr/bin/env python3
"""
star_integration.py — 将训练好的模型桥接到 MessageBus

功能:
  1. 加载所有训练好的模型 (TextCNN, Encoder, SNN, Transformer, GNN, VAE)
  2. 将每个模型包装为 MessageBus 专家节点
  3. 端到端验证: 输入文本 → 总线路由 → 多专家处理 → Mamba融合 → 输出

  这是 stars.py 集成的桥梁层。
  在 stars.py 中 import 本文件，调用 create_integrated_bus() 即可。
"""

import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.join(BASE_DIR, "training")
MODEL_DIR = os.path.join(BASE_DIR, "models")
BUS_DIR = os.path.join(BASE_DIR, "bus")

sys.path.insert(0, TRAINING_DIR)
sys.path.insert(0, BUS_DIR)

from message_bus import MessageBus, BusMessage, MessageType, MessagePriority

# ─── 模型加载器 ─────────────────────────────────────────────

class ModelLoader:
    """统一模型加载器"""

    @staticmethod
    def load_textcnn(path):
        from train_textcnn import TextCNN
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        cfg = ckpt["model_config"]
        model = TextCNN(
            vocab_size=cfg["vocab_size"], embed_dim=cfg["embed_dim"],
            num_filters=cfg["num_filters"], kernel_sizes=cfg["kernel_sizes"],
            num_classes=cfg["num_classes"], hidden_dim=cfg["hidden_dim"],
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, ckpt

    @staticmethod
    def load_encoder(path):
        from train_encoder import TinyTransformerEncoder
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        cfg = ckpt["model_config"]
        model = TinyTransformerEncoder(
            vocab_size=cfg["vocab_size"], embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"], num_layers=cfg["num_layers"],
            ffn_dim=cfg["ffn_dim"], max_seq_len=cfg["max_seq_len"],
            output_dim=cfg["output_dim"],
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, ckpt

    @staticmethod
    def load_snn(path):
        from train_snn_convert import ConvertedSNN
        from train_textcnn import TextCNN
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        cfg = ckpt["model_config"]

        # SNN 需要 TextCNN 的嵌入层
        textcnn_path = os.path.join(MODEL_DIR, "textcnn.pt")
        tcnn_ckpt = torch.load(textcnn_path, map_location='cpu', weights_only=False)
        tcnn_cfg = tcnn_ckpt["model_config"]
        textcnn = TextCNN(
            vocab_size=tcnn_cfg["vocab_size"], embed_dim=tcnn_cfg["embed_dim"],
            num_filters=tcnn_cfg["num_filters"], kernel_sizes=tcnn_cfg["kernel_sizes"],
            num_classes=tcnn_cfg["num_classes"], hidden_dim=tcnn_cfg["hidden_dim"],
        )
        textcnn.load_state_dict(tcnn_ckpt["model_state"])

        snn = ConvertedSNN(
            textcnn_model=textcnn, num_classes=cfg["num_classes"],
            num_neurons=cfg["num_neurons"], beta=cfg["beta"],
            threshold=cfg["threshold"],
        )
        snn.load_state_dict(ckpt["model_state"])
        snn.eval()
        return snn, ckpt

    @staticmethod
    def load_transformer(path):
        from train_transformer import SmallTransformer, Vocab
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        cfg = ckpt["model_config"]
        model = SmallTransformer(
            vocab_size=cfg["vocab_size"], embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"], num_layers=cfg["num_layers"],
            ffn_dim=cfg["ffn_dim"], max_seq_len=cfg["max_seq_len"],
            num_classes=cfg["num_classes"],
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, ckpt

    @staticmethod
    def load_gnn(path):
        from train_gnn import GNNModel
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        cfg = ckpt["model_config"]
        model = GNNModel(
            input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"],
            output_dim=cfg["output_dim"], num_layers=cfg["num_layers"],
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, ckpt

    @staticmethod
    def load_vae(path):
        from train_vae import VAE
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        cfg = ckpt["model_config"]
        model = VAE(
            input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"],
            latent_dim=cfg["latent_dim"],
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, ckpt


# ─── 专家包装器 ─────────────────────────────────────────────

class TextCNNExpert:
    """TextCNN 专家: 字符级攻击识别"""
    CLASS_NAMES = ["normal","sql_injection","xss","path_traversal",
                   "command_injection","ssrf","brute_force"]

    def __init__(self, model, max_seq_len=128):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, msg: BusMessage):
        text = msg.payload if isinstance(msg.payload, str) else str(msg.payload)
        chars = [min(ord(c), 127) for c in text[:self.max_seq_len]]
        while len(chars) < self.max_seq_len:
            chars.append(0)
        x = torch.tensor([chars], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(x)
            feat = self.model.get_feature_vector(x)
            prob = F.softmax(logits, dim=1)
        pred = prob.argmax(1).item()
        return {
            "label": self.CLASS_NAMES[pred],
            "confidence": prob[0][pred].item(),
            "feature": feat.squeeze(0).tolist(),
            "source": "textcnn",
        }


class EncoderExpert:
    """语义编码器专家: 文本→语义向量"""
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab

    def __call__(self, msg: BusMessage):
        text = msg.payload if isinstance(msg.payload, str) else str(msg.payload)
        ids = self.vocab.encode(text)
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            vec = self.model(x).squeeze(0)
        return {
            "vector": vec.tolist(),
            "norm": float(vec.norm()),
            "source": "encoder",
        }


class SNNExpert:
    """SNN 专家: 脉冲感知"""
    CLASS_NAMES = ["normal","sql_injection","xss","path_traversal",
                   "command_injection","ssrf","brute_force"]

    def __init__(self, model, max_seq_len=128):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, msg: BusMessage):
        text = msg.payload if isinstance(msg.payload, str) else str(msg.payload)
        chars = [min(ord(c), 127) for c in text[:self.max_seq_len]]
        while len(chars) < self.max_seq_len:
            chars.append(0)
        x = torch.tensor([chars], dtype=torch.long)
        with torch.no_grad():
            spike_rate = self.model.get_spike_pattern(x)
            logits = self.model(x)
            prob = F.softmax(logits, dim=1)
        pred = prob.argmax(1).item()
        return {
            "label": self.CLASS_NAMES[pred],
            "confidence": prob[0][pred].item(),
            "spike_rate": spike_rate.mean().item(),
            "active_neurons": int((spike_rate > 0.1).sum().item()),
            "source": "snn",
        }


class TransformerExpert:
    """Transformer 专家: 语言理解+意图分类"""
    CLASS_NAMES = ["normal","sql_injection","xss","path_traversal",
                   "command_injection","ssrf","brute_force"]

    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab

    def __call__(self, msg: BusMessage):
        text = msg.payload if isinstance(msg.payload, str) else str(msg.payload)
        ids = self.vocab.encode(text)
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(x, mode="classify")
            feat = self.model(x, mode="feature")
            prob = F.softmax(logits, dim=1)
        pred = prob.argmax(1).item()
        return {
            "label": self.CLASS_NAMES[pred],
            "confidence": prob[0][pred].item(),
            "feature": feat.squeeze(0).tolist(),
            "source": "transformer",
        }


class GNNExpert:
    """GNN 专家: 图推理"""
    def __init__(self, model, adj, node_feats, name_to_idx):
        self.model = model
        self.adj = adj
        self.node_feats = node_feats
        self.name_to_idx = name_to_idx

    def __call__(self, msg: BusMessage):
        text = msg.payload if isinstance(msg.payload, str) else str(msg.payload)
        # 找到文本中匹配的图节点
        matched = []
        text_lower = text.lower()
        for name, idx in self.name_to_idx.items():
            if name in text_lower:
                matched.append(idx)

        if not matched:
            return {"matched_nodes": 0, "source": "gnn"}

        with torch.no_grad():
            logits = self.model(self.node_feats, self.adj)
            node_preds = logits.argmax(1)

        # 返回匹配节点的预测
        results = []
        for idx in matched[:5]:
            results.append({
                "node": list(self.name_to_idx.keys())[idx],
                "predicted_class": node_preds[idx].item(),
            })
        return {"matched_nodes": len(matched), "predictions": results, "source": "gnn"}


class VAEExpert:
    """VAE 专家: 状态压缩"""
    def __init__(self, model):
        self.model = model

    def __call__(self, msg: BusMessage):
        # 如果有 feature_vector，压缩它
        if msg.feature_vector is not None:
            vec = msg.feature_vector
            if not isinstance(vec, torch.Tensor):
                vec = torch.tensor(vec, dtype=torch.float32)
            vec = vec[:64].unsqueeze(0) if vec.numel() >= 64 else torch.zeros(1, 64)
            with torch.no_grad():
                z = self.model.compress(vec)
                recon = self.model.reconstruct(z)
            return {
                "latent": z.squeeze(0).tolist(),
                "recon_error": float(F.mse_loss(recon, vec)),
                "source": "vae",
            }
        return {"source": "vae", "status": "no_input"}


# ─── 集成总线创建 ───────────────────────────────────────────

def create_integrated_bus(verbose=True):
    """
    创建集成总线: 加载所有模型 + 注册为专家

    返回: (bus, experts_dict)
    """
    print("=" * 60)
    print("  Star A.I. OS — 集成总线初始化")
    print("=" * 60)

    bus = MessageBus(num_experts=8, state_dim=128, top_k=3)
    experts = {}

    # 1. TextCNN
    path = os.path.join(MODEL_DIR, "textcnn.pt")
    if os.path.exists(path):
        model, ckpt = ModelLoader.load_textcnn(path)
        expert = TextCNNExpert(model, ckpt["model_config"]["max_seq_len"])
        bus.register_expert("textcnn", expert,
                            msg_types=[MessageType.SECURITY, MessageType.PATTERN],
                            description="字符级攻击识别")
        experts["textcnn"] = expert
        if verbose: print(f"  ✅ TextCNN 加载 (acc={ckpt['accuracy']:.1%})")

    # 2. Encoder
    path = os.path.join(MODEL_DIR, "encoder.pt")
    if os.path.exists(path):
        model, ckpt = ModelLoader.load_encoder(path)
        vocab_obj = type('V', (), {'encode': lambda self, text: ckpt["vocab"].get(text.strip().split()[0] if text.strip() else "", 1) and [ckpt["vocab"].get(t, 1) for t in text][:32]})()
        # 简化: 直接用 vocab dict 做 encode
        class SimpleVocab:
            def __init__(self, t2i):
                self.t2i = t2i
            def encode(self, text, max_len=32):
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
                ids = [self.t2i.get(t, 1) for t in tokens][:max_len]
                ids = [2] + ids + [3]
                while len(ids) < max_len: ids.append(0)
                return ids

        vocab = SimpleVocab(ckpt["vocab"])
        expert = EncoderExpert(model, vocab)
        bus.register_expert("encoder", expert,
                            msg_types=[MessageType.LANGUAGE, MessageType.MEMORY],
                            description="语义编码")
        experts["encoder"] = expert
        if verbose: print(f"  ✅ Encoder 加载 (loss={ckpt['best_loss']:.4f})")

    # 3. SNN
    path = os.path.join(MODEL_DIR, "snn.pt")
    if os.path.exists(path):
        model, ckpt = ModelLoader.load_snn(path)
        expert = SNNExpert(model, ckpt["model_config"]["max_seq_len"])
        bus.register_expert("snn", expert,
                            msg_types=[MessageType.SECURITY],
                            description="脉冲感知")
        experts["snn"] = expert
        if verbose: print(f"  ✅ SNN 加载 (acc={ckpt['accuracy']:.1%})")

    # 4. Transformer
    path = os.path.join(MODEL_DIR, "transformer.pt")
    if os.path.exists(path):
        model, ckpt = ModelLoader.load_transformer(path)
        class TVocab:
            def __init__(self, t2i, max_len=48):
                self.t2i = t2i; self.max_len = max_len
            def encode(self, text):
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
                ids = [self.t2i.get(t, 1) for t in tokens][:self.max_len]
                ids = [2] + ids + [3]
                while len(ids) < self.max_len: ids.append(0)
                return ids
        vocab = TVocab(ckpt["vocab"])
        expert = TransformerExpert(model, vocab)
        bus.register_expert("transformer", expert,
                            msg_types=[MessageType.LANGUAGE],
                            description="语言理解")
        experts["transformer"] = expert
        if verbose: print(f"  ✅ Transformer 加载 (acc={ckpt['accuracy']:.1%})")

    # 5. GNN
    path = os.path.join(MODEL_DIR, "gnn.pt")
    if os.path.exists(path):
        model, ckpt = ModelLoader.load_gnn(path)
        expert = GNNExpert(model, ckpt["adj_matrix"], ckpt["node_feats"], ckpt["name_to_idx"])
        bus.register_expert("gnn", expert,
                            msg_types=[MessageType.CAUSAL, MessageType.ANALOGY],
                            description="图推理")
        experts["gnn"] = expert
        if verbose: print(f"  ✅ GNN 加载 (acc={ckpt['accuracy']:.1%})")

    # 6. VAE
    path = os.path.join(MODEL_DIR, "vae.pt")
    if os.path.exists(path):
        model, ckpt = ModelLoader.load_vae(path)
        expert = VAEExpert(model)
        bus.register_expert("vae", expert,
                            msg_types=[MessageType.MEMORY],
                            description="状态压缩")
        experts["vae"] = expert
        if verbose: print(f"  ✅ VAE 加载 (loss={ckpt['best_loss']:.4f})")

    # 加载总线权重
    bus_path = os.path.join(MODEL_DIR, "message_bus_v0.0.0.1.pt")
    if os.path.exists(bus_path):
        bus.load(bus_path)
        if verbose: print(f"  ✅ MessageBus 权重已加载")

    print()
    return bus, experts


# ─── 端到端验证 ─────────────────────────────────────────────

def end_to_end_test():
    """端到端集成测试"""
    print("=" * 60)
    print("  端到端集成测试")
    print("=" * 60)
    print()

    bus, experts = create_integrated_bus(verbose=True)

    test_cases = [
        ("1' OR 1=1 --", MessageType.SECURITY, MessagePriority.CRITICAL),
        ("<script>alert('XSS')</script>", MessageType.SECURITY, MessagePriority.CRITICAL),
        ("../../../etc/passwd", MessageType.SECURITY, MessagePriority.HIGH),
        ("你好，请帮我查一下天气", MessageType.LANGUAGE, MessagePriority.HIGH),
        ("hello world", MessageType.LANGUAGE, MessagePriority.NORMAL),
        ("GET /index.html HTTP/1.1", MessageType.LANGUAGE, MessagePriority.LOW),
        ("POST /login user=admin&password=123456", MessageType.SECURITY, MessagePriority.HIGH),
    ]

    print("─" * 60)
    for text, msg_type, priority in test_cases:
        print(f"\n  输入: '{text[:50]}'")
        print(f"  类型: {msg_type.value}, 优先级: {priority.value}")

        msg = BusMessage(
            msg_type=msg_type, priority=priority,
            payload=text, source="test",
            ethical_signature=0.9,
        )
        result = bus.process(msg)

        if result["status"] == "rejected":
            print(f"  ❌ 被拒: {result['reason']}")
            continue

        routed = result.get("routed_to", [])
        print(f"  路由到: {routed}")

        for name, output in result.get("expert_outputs", {}).items():
            if isinstance(output, dict):
                label = output.get("label", output.get("status", "?"))
                conf = output.get("confidence", 0)
                if name in ("textcnn", "snn", "transformer"):
                    print(f"    {name:12s}: {label} ({conf:.1%})")
                elif name == "encoder":
                    print(f"    {name:12s}: norm={output.get('norm',0):.3f}")
                elif name == "gnn":
                    print(f"    {name:12s}: matched={output.get('matched_nodes',0)}")
                elif name == "vae":
                    print(f"    {name:12s}: error={output.get('recon_error',0):.4f}")

        fused = result.get("fused_state", {})
        if fused:
            print(f"  融合: norm={fused.get('norm',0):.3f}, state={fused.get('state_norm',0):.3f}")
        print(f"  延迟: {result.get('latency_ms',0):.1f}ms")

    print()
    print("─" * 60)
    print("  总线状态:")
    status = bus.get_status()
    print(f"  消息: {status['stats']}")
    print(f"  校验: {status['validator']}")
    print()
    print("=" * 60)
    print("  端到端集成测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    end_to_end_test()
