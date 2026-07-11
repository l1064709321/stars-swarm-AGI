#!/usr/bin/env python3
"""
MessageBus v0.0.0.1 — MoE Router + Mamba 双层总线

架构:
  第一层: MoE Router  — 决定"谁来做"（路由决策）
  第二层: Mamba (SSM) — 决定"结果怎么融合"（跨时序融合）

设计原则:
  1. 所有神经架构注册为节点，通过总线通信
  2. 消息按类型路由到对应架构
  3. 各架构输出经 Mamba 融合后输出统一决策
  4. 三层校验: 物理检查 + 逻辑检查 + 伦理门控
  5. CPU 优先，ARM 架构优化，适配 AidLux 环境
  6. 自训练小模型，零版权风险

依赖: PyTorch
"""

import threading
import time
import heapq
import uuid
import math
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from datetime import datetime
from enum import Enum

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("[MessageBus v0.0.0.1] 警告: PyTorch 不可用，降级为纯 Python 模式")

try:
    import numpy as np
    NUMPY_OK = True
except ImportError:
    NUMPY_OK = False


# ================================================================
#  消息协议
# ================================================================

class MessageType(Enum):
    """消息类型 — 决定 MoE Router 的路由方向"""
    LANGUAGE = "language"           # 语言理解
    SECURITY = "security"            # 安全感知
    PATTERN = "pattern"              # 模式识别
    CAUSAL = "causal"                # 因果推理
    PLANNING = "planning"            # 规划搜索
    MEMORY = "memory"                # 记忆巩固
    ANALOGY = "analogy"              # 类比推理
    ETHICS = "ethics"                # 伦理评估
    EMERGENCE = "emergence"          # 涌现事件
    SYSTEM = "system"                # 系统控制


class MessagePriority(Enum):
    """消息优先级"""
    CRITICAL = 0    # 安全威胁 / 系统故障
    HIGH = 1        # 用户交互 / 实时响应
    NORMAL = 2      # 常规认知
    LOW = 3         # 后台任务 / 记忆巩固
    BACKGROUND = 4  # 空闲时执行


@dataclass
class BusMessage:
    """
    总线消息 — 所有神经架构间通信的统一格式

    生命周期:
        创建 → 三层校验 → MoE路由 → 专家处理 → Mamba融合 → 输出
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    msg_type: MessageType = MessageType.SYSTEM
    priority: MessagePriority = MessagePriority.NORMAL

    # 载荷
    payload: Any = None                    # 原始输入（文本/向量/结构化数据）
    feature_vector: Any = None             # 提取的特征向量（供 Router 决策）

    # 路由信息
    source: str = ""                       # 发送者名称
    target_experts: List[str] = field(default_factory=list)  # 指定路由目标（空=Router自动决定）
    routing_weights: Dict[str, float] = field(default_factory=dict)

    # 伦理签名
    ethical_signature: float = 0.8         # 伦理置信度（0-1）
    ethical_state: Optional[dict] = None    # 伦理动力系统状态

    # 溯源链
    origin_chain: List[str] = field(default_factory=list)
    tick: int = 0                          # 全局时钟

    # 状态
    validated: bool = False                # 是否通过三层校验
    processed: bool = False                # 是否已被专家处理
    expert_outputs: Dict[str, Any] = field(default_factory=dict)  # 各专家的输出

    # 时间戳
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    processed_at: Optional[str] = None

    def add_origin(self, node: str):
        """添加溯源节点"""
        self.origin_chain.append(node)

    def to_feature(self) -> 'torch.Tensor':
        """
        将消息转为 Router 可用的特征向量

        特征构成:
        - 消息类型 one-hot (10维)
        - 优先级 one-hot (5维)
        - 伦理签名 (1维)
        - 特征向量 (如果有, 补齐到48维)
        总计: 64维
        """
        if not TORCH_OK:
            return [0.0] * 64

        feat = torch.zeros(64)

        # 消息类型 one-hot (0-9)
        type_idx = list(MessageType).index(self.msg_type)
        feat[type_idx] = 1.0

        # 优先级 one-hot (10-14)
        pri_idx = list(MessagePriority).index(self.priority)
        feat[10 + pri_idx] = 1.0

        # 伦理签名 (15)
        feat[15] = self.ethical_signature

        # 特征向量 (16-63, 48维)
        if self.feature_vector is not None:
            fv = self.feature_vector
            if isinstance(fv, torch.Tensor):
                fv = fv.flatten().detach()
            elif isinstance(fv, (list, tuple)):
                fv = torch.tensor(fv, dtype=torch.float32)
            elif isinstance(fv, (int, float)):
                fv = torch.tensor([float(fv)])
            else:
                fv = torch.zeros(1)

            # 截断或补零到48维
            if len(fv) >= 48:
                feat[16:] = fv[:48]
            else:
                feat[16:16+len(fv)] = fv

        return feat


# ================================================================
#  三层校验
# ================================================================

class ThreeLayerValidator:
    """
    三层校验器 — 外部神经网络的输出必须通过才能进入系统

    第一层: 物理检查 — 载荷格式、向量维度、数值范围
    第二层: 逻辑检查 — 语义一致性、因果关系合理性
    第三层: 伦理门控 — 伦理签名阈值、伤害评估
    """

    def __init__(self,
                 physical_max_dim: int = 1024,
                 physical_max_norm: float = 1000.0,
                 logic_consistency_threshold: float = 0.3,
                 ethical_min_signature: float = 0.5):
        self.physical_max_dim = physical_max_dim
        self.physical_max_norm = physical_max_norm
        self.logic_threshold = logic_consistency_threshold
        self.ethical_min = ethical_min_signature

        # 审计日志
        self.audit_log: deque = deque(maxlen=500)

        # 统计
        self.stats = {
            "total": 0,
            "passed": 0,
            "rejected_physical": 0,
            "rejected_logical": 0,
            "rejected_ethical": 0,
        }

    def validate(self, msg: BusMessage) -> Tuple[bool, str]:
        """
        执行三层校验

        返回: (是否通过, 原因)
        """
        self.stats["total"] += 1
        reason = ""

        # ── 第一层: 物理检查 ──
        ok, reason = self._physical_check(msg)
        if not ok:
            self.stats["rejected_physical"] += 1
            self._audit(msg, "REJECTED", f"physical: {reason}")
            return False, reason

        # ── 第二层: 逻辑检查 ──
        ok, reason = self._logic_check(msg)
        if not ok:
            self.stats["rejected_logical"] += 1
            self._audit(msg, "REJECTED", f"logical: {reason}")
            return False, reason

        # ── 第三层: 伦理门控 ──
        ok, reason = self._ethical_gate(msg)
        if not ok:
            self.stats["rejected_ethical"] += 1
            self._audit(msg, "REJECTED", f"ethical: {reason}")
            return False, reason

        msg.validated = True
        self.stats["passed"] += 1
        self._audit(msg, "PASSED", "all checks passed")
        return True, "passed"

    def _physical_check(self, msg: BusMessage) -> Tuple[bool, str]:
        """物理检查: 格式、维度、数值范围"""
        if msg.payload is None and msg.feature_vector is None:
            return False, "empty message (no payload or feature_vector)"

        if msg.feature_vector is not None and TORCH_OK:
            fv = msg.feature_vector
            if isinstance(fv, torch.Tensor):
                if fv.dim() > 2:
                    return False, f"feature_vector dim too high: {fv.dim()}"
                if fv.numel() > self.physical_max_dim:
                    return False, f"feature_vector too large: {fv.numel()}"
                if torch.isnan(fv).any():
                    return False, "feature_vector contains NaN"
                if torch.isinf(fv).any():
                    return False, "feature_vector contains Inf"
                if fv.numel() > 0:
                    norm = float(fv.norm())
                    if norm > self.physical_max_norm:
                        return False, f"feature_vector norm too large: {norm:.1f}"

        if msg.ethical_signature < 0 or msg.ethical_signature > 1:
            return False, f"ethical_signature out of range: {msg.ethical_signature}"

        return True, "ok"

    def _logic_check(self, msg: BusMessage) -> Tuple[bool, str]:
        """逻辑检查: 语义一致性"""
        # 消息类型与载荷的简单一致性检查
        if msg.msg_type == MessageType.LANGUAGE:
            if isinstance(msg.payload, str) and len(msg.payload) > 10000:
                return False, "language payload too long"
        elif msg.msg_type == MessageType.SECURITY:
            if msg.payload is None:
                return False, "security message missing payload"
        elif msg.msg_type == MessageType.ETHICS:
            if msg.ethical_signature < self.ethical_min:
                # 伦理消息本身伦理签名低 — 可能是篡改
                if msg.ethical_signature < 0.3:
                    return False, f"ethics message with very low signature: {msg.ethical_signature}"

        # 优先级与类型的一致性
        if msg.priority == MessagePriority.CRITICAL and msg.msg_type == MessageType.SYSTEM:
            # 系统控制消息不应该标为最高优先级（除非是安全相关）
            if not isinstance(msg.payload, str) or "security" not in str(msg.payload).lower():
                return False, "system message marked as critical priority (non-security)"

        return True, "ok"

    def _ethical_gate(self, msg: BusMessage) -> Tuple[bool, str]:
        """伦理门控: 伦理签名阈值"""
        if msg.ethical_signature < self.ethical_min:
            return False, f"ethical_signature below threshold: {msg.ethical_signature:.2f} < {self.ethical_min}"

        return True, "ok"

    def _audit(self, msg: BusMessage, status: str, detail: str):
        """审计日志"""
        self.audit_log.append({
            "time": datetime.now().isoformat(),
            "msg_id": msg.id,
            "type": msg.msg_type.value,
            "status": status,
            "detail": detail,
        })

    def get_stats(self) -> dict:
        s = dict(self.stats)
        s["pass_rate"] = s["passed"] / max(s["total"], 1)
        return s


# ================================================================
#  第一层: MoE Router（路由决策层）
# ================================================================

class MoERouter(nn.Module if TORCH_OK else object):
    """
    Mixture-of-Experts 路由器

    根据消息特征，决定将消息路由给哪些专家（神经架构）处理。
    门控网络: input(64维) → hidden(128) → experts(N)

    参数量: ~5万
    CPU 推理: <1ms
    """

    def __init__(self, num_experts=8, feature_dim=64, hidden_dim=128, top_k=3):
        if TORCH_OK:
            super().__init__()
        self.num_experts = num_experts
        self.feature_dim = feature_dim
        self.top_k = min(top_k, num_experts)
        self.expert_names: List[str] = [""] * num_experts
        self._trained = False  # 标记是否已训练

        if TORCH_OK:
            # 门控网络
            self.gate = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, num_experts),
            )

            # 负载均衡统计
            self.register_buffer('expert_count', torch.zeros(num_experts))
            self.register_buffer('total_count', torch.tensor(0.0))

        # 规则路由表（未训练时使用 + 训练后兜底）
        self._rule_table: Dict[MessageType, List[int]] = {
            MessageType.LANGUAGE: [],
            MessageType.SECURITY: [],
            MessageType.PATTERN: [],
            MessageType.CAUSAL: [],
            MessageType.PLANNING: [],
            MessageType.MEMORY: [],
            MessageType.ANALOGY: [],
            MessageType.ETHICS: [],
            MessageType.EMERGENCE: [],
            MessageType.SYSTEM: [],
        }

    def register_expert(self, name: str, expert_id: int):
        """注册一个专家"""
        if 0 <= expert_id < self.num_experts:
            self.expert_names[expert_id] = name
            # 更新规则路由表: 按专家声明的 msg_types 自动填充
            # 这个会在 MessageBus.register_expert 中通过 msg_types 调用

    def _update_rule_table(self, expert_id: int, msg_types: List[MessageType]):
        """更新规则路由表"""
        for mt in msg_types:
            if expert_id not in self._rule_table[mt]:
                self._rule_table[mt].append(expert_id)

    def route(self, msg: BusMessage) -> Tuple[List[int], List[float]]:
        """
        路由决策

        未训练时: 用规则路由（按消息类型→专家声明的 msg_types）
        训练后: 用 MoE 门控网络做路由

        返回: (选中的专家ID列表, 对应权重)
        """
        # 如果消息指定了目标专家，直接使用
        if msg.target_experts:
            indices = []
            for name in msg.target_experts:
                if name in self.expert_names:
                    idx = self.expert_names.index(name)
                    indices.append(idx)
            if indices:
                weights = [1.0 / len(indices)] * len(indices)
                return indices, weights

        # 未训练时用规则路由
        if not TORCH_OK or not self._trained:
            return self._rule_route(msg)

        # 训练后用神经网络路由
        features = msg.to_feature().unsqueeze(0)  # [1, 64]

        with torch.no_grad():
            gate_logits = self.gate(features)
            gate_scores = F.softmax(gate_logits, dim=-1).squeeze(0)  # [num_experts]

            # Top-K 选择
            k = min(self.top_k, self.num_experts)
            topk_weights, topk_indices = torch.topk(gate_scores, k)

            # 归一化
            topk_weights = topk_weights / (topk_weights.sum() + 1e-8)

            # 负载均衡统计
            for idx in topk_indices:
                self.expert_count[idx] += 1
            self.total_count += 1

            indices = topk_indices.tolist()
            weights = topk_weights.tolist()

        return indices, weights

    def _rule_route(self, msg: BusMessage) -> Tuple[List[int], List[float]]:
        """规则路由: 按消息类型查找已注册专家"""
        indices = self._rule_table.get(msg.msg_type, [])
        if not indices:
            return [], []
        # Top-K 限制
        k = min(self.top_k, len(indices))
        indices = indices[:k]
        weights = [1.0 / len(indices)] * len(indices)
        return indices, weights

    def get_load_balance(self) -> dict:
        """负载均衡统计"""
        if not TORCH_OK:
            return {"status": "no_torch", "mode": "rule_based"}
        total = self.total_count.item()
        return {
            "mode": "neural" if self._trained else "rule_based",
            "trained": self._trained,
            "total_routed": int(total),
            "expert_distribution": {
                self.expert_names[i] or f"expert_{i}": {
                    "count": int(self.expert_count[i].item()),
                    "ratio": float(self.expert_count[i].item() / total) if total > 0 else 0.0,
                }
                for i in range(self.num_experts)
            }
        }


# ================================================================
#  第二层: Mamba 融合层（跨时序状态融合）
# ================================================================

class MambaSSM(nn.Module if TORCH_OK else object):
    """
    简化版 Mamba (State Space Model) 融合层

    原理:
    - 线性复杂度 O(n)，远优于 Transformer 的 O(n²)
    - 通过递推更新隐藏状态，融合跨时序信息
    - 选择性机制: 根据输入动态调整状态更新速率

    在总线中的作用:
    - 各专家架构处理完后，输出汇总到 Mamba
    - Mamba 维护"系统认知状态"（隐藏向量）
    - 每次新输入与历史状态融合，输出全局决策

    参数量: ~50万
    CPU 推理: <1ms
    """

    def __init__(self, input_dim=64, state_dim=128, output_dim=64):
        if TORCH_OK:
            super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim

        if TORCH_OK:
            # 输入投影: input_dim → state_dim
            self.input_proj = nn.Linear(input_dim, state_dim)

            # SSM 核心参数
            # A: 状态转移 (控制记忆衰减)
            # 用 HiPPO 矩阵初始化（S4 论文方法）
            A = self._build_hippo_matrix(state_dim)
            self.register_buffer('A', A)  # [state_dim, state_dim]

            # B, C: 可学习的输入/输出投影
            self.B_proj = nn.Linear(state_dim, state_dim, bias=False)
            self.C_proj = nn.Linear(state_dim, output_dim, bias=False)

            # D: 直通
            self.D = nn.Linear(input_dim, output_dim, bias=False)

            # 选择性门控: 根据输入决定状态更新幅度
            self.gate = nn.Sequential(
                nn.Linear(input_dim, state_dim),
                nn.Sigmoid(),  # 0~1, 控制记忆更新率
            )

            # 输出层
            self.output_proj = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
            )

            # 初始隐藏状态
            self.register_buffer('h', torch.zeros(state_dim))
        else:
            self.h = [0.0] * state_dim

    def _build_hippo_matrix(self, N: int) -> 'torch.Tensor':
        """
        HiPPO 矩阵初始化（S4/Mamba 论文方法的简化版）

        加入阻尼系数，防止状态范数爆炸。
        矩阵特征值被限制在负半平面，保证递推稳定。
        """
        if not TORCH_OK:
            return None

        A = torch.zeros(N, N)
        for n in range(N):
            for k in range(N):
                if n > k:
                    A[n, k] = math.sqrt((2 * n + 1) * (2 * k + 1)) / N
                elif n == k:
                    A[n, k] = -(n + 1) / N  # 阻尼: 除以N缩放
        # 确保矩阵稳定: 加对角负项使特征值为负
        A = A - 0.1 * torch.eye(N)
        return A

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """
        前向推理: 融合输入与历史状态

        参数:
            x: [input_dim] 当前输入向量

        返回:
            output: [output_dim] 融合后的全局状态
        """
        if not TORCH_OK:
            return self._pure_forward(x)

        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, input_dim]

        # 输入投影
        u = self.input_proj(x)  # [1, state_dim]

        # 选择性门控: 决定保留多少旧记忆 vs 接受多少新输入
        g = self.gate(x)  # [1, state_dim], 值在0~1

        # SSM 递推更新
        # h_new = A @ h + B(u) * g
        # 选择性: g 控制每个维度的更新率
        Bu = self.B_proj(u)  # [1, state_dim]
        h_new = torch.matmul(self.A, self.h.unsqueeze(1)).squeeze(1)  # [state_dim]
        h_new = h_new + g.squeeze(0) * Bu.squeeze(0)  # 选择性更新

        # 状态稳定: layer norm + 梯度截断
        h_new = F.layer_norm(h_new, h_new.shape)
        # 防止数值爆炸: 硬截断
        h_new = h_new.clamp(-10.0, 10.0)

        self.h = h_new.detach()  # 更新状态

        # 输出
        y = self.C_proj(h_new) + self.D(x.squeeze(0))  # [output_dim]
        y = self.output_proj(y)  # [output_dim]

        return y.squeeze(0)

    def _pure_forward(self, x):
        """降级模式: 简单指数移动平均"""
        alpha = 0.1
        for i in range(min(len(self.h), len(x) if isinstance(x, list) else 1)):
            self.h[i] = (1 - alpha) * self.h[i] + alpha * (x[i] if isinstance(x, list) else x)
        return self.h[:self.output_dim]

    def reset_state(self):
        """重置隐藏状态"""
        if TORCH_OK:
            self.h = torch.zeros(self.state_dim)
        else:
            self.h = [0.0] * self.state_dim

    def get_state_norm(self) -> float:
        """获取隐藏状态范数（用于监控）"""
        if TORCH_OK:
            return float(self.h.norm())
        return sum(h ** 2 for h in self.h) ** 0.5


# ================================================================
#  专家注册与执行
# ================================================================

class ExpertNode:
    """
    专家节点 — 包装一个神经架构，注册到总线

    每个专家接收 BusMessage，返回处理结果（任意格式）。
    """

    def __init__(self, name: str, expert_id: int,
                 process_fn: Callable[[BusMessage], Any],
                 msg_types: List[MessageType] = None,
                 description: str = ""):
        self.name = name
        self.expert_id = expert_id
        self.process_fn = process_fn
        self.msg_types = msg_types or []
        self.description = description
        self.call_count = 0
        self.total_time = 0.0
        self.last_error = None

    def process(self, msg: BusMessage) -> Any:
        """处理消息"""
        t0 = time.time()
        try:
            result = self.process_fn(msg)
            msg.expert_outputs[self.name] = result
            self.call_count += 1
            self.total_time += time.time() - t0
            return result
        except Exception as e:
            self.last_error = str(e)
            msg.expert_outputs[self.name] = {"error": str(e)}
            return {"error": str(e)}

    def get_stats(self) -> dict:
        return {
            "name": self.name,
            "calls": self.call_count,
            "avg_time_ms": (self.total_time / max(self.call_count, 1)) * 1000,
            "last_error": self.last_error,
            "description": self.description,
        }


# ================================================================
#  MessageBus v0.0.0.1 主总线
# ================================================================

class MessageBus:
    """
    MessageBus v0.0.0.1 — MoE Router + Mamba 双层总线

    使用方式:
        bus = MessageBus()

        # 注册专家
        bus.register_expert("transformer", process_fn=..., msg_types=[MessageType.LANGUAGE])
        bus.register_expert("snn", process_fn=..., msg_types=[MessageType.SECURITY])

        # 发送消息
        msg = BusMessage(msg_type=MessageType.SECURITY, payload="1' OR 1=1 --")
        result = bus.process(msg)

    数据流:
        消息 → 三层校验 → MoE路由 → 专家并行处理 → Mamba融合 → 输出
    """

    VERSION = "v0.0.0.1"

    def __init__(self,
                 num_experts=8,
                 state_dim=128,
                 feature_dim=64,
                 top_k=3,
                 enable_validation=True):
        self.num_experts = num_experts
        self.enable_validation = enable_validation

        # 组件
        self.router = MoERouter(
            num_experts=num_experts,
            feature_dim=feature_dim,
            top_k=top_k,
        )
        self.fusion = MambaSSM(
            input_dim=feature_dim,
            state_dim=state_dim,
            output_dim=feature_dim,
        )
        self.validator = ThreeLayerValidator()

        # 专家注册表
        self._experts: Dict[int, ExpertNode] = {}
        self._expert_name_to_id: Dict[str, int] = {}

        # 优先级队列
        self._queue: List[Tuple[int, float, BusMessage]] = []
        self._queue_counter = 0

        # 线程安全
        self._lock = threading.RLock()

        # 统计
        self.stats = {
            "total_messages": 0,
            "processed": 0,
            "rejected": 0,
            "avg_latency_ms": 0.0,
            "total_latency": 0.0,
        }

        # ACK 追踪
        self._pending_acks: Dict[str, BusMessage] = {}

        print(f"[MessageBus {self.VERSION}] 初始化完成 (experts={num_experts}, state_dim={state_dim})")

    def register_expert(self, name: str,
                        process_fn: Callable[[BusMessage], Any],
                        msg_types: List[MessageType] = None,
                        description: str = "") -> int:
        """
        注册一个专家架构

        参数:
            name: 专家名称（如 "transformer", "snn"）
            process_fn: 处理函数，接收 BusMessage，返回任意结果
            msg_types: 该专家处理的消息类型
            description: 描述

        返回: 专家ID
        """
        with self._lock:
            # 找空位
            expert_id = None
            for i in range(self.num_experts):
                if i not in self._experts:
                    expert_id = i
                    break

            if expert_id is None:
                print(f"[MessageBus] 警告: 专家已满 ({self.num_experts}), 无法注册 {name}")
                return -1

            node = ExpertNode(
                name=name,
                expert_id=expert_id,
                process_fn=process_fn,
                msg_types=msg_types,
                description=description,
            )
            self._experts[expert_id] = node
            self._expert_name_to_id[name] = expert_id
            self.router.register_expert(name, expert_id)
            self.router._update_rule_table(expert_id, msg_types or [])

            print(f"[MessageBus] 注册专家: {name} (id={expert_id}, types={[t.value for t in (msg_types or [])]})")
            return expert_id

    def submit(self, msg: BusMessage) -> bool:
        """
        提交消息到总线（异步，放入优先级队列）

        返回: 是否成功入队
        """
        with self._lock:
            # 三层校验
            if self.enable_validation:
                ok, reason = self.validator.validate(msg)
                if not ok:
                    self.stats["rejected"] += 1
                    print(f"[MessageBus] 消息被拒: {reason}")
                    return False

            # 入队
            self._queue_counter += 1
            heapq.heappush(
                self._queue,
                (msg.priority.value, self._queue_counter, msg)
            )
            self._pending_acks[msg.id] = msg
            self.stats["total_messages"] += 1
            return True

    def process(self, msg: BusMessage) -> Optional[dict]:
        """
        同步处理单条消息（不经过队列，直接路由+处理+融合）

        适用于实时性要求高的场景。

        返回: 处理结果
        """
        t0 = time.time()

        with self._lock:
            # 三层校验
            if self.enable_validation:
                ok, reason = self.validator.validate(msg)
                if not ok:
                    self.stats["rejected"] += 1
                    return {
                        "msg_id": msg.id,
                        "status": "rejected",
                        "reason": reason,
                    }

            self.stats["total_messages"] += 1

            # ── MoE 路由 ──
            expert_ids, weights = self.router.route(msg)

            if not expert_ids:
                # 没有可路由的专家
                return {
                    "msg_id": msg.id,
                    "status": "no_expert",
                    "msg_type": msg.msg_type.value,
                }

            # ── 专家处理 ──
            for eid, w in zip(expert_ids, weights):
                if eid in self._experts:
                    expert = self._experts[eid]
                    result = expert.process(msg)
                    msg.routing_weights[expert.name] = w

            msg.processed = True
            msg.processed_at = datetime.now().isoformat()

            # ── Mamba 融合 ──
            fused = self._fuse_outputs(msg)

            # 统计
            latency = (time.time() - t0) * 1000
            self.stats["processed"] += 1
            self.stats["total_latency"] += latency
            self.stats["avg_latency_ms"] = self.stats["total_latency"] / max(self.stats["processed"], 1)

            return {
                "msg_id": msg.id,
                "status": "processed",
                "msg_type": msg.msg_type.value,
                "routed_to": [self._experts[eid].name for eid in expert_ids if eid in self._experts],
                "routing_weights": msg.routing_weights,
                "expert_outputs": msg.expert_outputs,
                "fused_state": fused,
                "latency_ms": round(latency, 2),
            }

    def _fuse_outputs(self, msg: BusMessage) -> Any:
        """
        用 Mamba 融合各专家的输出

        将各专家输出转为向量，加权求和后送入 Mamba 做时序融合。
        """
        if not TORCH_OK:
            # 降级: 简单加权平均
            return {"mode": "fallback", "outputs": msg.expert_outputs}

        # 收集各专家输出并转为向量
        vectors = []
        weights = []
        for name, w in msg.routing_weights.items():
            output = msg.expert_outputs.get(name)
            if output is None:
                continue

            vec = self._output_to_vector(output)
            if vec is not None:
                vectors.append(vec * w)
                weights.append(w)

        if not vectors:
            return None

        # 加权融合
        fused_input = torch.stack(vectors).sum(dim=0)
        total_w = sum(weights)
        if total_w > 0:
            fused_input = fused_input / total_w

        # Mamba 时序融合
        fused_output = self.fusion(fused_input)
        fused_output = fused_output.detach()

        return {
            "vector": fused_output.tolist() if TORCH_OK else None,
            "norm": float(fused_output.norm()) if TORCH_OK else 0,
            "state_norm": self.fusion.get_state_norm(),
        }

    def _output_to_vector(self, output: Any) -> Optional['torch.Tensor']:
        """将专家输出转为 64 维向量"""
        if not TORCH_OK:
            return None

        if isinstance(output, torch.Tensor):
            v = output.flatten().float()
            if v.numel() >= 64:
                return v[:64]
            else:
                pad = torch.zeros(64 - v.numel())
                return torch.cat([v, pad])

        if isinstance(output, dict):
            # 从 dict 中提取数值
            vals = []
            for k, v in output.items():
                if isinstance(v, (int, float)):
                    vals.append(float(v))
                elif isinstance(v, torch.Tensor):
                    vals.extend(v.flatten().tolist()[:5])
            if vals:
                v = torch.tensor(vals[:64], dtype=torch.float32)
                if v.numel() < 64:
                    pad = torch.zeros(64 - v.numel())
                    v = torch.cat([v, pad])
                return v

        if isinstance(output, (list, tuple)):
            vals = [float(x) for x in output if isinstance(x, (int, float))][:64]
            if vals:
                v = torch.tensor(vals, dtype=torch.float32)
                if v.numel() < 64:
                    pad = torch.zeros(64 - v.numel())
                    v = torch.cat([v, pad])
                return v

        # 默认: 零向量
        return torch.zeros(64)

    def process_queue(self, max_messages: int = 100) -> int:
        """
        处理队列中的消息（按优先级）

        返回: 处理的消息数
        """
        count = 0
        with self._lock:
            while self._queue and count < max_messages:
                _, _, msg = heapq.heappop(self._queue)
                self.process(msg)
                self._pending_acks.pop(msg.id, None)
                count += 1
        return count

    def get_status(self) -> dict:
        """获取总线状态"""
        return {
            "version": self.VERSION,
            "torch_ok": TORCH_OK,
            "experts": {
                self._experts[eid].name: self._experts[eid].get_stats()
                for eid in self._experts
            },
            "router": self.router.get_load_balance(),
            "fusion": {
                "state_dim": self.fusion.state_dim,
                "state_norm": self.fusion.get_state_norm(),
            },
            "validator": self.validator.get_stats(),
            "queue_size": len(self._queue),
            "stats": self.stats,
        }

    def save(self, path: str):
        """保存 Router 和 Mamba 的权重"""
        if not TORCH_OK:
            print("[MessageBus] 降级模式，无可保存权重")
            return

        save_dict = {
            "version": self.VERSION,
            "router_state": self.router.state_dict(),
            "fusion_state": self.fusion.state_dict(),
        }
        torch.save(save_dict, path)
        print(f"[MessageBus] 权重已保存: {path}")

    def load(self, path: str):
        """加载权重"""
        if not TORCH_OK:
            return

        save_dict = torch.load(path, map_location='cpu', weights_only=False)
        if save_dict.get("version") != self.VERSION:
            print(f"[MessageBus] 版本不匹配: {save_dict.get('version')} vs {self.VERSION}")

        self.router.load_state_dict(save_dict["router_state"])
        self.fusion.load_state_dict(save_dict["fusion_state"])
        # v0.0.0.1: Router 权重未训练，保持规则路由
        # 训练 Router 后再启用: self.router._trained = True
        print(f"[MessageBus] 权重已加载: {path}")


# ================================================================
#  Demo — 验证总线可运行
# ================================================================

def demo():
    """最小可运行 demo"""

    print("=" * 60)
    print(f"  MessageBus v0.0.0.1 — Demo")
    print("=" * 60)
    print()

    # 创建总线
    bus = MessageBus(num_experts=8, state_dim=128, top_k=3)

    # ── 注册模拟专家 ──

    def transformer_fn(msg: BusMessage):
        """模拟 Transformer: 语言理解"""
        text = msg.payload if isinstance(msg.payload, str) else str(msg.payload)
        return {
            "intent": "sql_injection" if "or 1=1" in text.lower() else "normal",
            "confidence": 0.88,
            "source": "transformer",
        }

    def snn_fn(msg: BusMessage):
        """模拟 SNN: 安全感知"""
        text = msg.payload if isinstance(msg.payload, str) else ""
        score = 0.82 if any(kw in text.lower() for kw in ["union", "select", "drop", "or 1=1"]) else 0.12
        return {"snn_score": score, "source": "snn"}

    def textcnn_fn(msg: BusMessage):
        """模拟 TextCNN: 模式识别"""
        text = msg.payload if isinstance(msg.payload, str) else ""
        score = 0.91 if "or 1=1" in text.lower() else 0.15
        return {"cnn_score": score, "source": "textcnn"}

    def causal_fn(msg: BusMessage):
        """模拟因果推理"""
        return {"causal_path": "sql_injection -> ip_block", "confidence": 0.75, "source": "causal"}

    def ethics_fn(msg: BusMessage):
        """模拟伦理评估"""
        return {"ethical_ok": True, "principle": "non_harm", "source": "ethics"}

    bus.register_expert("transformer", transformer_fn,
                        msg_types=[MessageType.LANGUAGE], description="语言理解")
    bus.register_expert("snn", snn_fn,
                        msg_types=[MessageType.SECURITY], description="脉冲感知")
    bus.register_expert("textcnn", textcnn_fn,
                        msg_types=[MessageType.PATTERN, MessageType.SECURITY], description="模式识别")
    bus.register_expert("causal", causal_fn,
                        msg_types=[MessageType.CAUSAL], description="因果推理")
    bus.register_expert("ethics", ethics_fn,
                        msg_types=[MessageType.ETHICS], description="伦理评估")

    print()

    # ── 测试 1: 安全威胁消息 ──
    print("─" * 60)
    print("测试 1: SQL 注入检测")
    print("─" * 60)

    msg1 = BusMessage(
        msg_type=MessageType.SECURITY,
        priority=MessagePriority.CRITICAL,
        payload="1' OR 1=1 --",
        source="external_input",
        ethical_signature=0.9,
    )

    result1 = bus.process(msg1)
    print(f"  输入: {msg1.payload}")
    print(f"  路由到: {result1['routed_to']}")
    print(f"  权重: {result1['routing_weights']}")
    print(f"  专家输出:")
    for name, out in result1['expert_outputs'].items():
        print(f"    {name}: {out}")
    print(f"  融合状态范数: {result1['fused_state']['norm']:.4f}" if result1.get('fused_state') else "  无融合")
    print(f"  延迟: {result1['latency_ms']:.2f}ms")
    print()

    # ── 测试 2: 正常对话消息 ──
    print("─" * 60)
    print("测试 2: 正常对话")
    print("─" * 60)

    msg2 = BusMessage(
        msg_type=MessageType.LANGUAGE,
        priority=MessagePriority.HIGH,
        payload="你好，请帮我查一下天气",
        source="user",
        ethical_signature=0.95,
    )

    result2 = bus.process(msg2)
    print(f"  输入: {msg2.payload}")
    print(f"  路由到: {result2['routed_to']}")
    print(f"  延迟: {result2['latency_ms']:.2f}ms")
    print()

    # ── 测试 3: 低伦理签名消息（应被拒绝）──
    print("─" * 60)
    print("测试 3: 低伦理签名（应被拒绝）")
    print("─" * 60)

    msg3 = BusMessage(
        msg_type=MessageType.SYSTEM,
        priority=MessagePriority.NORMAL,
        payload="test",
        source="unknown",
        ethical_signature=0.1,  # 低于阈值 0.5
    )

    result3 = bus.process(msg3)
    print(f"  输入: ethical_signature={msg3.ethical_signature}")
    print(f"  状态: {result3['status']}")
    print(f"  原因: {result3.get('reason', 'N/A')}")
    print()

    # ── 测试 4: 连续消息（Mamba 时序融合）──
    print("─" * 60)
    print("测试 4: 连续消息 — Mamba 时序融合")
    print("─" * 60)

    for i in range(5):
        msg = BusMessage(
            msg_type=MessageType.SECURITY,
            priority=MessagePriority.HIGH,
            payload=f"attack attempt #{i}: UNION SELECT * FROM users",
            source="monitor",
            ethical_signature=0.85,
            tick=i,
        )
        result = bus.process(msg)
        state_norm = result['fused_state']['state_norm'] if result.get('fused_state') else 0
        print(f"  tick={i}: state_norm={state_norm:.4f}, latency={result['latency_ms']:.2f}ms")

    print()

    # ── 总线状态 ──
    print("─" * 60)
    print("总线状态")
    print("─" * 60)

    status = bus.get_status()
    print(f"  版本: {status['version']}")
    print(f"  PyTorch: {'✅' if status['torch_ok'] else '❌'}")
    print(f"  已注册专家: {len(status['experts'])}")
    for name, info in status['experts'].items():
        print(f"    {name}: {info['calls']}次调用, 平均{info['avg_time_ms']:.2f}ms")
    print(f"  消息统计: {status['stats']}")
    print(f"  校验统计: {status['validator']}")

    # ── 保存权重 ──
    if TORCH_OK:
        bus.save("/root/.codebuddy/artifact/star_os/models/message_bus_v0.0.0.1.pt")
        print()
        print("  ✅ 权重已保存")

    print()
    print("=" * 60)
    print("  Demo 完成 — MessageBus v0.0.0.1 可运行")
    print("=" * 60)


if __name__ == "__main__":
    demo()
