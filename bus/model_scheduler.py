#!/usr/bin/env python3
"""
model_scheduler.py — AidLux 串行模型调度器

功能:
  在 AidLux 手机环境 (12GB RAM) 上管理多个模型的加载/卸载。
  - 常驻模型: 始终在内存中（< 7MB 总计）
  - 按需模型: 用时加载，用完卸载
  - 自动内存管理: 内存不足时强制卸载按需模型

  设计目标:
  1. 常驻模型总内存 < 10MB
  2. 单个按需模型加载 < 500ms
  3. 内存监控 + 自动降级
"""

import os
import time
import threading
from typing import Any, Dict, Optional, List, Callable
from collections import OrderedDict

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


class ModelScheduler:
    """
    模型调度器

    使用方式:
        scheduler = ModelScheduler()

        # 加载常驻模型（启动时）
        scheduler.load_resident("textcnn", "models/textcnn.pt", load_fn)
        scheduler.load_resident("encoder", "models/encoder.pt", load_fn)
        scheduler.load_resident("snn", "models/snn.pt", load_fn)

        # 按需加载模型
        model = scheduler.load_on_demand("transformer", "models/transformer.pt", load_fn)
        result = model(input_data)
        scheduler.unload_on_demand("transformer")  # 用完卸载
    """

    # 常驻模型列表（始终在内存中）
    RESIDENT_MODELS = ["textcnn", "encoder", "snn", "message_bus"]

    # 按需模型列表（用时加载）
    ON_DEMAND_MODELS = ["transformer", "gnn", "vae"]

    # 内存限制（MB）
    MAX_MEMORY_MB = 6 * 1024  # 6GB 上限
    MEMORY_WARNING_MB = 5 * 1024  # 5GB 警告

    def __init__(self):
        self._resident: Dict[str, Any] = {}       # 常驻模型
        self._on_demand: OrderedDict[str, Any] = OrderedDict()  # 按需模型 (LRU)
        self._loaders: Dict[str, Callable] = {}    # 加载函数
        self._paths: Dict[str, str] = {}            # 模型文件路径
        self._stats: Dict[str, dict] = {}           # 统计信息
        self._lock = threading.RLock()

        # 内存监控
        self._monitor_thread = None
        self._monitoring = False

        print("[ModelScheduler] 初始化完成")
        print(f"  常驻模型: {self.RESIDENT_MODELS}")
        print(f"  按需模型: {self.ON_DEMAND_MODELS}")

    def register(self, name: str, path: str, loader: Callable, resident: bool = False):
        """注册一个模型"""
        with self._lock:
            self._paths[name] = path
            self._loaders[name] = loader
            self._stats[name] = {
                "load_count": 0,
                "total_load_time": 0.0,
                "inference_count": 0,
                "resident": resident,
                "loaded": False,
            }

    def load_resident(self, name: str, path: str, loader: Callable):
        """加载常驻模型"""
        with self._lock:
            if name in self._resident:
                return self._resident[name]

            if not os.path.exists(path):
                print(f"[ModelScheduler] 文件不存在: {path}")
                return None

            t0 = time.time()
            model = loader(path)
            load_time = time.time() - t0

            self._resident[name] = model
            if name not in self._stats:
                self._stats[name] = {
                    "load_count": 0, "total_load_time": 0.0,
                    "inference_count": 0, "resident": True, "loaded": False,
                }
            self._stats[name]["load_count"] = 1
            self._stats[name]["total_load_time"] = load_time
            self._stats[name]["loaded"] = True

            file_size = os.path.getsize(path) / 1024
            print(f"[ModelScheduler] 常驻模型已加载: {name} "
                  f"({file_size:.1f}KB, {load_time*1000:.0f}ms)")

            return model

    def load_on_demand(self, name: str, path: str = None, loader: Callable = None) -> Any:
        """按需加载模型"""
        with self._lock:
            # 已加载？
            if name in self._on_demand:
                # LRU: 移到末尾
                self._on_demand.move_to_end(name)
                return self._on_demand[name]

            # 获取路径和加载器
            path = path or self._paths.get(name)
            loader = loader or self._loaders.get(name)

            if not path or not loader:
                print(f"[ModelScheduler] 模型 {name} 未注册")
                return None

            if not os.path.exists(path):
                print(f"[ModelScheduler] 文件不存在: {path}")
                return None

            # 内存检查
            self._check_memory()

            # 加载
            t0 = time.time()
            model = loader(path)
            load_time = time.time() - t0

            self._on_demand[name] = model
            stats = self._stats.setdefault(name, {
                "load_count": 0, "total_load_time": 0.0,
                "inference_count": 0, "resident": False, "loaded": False,
            })
            stats["load_count"] += 1
            stats["total_load_time"] += load_time
            stats["loaded"] = True

            file_size = os.path.getsize(path) / 1024
            print(f"[ModelScheduler] 按需模型已加载: {name} "
                  f"({file_size:.1f}KB, {load_time*1000:.0f}ms)")

            return model

    def unload_on_demand(self, name: str):
        """卸载按需模型"""
        with self._lock:
            if name in self._on_demand:
                del self._on_demand[name]
                if name in self._stats:
                    self._stats[name]["loaded"] = False
                print(f"[ModelScheduler] 模型已卸载: {name}")

    def get_model(self, name: str) -> Optional[Any]:
        """获取已加载的模型"""
        with self._lock:
            if name in self._resident:
                return self._resident[name]
            if name in self._on_demand:
                self._on_demand.move_to_end(name)
                return self._on_demand[name]
            return None

    def _check_memory(self):
        """内存检查：超限时自动卸载最旧的按需模型"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            used_mb = (mem.total - mem.available) / (1024 * 1024)

            if used_mb > self.MEMORY_WARNING_MB:
                print(f"[ModelScheduler] 内存警告: {used_mb:.0f}MB > {self.MEMORY_WARNING_MB}MB")
                # 卸载最旧的按需模型
                while self._on_demand and used_mb > self.MEMORY_WARNING_MB:
                    oldest = next(iter(self._on_demand))
                    self.unload_on_demand(oldest)
                    mem = psutil.virtual_memory()
                    used_mb = (mem.total - mem.available) / (1024 * 1024)
        except ImportError:
            # psutil 不可用，跳过内存检查
            pass

    def start_monitor(self, interval_sec: int = 30):
        """启动内存监控线程"""
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval_sec,), daemon=True
        )
        self._monitor_thread.start()

    def stop_monitor(self):
        """停止监控"""
        self._monitoring = False

    def _monitor_loop(self, interval: int):
        while self._monitoring:
            time.sleep(interval)
            self._check_memory()

    def get_status(self) -> dict:
        """获取调度器状态"""
        with self._lock:
            status = {
                "resident_models": {
                    name: {
                        "loaded": True,
                        "stats": self._stats.get(name, {}),
                    }
                    for name in self._resident
                },
                "on_demand_models": {
                    name: {
                        "loaded": True,
                        "stats": self._stats.get(name, {}),
                    }
                    for name in self._on_demand
                },
                "total_models_loaded": len(self._resident) + len(self._on_demand),
            }

            try:
                import psutil
                mem = psutil.virtual_memory()
                status["memory"] = {
                    "total_mb": mem.total / (1024 * 1024),
                    "available_mb": mem.available / (1024 * 1024),
                    "used_percent": mem.percent,
                }
            except ImportError:
                status["memory"] = None

            return status


# ─── Demo ──────────────────────────────────────────────────

def demo():
    """测试调度器"""

    print()
    print("=" * 60)
    print("  ModelScheduler Demo")
    print("=" * 60)
    print()

    scheduler = ModelScheduler()

    # 模拟加载函数
    def make_loader(model_name):
        def loader(path):
            if TORCH_OK:
                return torch.load(path, map_location='cpu', weights_only=False)
            return {"path": path}
        return loader

    # 加载常驻模型
    model_dir = "/root/.codebuddy/artifact/star_os/models"

    print("--- 加载常驻模型 ---")
    for name, filename in [
        ("message_bus", "message_bus_v0.0.0.1.pt"),
        ("textcnn", "textcnn.pt"),
        ("encoder", "encoder.pt"),
        ("snn", "snn.pt"),
    ]:
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            scheduler.load_resident(name, path, make_loader(name))
        else:
            print(f"  跳过: {filename} (不存在)")

    print()
    print("--- 状态 ---")
    status = scheduler.get_status()
    print(f"  已加载模型: {status['total_models_loaded']}")
    print(f"  常驻: {list(status['resident_models'].keys())}")
    if status.get('memory'):
        mem = status['memory']
        print(f"  内存: {mem['available_mb']:.0f}MB 可用 / {mem['total_mb']:.0f}MB 总计")

    print()
    print("=" * 60)
    print("  ModelScheduler Demo 完成")
    print("=" * 60)


if __name__ == "__main__":
    demo()
