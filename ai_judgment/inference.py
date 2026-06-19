# coding: utf-8
"""
推理引擎 (Module 1)
===================

职责：
  1. 进程级**单例**加载 DPSG 权重（避免每个请求重复 load checkpoint）。
  2. 注入数据依赖（邻接表 / embedding 字典 / 邻居字典）。
  3. 对外暴露 analyze(news_id) -> JudgmentMetrics：跑一次推理，取出中间变量，
     调用 internals.compute_judgment_metrics 得到结构化量化研判结果。

注意：把"一条新闻 + 其传播子图"组织成模型输入（Het_Node + neighbor_order +
all_t）依赖项目自有的图构建逻辑（GossipCop.py 的 neighbor_loader/data_loader）。
该部分与训练机的数据落盘格式强绑定，这里以 DataBundle 抽象出**唯一集成点**
（load_data_bundle），部署时对接真实数据即可，其余链路全部就绪。
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Optional

from . import config
from .internals import JudgmentMetrics, compute_judgment_metrics


# ---------------------------------------------------------------------------
# 轻量 Het_Node（与 GossipCop.py 的 Het_Node 字段一致，避免 import 训练脚本）
# ---------------------------------------------------------------------------
@dataclass
class HetNode:
    node_type: str
    node_id: str
    emb: object = None
    label: Optional[int] = None


@dataclass
class DataBundle:
    """图构建所需的全部数据。集成时由 load_data_bundle 填充。"""
    store: object                  # model.DataStore
    neighbor_order: dict           # node_id -> neighbor_order_n_p_u 列表
    add_time: dict                 # node_id -> 新闻自身时间戳
    neighbor_time: dict            # node_id -> 邻居时间戳列表
    nodes_by_id: dict              # node_id -> HetNode


# ---------------------------------------------------------------------------
# 推理引擎（单例）
# ---------------------------------------------------------------------------
class InferenceEngine:
    _instance: Optional["InferenceEngine"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.device = config.resolve_device()
        self.model = None
        self.bundle: Optional[DataBundle] = None
        self.ratio = config.DATASET_RATIO
        self._ready = False

    # -------- 单例入口 --------
    @classmethod
    def instance(cls) -> "InferenceEngine":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # -------- 启动加载 --------
    def load(self, checkpoint_path: Optional[str] = None):
        """加载权重 + 数据依赖。在 FastAPI startup 钩子里调用一次。"""
        import torch
        from .model import build_model

        ckpt_path = checkpoint_path or config.MODEL_CHECKPOINT_PATH
        self.model = build_model(config.MODEL_HPARAMS, device=self.device)

        if os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            state = checkpoint.get("model_state_dict", checkpoint)
            # strict=False：抽取版去掉了若干训练专用 buffer，容忍少量键差异
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"[InferenceEngine] load_state_dict 部分匹配 "
                      f"missing={len(missing)} unexpected={len(unexpected)}")
            print(f"[InferenceEngine] 已加载权重: {ckpt_path}")
        else:
            print(f"[InferenceEngine][WARN] 未找到权重 {ckpt_path}，使用随机初始化"
                  f"（仅用于打通链路，结果无意义）")

        self.model.eval()

        # —— 数据依赖：集成点 ——
        self.bundle = self.load_data_bundle()
        if self.bundle is not None:
            self.model.bind_data_store(self.bundle.store)
            if getattr(self.bundle, "ratio", None):
                self.ratio = self.bundle.ratio

        self._ready = True
        return self

    def load_data_bundle(self) -> Optional[DataBundle]:
        """
        【唯一集成点】把训练机的数据落盘格式装配成 DataBundle。

        实现见 data_adapter.build_data_bundle()：复用 GossipCop.py 的
        neighbor_loader / data_loader 格式，从 config 的 7 个路径环境变量装配。
        路径未配置（或文件不存在）时返回 None —— 表示"数据未挂载"，API 会以
        503 明确报错而非静默出错。
        """
        from .data_adapter import build_data_bundle
        return build_data_bundle()

    @property
    def ready(self) -> bool:
        return self._ready and self.bundle is not None

    # -------- 推理 --------
    def analyze(self, news_id: str) -> JudgmentMetrics:
        """对单条新闻做研判，返回量化指标。"""
        if not self.ready:
            raise RuntimeError("DATA_NOT_LOADED: 数据依赖未挂载，请实现 load_data_bundle。")
        if news_id not in self.bundle.nodes_by_id:
            raise KeyError(f"NEWS_NOT_FOUND: 未知 news_id={news_id}")

        import torch

        node = self.bundle.nodes_by_id[news_id]
        neighbor_order = self.bundle.neighbor_order[news_id]
        all_t = [self.bundle.add_time[news_id]] + list(self.bundle.neighbor_time[news_id])

        with torch.no_grad():
            _pred, _dist, internals = self.model(
                node, neighbor_order, all_t, return_internals=True
            )

        return compute_judgment_metrics(
            probability=internals.probability,
            dist_mse=internals.dist_mse,
            rep_scale=internals.rep_scale,
            modality_energy=internals.modality_energy,
            num_neighbors=internals.num_neighbors,
            ratio=self.ratio,
            fake_risk_scale=config.FAKE_RISK_SCALE,
            risk_tiers=config.RISK_TIERS,
            structural_conflict_alert_th=config.STRUCTURAL_CONFLICT_ALERT,
            modality_dominance_th=config.MODALITY_DOMINANCE_ALERT,
            sparse_threshold=config.SPARSE_PROPAGATION_THRESHOLD,
        )


def get_engine() -> InferenceEngine:
    return InferenceEngine.instance()
