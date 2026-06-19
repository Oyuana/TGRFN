# coding: utf-8
"""
数据适配器 (Module 1 集成点实现)
================================

把队员/学姐预处理好的落盘数据，**无副作用**地装配成推理引擎所需的 DataBundle。
逻辑与 `models/train_and_evaluation/GossipCop.py` 的加载流程一一对应，但：

  * 不在 import 时执行（必须显式调用 build_data_bundle）；
  * 所有路径来自 config（环境变量），不写死训练机绝对路径；
  * 只取推理需要的部分，不触发任何训练循环。

落盘格式（与 GossipCop.py 完全一致）：
  - n_add_time.txt / p_add_time.txt : 每行 "key value"，新闻时间戳带 'gossipcop-' 前缀
  - n_neighbors.txt                 : neighbor_loader 的输入
  - original_adj                    : json，邻接表 adj_list
  - normalized_{news,post,user}_nodes/batch_*.txt : data_loader 的输入

部署时通常**无需改动本文件**，只要把 7 个路径环境变量指向真实数据即可。
若队员的数据集前缀不是 'gossipcop-'，改 TGRFN_NEWS_ID_PREFIX 环境变量。
"""
from __future__ import annotations

import json
import os
from typing import Optional

from . import config


NEWS_ID_PREFIX = os.getenv("TGRFN_NEWS_ID_PREFIX", "gossipcop-")


# ---------------------------------------------------------------------------
# 时间戳字典（对应 GossipCop.py:34-47）
# ---------------------------------------------------------------------------
def _load_time_dicts(n_time_file: str, p_time_file: str):
    n_add_time_dict, p_add_time_dict = {}, {}
    with open(n_time_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                key, value = line.split(" ", 1)
                n_add_time_dict[NEWS_ID_PREFIX + key] = value
    with open(p_time_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                key, value = line.split(" ", 1)
                p_add_time_dict[key] = value
    return n_add_time_dict, p_add_time_dict


# ---------------------------------------------------------------------------
# 邻居字典（对应 GossipCop.py:59-106 的 neighbor_loader）
# 改为显式接收 time 字典，去掉对模块级全局的依赖。
# ---------------------------------------------------------------------------
def _neighbor_loader(pathway: str, n_add_time_dict: dict, p_add_time_dict: dict):
    neighbor_dict_post, neighbor_dict_user, neighbor_dict_news = {}, {}, {}
    neighbor_dict_n_p_u, neighbor_dict_n_p_u_add_time, neighbor_dict_n = {}, {}, {}
    with open(pathway) as f:
        Lines = f.readlines()

    for line in Lines:
        neighbor_list = line.split()
        head = neighbor_list[0][1:-1]
        for item in neighbor_list[1:]:
            node_type, node_id = item[0], item[1:]
            if node_type == "n" and node_id != "PADDING":
                time_value = n_add_time_dict.get(node_id, "0")
            elif node_type == "p" and node_id != "PADDING":
                time_value = p_add_time_dict.get(node_id, "0")
            elif node_type == "u" and node_id != "PADDING":
                time_value = "0"
            else:
                continue
            neighbor_dict_n_p_u_add_time.setdefault(head, []).append(time_value)

        key = head.split("t")[0]
        neighbor_dict_n_p_u[key] = [
            (item[0], item[1:].split("t")[0])
            for item in neighbor_list[1:] if item[1:] != "PADDING"
        ]
        neighbor_dict_n[key] = [
            (item[0], item[1:].split("t")[0])
            for item in neighbor_list[1:] if item[0] == "n" and item[1:] != "PADDING"
        ]
        neighbor_dict_news[head] = [item[1:] for item in neighbor_list[1:] if item[0] == "n" and item[1:] != "PADDING"]
        neighbor_dict_user[head] = [item[1:] for item in neighbor_list[1:] if item[0] == "u" and item[1:] != "PADDING"]
        neighbor_dict_post[head] = [item[1:] for item in neighbor_list[1:] if item[0] == "p" and item[1:] != "PADDING"]

    return (neighbor_dict_n_p_u, neighbor_dict_n, neighbor_dict_n_p_u_add_time,
            neighbor_dict_news, neighbor_dict_post, neighbor_dict_user)


# ---------------------------------------------------------------------------
# 节点 embedding（对应 GossipCop.py:114-207 的 data_loader）
# 返回 (emb_dict, label_dict)；post/user 无 label。
# ---------------------------------------------------------------------------
def _load_news_nodes(pathway: str):
    emb_dict, label_dict = {}, {}
    for fname in sorted(os.listdir(pathway)):
        if not fname.startswith("batch_"):
            continue
        with open(os.path.join(pathway, fname)) as f:
            Lines = f.readlines()
        embed = []
        cur_id = None
        for j, line in enumerate(Lines):
            r = j % 7
            if r == 0:
                _, cur_id, label = line.split()
                label_dict[cur_id] = int(label)
                embed = []
            elif r in (1, 2, 3):
                embed.append(list(map(float, line.split())))
                if r == 3:
                    emb_dict[cur_id] = embed
    return emb_dict, label_dict


def _load_pu_nodes(pathway: str):
    emb_dict = {}
    for fname in sorted(os.listdir(pathway)):
        if not fname.startswith("batch_"):
            continue
        with open(os.path.join(pathway, fname)) as f:
            Lines = f.readlines()
        embed = []
        cur_id = None
        for j, line in enumerate(Lines):
            r = j % 6
            if r == 0:
                cur_id = line.split()[1]
                embed = []
            elif r in (1, 2):
                embed.append(list(map(float, line.split())))
                if r == 2:
                    emb_dict[cur_id] = embed
    return emb_dict


# ---------------------------------------------------------------------------
# 装配 DataBundle —— 集成点真正的实现
# ---------------------------------------------------------------------------
def build_data_bundle():
    """读取 config 中的 7 个路径，装配并返回 DataBundle；任一路径缺失则返回 None。"""
    paths = [
        config.DATA_NEIGHBORS_FILE, config.DATA_ADJ_FILE,
        config.DATA_N_TIME_FILE, config.DATA_P_TIME_FILE,
        config.DATA_NEWS_DIR, config.DATA_POST_DIR, config.DATA_USER_DIR,
    ]
    if not all(paths) or not all(os.path.exists(p) for p in paths):
        return None

    from .model import DataStore
    from .inference import DataBundle, HetNode

    n_add_time_dict, p_add_time_dict = _load_time_dicts(
        config.DATA_N_TIME_FILE, config.DATA_P_TIME_FILE)
    neighbor_dict = _neighbor_loader(
        config.DATA_NEIGHBORS_FILE, n_add_time_dict, p_add_time_dict)
    with open(config.DATA_ADJ_FILE, "r") as f:
        adj_list = json.load(f)

    news_emb_dict, news_label_dict = _load_news_nodes(config.DATA_NEWS_DIR)
    post_emb_dict = _load_pu_nodes(config.DATA_POST_DIR)
    user_emb_dict = _load_pu_nodes(config.DATA_USER_DIR)

    store = DataStore(
        adj_list=adj_list, neighbor_dict=neighbor_dict,
        news_emb_dict=news_emb_dict, post_emb_dict=post_emb_dict,
        user_emb_dict=user_emb_dict,
    )

    neighbor_order_dict = neighbor_dict[0]       # neighbor_dict_n_p_u
    neighbor_time_dict = neighbor_dict[2]        # neighbor_dict_n_p_u_add_time

    # 训练集正负样本比 ratio（GossipCop.py:230），若未通过环境变量给定则从数据统计。
    n_real = sum(1 for v in news_label_dict.values() if v == 1)
    n_fake = sum(1 for v in news_label_dict.values() if v == 0)
    ratio = (n_real / n_fake) if n_fake else config.DATASET_RATIO

    # 只保留可推理的新闻节点：必须同时有邻居顺序与自身时间戳（对应 GossipCop.py:1070 的保险判断）
    nodes_by_id, neighbor_order, add_time, neighbor_time = {}, {}, {}, {}
    for news_id in news_emb_dict:
        if news_id in neighbor_order_dict and news_id in n_add_time_dict:
            nodes_by_id[news_id] = HetNode(
                node_type="news", node_id=news_id,
                emb=news_emb_dict[news_id], label=news_label_dict.get(news_id))
            neighbor_order[news_id] = neighbor_order_dict[news_id]
            add_time[news_id] = n_add_time_dict[news_id]
            neighbor_time[news_id] = neighbor_time_dict.get(news_id, [])

    bundle = DataBundle(
        store=store, neighbor_order=neighbor_order, add_time=add_time,
        neighbor_time=neighbor_time, nodes_by_id=nodes_by_id,
    )
    bundle.ratio = ratio  # InferenceEngine.load 会读取它覆盖 config.DATASET_RATIO
    return bundle
