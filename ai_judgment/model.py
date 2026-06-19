# coding: utf-8
"""
DPSG 模型架构 —— 推理可用、import 无副作用版 (Module 1)
=====================================================

为什么需要这个文件？
  models/train_and_evaluation/GossipCop.py 在 **import 时**就会加载训练机上的
  绝对路径数据并执行完整训练循环（见该文件 36-230 行与 989 行往后），无法被
  API 进程安全引用。因此这里把模型架构**原样抽取**为一个纯净模块：

  1. import 不产生任何副作用（不读数据、不训练）。
  2. 数据依赖（邻接表 adj_list、各类 embedding 字典、邻居字典）改为**注入式**，
     通过 model.bind_data_store(...) 在推理引擎里挂载，而非模块级全局。
  3. forward(..., return_internals=True) 在原有 (prediction, dist) 之外，额外
     返回 ModelInternals：结构冲突所需的 align_c/align_g、三模态表示能量、
     邻居规模等中间变量，供 internals.compute_judgment_metrics 使用。
  4. 修正推理 bug：原 Bi_RNN 的高斯噪声注入未用 self.training 门控，导致 eval
     时仍加噪声、结果不确定；此处统一用 self.training 门控，保证推理可复现。

架构本身（PositionalEncoding / GraphAttentionLayer / Signed_GAT /
TransformerBlock / DPSG）与 GossipCop.py 保持一致，便于直接 load_state_dict
加载已训练权重。后续可让训练脚本反向 import 本模块以消除重复定义。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import scipy.sparse as sp


# ===========================================================================
# 中间变量容器
# ===========================================================================
@dataclass
class ModelInternals:
    """forward(return_internals=True) 额外返回的全部中间量。"""
    probability: float
    dist_mse: float                       # MSELoss(align_c, align_g) == loss_dis
    rep_scale: float                      # align_c/align_g 平均能量，用于归一化
    modality_energy: Dict[str, float] = field(default_factory=dict)
    num_neighbors: int = 0


# ===========================================================================
# 基础组件（与 GossipCop.py 一致）
# ===========================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, device="cpu"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.wtrans = nn.Parameter(torch.zeros(size=(2 * out_features, out_features)))
        nn.init.xavier_uniform_(self.wtrans.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        h = torch.mm(inp, self.W)
        Wh1 = torch.mm(h, self.a[: self.out_features, :])
        Wh2 = torch.mm(h, self.a[self.out_features:, :])
        e = self.leakyrelu(Wh1 + Wh2.T)
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        negative_attention = torch.where(adj > 0, -e, zero_vec)
        attention = F.softmax(attention, dim=1)
        negative_attention = -F.softmax(negative_attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        negative_attention = F.dropout(negative_attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, inp)
        h_prime_negative = torch.matmul(negative_attention, inp)
        h_prime_double = torch.cat([h_prime, h_prime_negative], dim=1)
        new_h_prime = torch.mm(h_prime_double, self.wtrans)
        return F.elu(new_h_prime) if self.concat else new_h_prime


class Signed_GAT(nn.Module):
    def __init__(self, nb_heads=1, dropout=0, alpha=0.3, device="cpu"):
        super().__init__()
        self.dropout = dropout
        self.nb_heads = nb_heads
        self.alpha = alpha
        self.device = device

    def forward(self, node_embedding, cosmatrix, original_adj, X_tid):
        device = node_embedding.device
        embedding_dim = node_embedding.shape[1]
        node_num = original_adj.shape[0]
        user_tweet_embedding = nn.Embedding(num_embeddings=node_num, embedding_dim=embedding_dim, padding_idx=0).to(device)
        user_tweet_embedding.from_pretrained(node_embedding)
        original_adj = torch.from_numpy(original_adj.astype(np.float64)).to(device)
        potentinal_adj = torch.where(cosmatrix > 0.5, torch.ones_like(cosmatrix), torch.zeros_like(cosmatrix)).to(device)
        adj = original_adj + potentinal_adj
        adj = torch.where(adj > 0, torch.ones_like(adj), torch.zeros_like(adj))
        self.attentions = [
            GraphAttentionLayer(embedding_dim, embedding_dim, dropout=self.dropout, alpha=self.alpha, concat=True, device=device)
            for _ in range(self.nb_heads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)
        out_att = GraphAttentionLayer(embedding_dim * self.nb_heads, embedding_dim, dropout=self.dropout, alpha=self.alpha, concat=False, device=device)
        X = user_tweet_embedding(torch.arange(0, node_num).long().to(device)).to(torch.float32)
        x = F.dropout(X, self.dropout, training=self.training)
        adj = adj.to(torch.float32)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.sigmoid(out_att(x, adj))
        return x, x[X_tid]


class TransformerBlock(nn.Module):
    def __init__(self, input_size, d_k=16, d_v=16, n_heads=8, is_layer_norm=False, attn_dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else input_size
        self.d_v = d_v if d_v is not None else input_size
        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)
        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * d_v))
        self.W_o = nn.Parameter(torch.Tensor(d_v * n_heads, input_size))
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(attn_dropout)
        self.__init_weights__()

    def __init_weights__(self):
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v)
        init.xavier_normal_(self.W_o)
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        return self.dropout(self.linear2(F.relu(self.linear1(X))))

    def scaled_dot_product_attention(self, Q, K, V, episilon=1e-6):
        temperature = self.d_k ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)
        Q_K_score = self.dropout(F.softmax(Q_K, dim=-1))
        return Q_K_score.bmm(V)

    def multi_head_attention(self, Q, K, V):
        bsz, q_len, _ = Q.size()
        _, k_len, _ = K.size()
        _, v_len, _ = V.size()
        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)
        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_v)
        V_att = self.scaled_dot_product_attention(Q_, K_, V_)
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads * self.d_v)
        return self.dropout(V_att.matmul(self.W_o))

    def forward(self, Q, K, V):
        V_att = self.multi_head_attention(Q, K, V)
        if self.is_layer_norm:
            X = self.layer_morm(Q + V_att)
            output = self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X
        return output


# ===========================================================================
# 数据注入容器：把原本的模块级全局改为可注入对象
# ===========================================================================
@dataclass
class DataStore:
    """推理时由 InferenceEngine 构建并注入模型的数据依赖。"""
    adj_list: dict
    neighbor_dict: tuple            # neighbor_loader 的返回元组
    news_emb_dict: dict
    post_emb_dict: dict
    user_emb_dict: dict


# ===========================================================================
# DPSG 主模型
# ===========================================================================
class DPSG(nn.Module):
    def __init__(self, input_dim, n_hidden_dim, u_hidden_dim, p_hidden_dim,
                 out_embed_d, d_model=512, attn_heads=2, enc_layers=1, outemb_d=1,
                 content_dict=None, npu=30, self_attn_heads=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = out_embed_d
        self.embed_d = out_embed_d
        self.n_input_dim = out_embed_d
        self.n_hidden_dim = n_hidden_dim
        self.n_output_dim = out_embed_d
        self.u_input_dim = out_embed_d
        self.u_hidden_dim = u_hidden_dim
        self.u_output_dim = out_embed_d
        self.p_input_dim = out_embed_d
        self.p_hidden_dim = p_hidden_dim
        self.p_output_dim = out_embed_d
        self.d_model = d_model
        self.out_embed_d = out_embed_d
        self.outemb_d = outemb_d
        self.content_dict = content_dict or {}
        self.npu = npu

        # —— 注入式数据依赖（默认 None，推理前必须 bind_data_store）——
        self._store: Optional[DataStore] = None

        self.pos_encoder = PositionalEncoding(d_model, dropout=0.1, max_len=200)
        self.pos_decoder = PositionalEncoding(d_model, dropout=0.1, max_len=200)
        self.type_encoder = nn.Embedding(3, d_model, padding_idx=0)
        self.n_neigh_att = nn.Parameter(torch.ones(self.embed_d * 2, 1), requires_grad=True)
        self.p_neigh_att = nn.Parameter(torch.ones(self.embed_d * 2, 1), requires_grad=True)
        self.u_neigh_att = nn.Parameter(torch.ones(self.embed_d * 2, 1), requires_grad=True)
        self.register_parameter("bias", None)
        self.transformer = nn.Transformer(d_model=d_model, nhead=attn_heads, num_encoder_layers=enc_layers,
                                          num_decoder_layers=1, dim_feedforward=512, dropout=0.1, activation="relu")

        self.init_linear_text = nn.Linear(self.input_dim[0], self.hidden_dim)
        self.init_linear_image = nn.Linear(self.input_dim[1], self.hidden_dim)
        self.init_linear_time = nn.Linear(1, self.hidden_dim)
        self.init_linear_other_p = nn.Linear(self.input_dim[2], self.hidden_dim)
        self.init_linear_other_user = nn.Linear(self.input_dim[3], self.hidden_dim)

        self.post_content_attention_other = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout=0.2)
        self.news_title_attention_text = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout=0.2)
        self.news_content_attention_text = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout=0.2)
        self.post_content_attention_text = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout=0.2)
        self.user_content_attention_text = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout=0.2)
        self.attention_image = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout=0.2)
        self.attention_time = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout=0.2)
        self.attention_other = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout=0.2)
        self.layernorm1 = nn.LayerNorm([1, out_embed_d])
        self.layernorm2 = nn.LayerNorm([1, out_embed_d])
        self.layernorm3 = nn.LayerNorm([1, out_embed_d])
        self.layernorm4 = nn.LayerNorm([1, out_embed_d])
        self.layernorm5 = nn.LayerNorm([1, out_embed_d])
        self.layernorm6 = nn.LayerNorm([1, out_embed_d])

        self.n_init_linear = nn.Linear(self.n_input_dim, self.n_hidden_dim)
        self.n_attention = nn.MultiheadAttention(self.n_hidden_dim, self_attn_heads, dropout=0.2)
        self.n_linear = nn.Linear(self.n_hidden_dim, self.n_output_dim)
        self.u_init_linear = nn.Linear(self.u_input_dim, self.u_hidden_dim)
        self.u_attention = nn.MultiheadAttention(self.u_hidden_dim, self_attn_heads, dropout=0.2)
        self.u_linear = nn.Linear(self.u_hidden_dim, self.u_output_dim)
        self.p_init_linear = nn.Linear(self.p_input_dim, self.p_hidden_dim)
        self.p_attention = nn.MultiheadAttention(self.p_hidden_dim, self_attn_heads, dropout=0.2)
        self.p_linear = nn.Linear(self.p_hidden_dim, self.p_output_dim)
        self.act = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.out_dropout = nn.Dropout(p=0.25)
        self.out_linear = nn.Linear(self.out_embed_d, self.outemb_d)
        self.output_act = nn.Sigmoid()
        self.mh_attention1 = TransformerBlock(input_size=self.hidden_dim, n_heads=8, attn_dropout=0)
        self.mh_attention = TransformerBlock(input_size=self.hidden_dim, n_heads=8, attn_dropout=0, is_layer_norm=True)
        self.alignfc_g = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.alignfc_c = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(0.6)
        self.gat_relation = Signed_GAT()

        # 单次 forward 期间的模态能量缓存（return_internals 时填充）
        self._modality_energy: Dict[str, float] = {}

    # -------------------- 数据注入 --------------------
    def bind_data_store(self, store: DataStore):
        """挂载推理所需的数据依赖（邻接表 / embedding 字典 / 邻居字典）。"""
        self._store = store
        return self

    @property
    def store(self) -> DataStore:
        if self._store is None:
            raise RuntimeError("DataStore 未注入：请先调用 model.bind_data_store(store)。")
        return self._store

    # -------------------- 基础方法 --------------------
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)

    def build_symmetric_adjacency_matrix(self, edges, shape):
        def normalize_adj(mx):
            rowsum = np.array(mx.sum(1))
            r_inv_sqrt = np.power(rowsum, -0.5).flatten()
            r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
            r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
            return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
        adj = sp.coo_matrix((edges[:, 2], (edges[:, 0], edges[:, 1])), shape=shape, dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        return adj.tocoo()

    def calculate_cos_matrix(self, node_embedding):
        a, b = node_embedding, node_embedding.T
        c = torch.mm(a, b)
        aa = torch.mul(a, a)
        bb = torch.mul(b, b)
        asum = torch.sqrt(torch.sum(aa, dim=1, keepdim=True))
        bsum = torch.sqrt(torch.sum(bb, dim=0, keepdim=True))
        return torch.div(c, torch.mm(asum, bsum))

    def _device(self):
        return next(self.parameters()).device

    def Bi_RNN(self, neighbor_id, neighbor_time, node_type):
        """与 GossipCop.py 的 Bi_RNN 一致，但：(1) 数据字典取自注入的 store；
        (2) 噪声注入用 self.training 门控，保证推理可复现；
        (3) return_internals 时顺带记录模态表示能量。"""
        device = self._device()
        store = self.store

        def _record(key, tensor):
            if getattr(self, "_capture_internals", False):
                self._modality_energy[key] = self._modality_energy.get(key, 0.0) + \
                    float(tensor.detach().abs().mean().cpu())

        if node_type == "news":
            input_title, input_content, input_image = [], [], []
            for i in neighbor_id:
                if ("news", i) not in self.content_dict:
                    input_title.append(store.news_emb_dict[i][0])
                    input_content.append(store.news_emb_dict[i][1])
                    input_image.append(store.news_emb_dict[i][2])
            input_title = torch.Tensor(input_title).to(device)
            input_image = torch.Tensor(input_image).to(device)
            linear_input_title = self.init_linear_text(input_title)
            linear_input_image = self.init_linear_image(input_image)
            _record("text", linear_input_title)   # 文本模态能量（噪声注入前的干净表示）
            _record("image", linear_input_image)  # 图像模态能量
            linear_input_title = linear_input_title.view(linear_input_title.shape[0], 1, linear_input_title.shape[1])
            mean_pooling = linear_input_title.squeeze(1)
            if self.training:
                mean_pooling = mean_pooling[torch.randperm(mean_pooling.size(0))]
                mean_pooling = mean_pooling + torch.randn_like(mean_pooling) * 0.5
                mean_pooling = F.dropout(mean_pooling, p=0.6, training=True)
            return mean_pooling

        if node_type == "post":
            input_a = []
            for i in neighbor_id:
                if ("post", i) not in self.content_dict:
                    input_a.append(store.post_emb_dict[i][1])
            input_a = torch.Tensor(input_a).to(device)
            linear_input_text = self.init_linear_text(input_a)
            _record("text", linear_input_text)
            linear_input_text = linear_input_text.view(linear_input_text.shape[0], 1, linear_input_text.shape[1])
            mean_pooling = linear_input_text.squeeze(1)
            if self.training:
                mean_pooling = mean_pooling[torch.randperm(mean_pooling.size(0))]
                mean_pooling = mean_pooling + torch.randn_like(mean_pooling) * 0.5
                mean_pooling = F.dropout(mean_pooling, p=0.6, training=True)
            return mean_pooling

        if node_type == "user":
            input_b = []
            for i in neighbor_id:
                if ("user", i) not in self.content_dict:
                    input_b.append(store.user_emb_dict[i][1])
            input_b = torch.Tensor(input_b).to(device)
            linear_input_text = self.init_linear_text(input_b)
            _record("text", linear_input_text)
            linear_input_text = linear_input_text.view(linear_input_text.shape[0], 1, linear_input_text.shape[1])
            mean_pooling = linear_input_text.squeeze(1)
            if self.training:
                mean_pooling = mean_pooling[torch.randperm(mean_pooling.size(0))]
                mean_pooling = mean_pooling + torch.randn_like(mean_pooling) * 0.5
                mean_pooling = F.dropout(mean_pooling, p=0.6, training=True)
            return mean_pooling

    def SameType_Agg_Bi_RNN(self, neighbor_id, neighbor_time, node_type):
        content_embedings = self.Bi_RNN(neighbor_id, neighbor_time, node_type)
        return {neighbor_id[i]: content_embedings[i] for i in range(len(neighbor_id))}

    def transformer_agg(self, het_node, neighbor_order_n_p_u, all_t0, npu=30):
        device = self._device()
        adj_list = self.store.adj_list

        all_t = [float(i.split("t")[-1]) for i in all_t0]
        all_tt = torch.tensor(all_t[1:]) - all_t[0]
        all_t_normalized = (all_tt - all_tt.min()) / max((all_tt.max() - all_tt.min()), 1e-8)
        all_tt_exp = torch.exp(-all_t_normalized).to(device)
        all_tt_exp = (all_tt_exp - all_tt_exp.min()) / max((all_tt_exp.max() - all_tt_exp.min()), 1e-8)

        # 时间模态能量：温度衰减权重经 init_linear_time 投影后的能量
        if getattr(self, "_capture_internals", False):
            t_proj = self.init_linear_time(all_tt_exp.view(-1, 1))
            self._modality_energy["time"] = float(t_proj.detach().abs().mean().cpu())

        node_embedding = [self.Bi_RNN([het_node.node_id], [0], het_node.node_type)[0]]
        node_embedding_n = [node_embedding[0]]
        now_node = ["n" + het_node.node_id]
        now_node_n = ["n" + het_node.node_id]

        post_neighbor, user_neighbor, news_neighbor = [], [], []
        for item in neighbor_order_n_p_u[:npu]:
            if item[0] == "p":
                post_neighbor.append(item[1])
            elif item[0] == "u":
                user_neighbor.append(item[1])
            elif item[0] == "n":
                news_neighbor.append(item[1])

        n_aft_rnn_dict, u_aft_rnn_dict, p_aft_rnn_dict = {}, {}, {}
        if news_neighbor:
            n_aft_rnn_dict = self.SameType_Agg_Bi_RNN(news_neighbor, None, "news")
            for nid, value in n_aft_rnn_dict.items():
                now_node.append("n" + nid); now_node_n.append("n" + nid)
                node_embedding.append(value); node_embedding_n.append(value)
        if user_neighbor:
            u_aft_rnn_dict = self.SameType_Agg_Bi_RNN(user_neighbor, None, "user")
            for uid, value in u_aft_rnn_dict.items():
                now_node.append("u" + uid); node_embedding.append(value)
        if post_neighbor:
            p_aft_rnn_dict = self.SameType_Agg_Bi_RNN(post_neighbor, None, "post")
            for pid, value in p_aft_rnn_dict.items():
                now_node.append("p" + pid); node_embedding.append(value)

        node_embedding = torch.stack(node_embedding)
        node_embedding_n = torch.stack(node_embedding_n)
        self._num_neighbors = len(neighbor_order_n_p_u[:npu])

        now_adj_list_old = {}
        for node in now_node:
            try:
                c = list(set(adj_list[node]) & set(now_node))
            except KeyError:
                c = list(set(now_node))
            if c:
                now_adj_list_old[node] = c
        node_id2_xulie = {nid: i for i, nid in enumerate(now_node)}
        now_node_n = [node_id2_xulie[i] for i in now_node_n]
        num_node = len(node_id2_xulie)
        now_adj_list = {}
        for nid, v in now_adj_list_old.items():
            i = node_id2_xulie[nid]
            now_adj_list[i] = [node_id2_xulie[jid] for jid in v]
        relation = np.array([[i, dst, "1"] for i, v in now_adj_list.items() for dst in v])
        adj = self.build_symmetric_adjacency_matrix(edges=relation, shape=(num_node, num_node))
        original_adj = np.zeros(shape=adj.shape)
        for i, v in now_adj_list.items():
            original_adj[int(i), [int(e) for e in v]] = 1

        self.cosmatrix = self.calculate_cos_matrix(node_embedding)
        gx, graph_feature = self.gat_relation.forward(node_embedding=node_embedding, cosmatrix=self.cosmatrix,
                                                      original_adj=original_adj, X_tid=now_node_n)
        graph_feature_all = [gx[0]]
        for type_id in neighbor_order_n_p_u[:npu]:
            graph_feature_all.append(gx[now_node.index(type_id[0] + type_id[1])])
        graph_feature_all = torch.stack(graph_feature_all)

        bsz = len(now_node_n)
        bsz_all = len(neighbor_order_n_p_u[:npu]) + 1
        self_att_g = self.mh_attention1(graph_feature.view(bsz, -1, self.hidden_dim), graph_feature.view(bsz, -1, self.hidden_dim), graph_feature.view(bsz, -1, self.hidden_dim))
        self_att_c = self.mh_attention1(node_embedding_n.view(bsz, -1, self.hidden_dim), node_embedding_n.view(bsz, -1, self.hidden_dim), node_embedding_n.view(bsz, -1, self.hidden_dim))
        self_att_g_all = self.mh_attention1(graph_feature_all.view(bsz_all, -1, self.hidden_dim), graph_feature_all.view(bsz_all, -1, self.hidden_dim), graph_feature_all.view(bsz_all, -1, self.hidden_dim))
        align_c = self.alignfc_c(self_att_c).view(bsz, self.hidden_dim)
        align_g = self.alignfc_g(self_att_g).view(bsz, self.hidden_dim)
        align_g_all = self.alignfc_g(self_att_g_all).view(bsz_all, self.hidden_dim)
        dist = [align_c, align_g]

        encoder_emb_input, encoder_type_input = [self.Bi_RNN([het_node.node_id], [0], het_node.node_type)[0]], [0]
        decoder_emb_input, decoder_type_input, decoder_order_input = [node_embedding_n[0]], [0], []
        for i, type_id in enumerate(neighbor_order_n_p_u[:npu]):
            if type_id[0] == "p":
                encoder_emb_input.append(p_aft_rnn_dict[type_id[1]]); encoder_type_input.append(1)
            elif type_id[0] == "u":
                encoder_emb_input.append(u_aft_rnn_dict[type_id[1]]); encoder_type_input.append(2)
            else:
                encoder_emb_input.append(n_aft_rnn_dict[type_id[1]]); encoder_type_input.append(0)
        encoder_type_input = torch.LongTensor(encoder_type_input).to(device)
        for i, type_id in enumerate(news_neighbor):
            decoder_emb_input.append(n_aft_rnn_dict[type_id]); decoder_order_input.append(i + 1); decoder_type_input.append(0)
        decoder_type_input = torch.LongTensor(decoder_type_input).to(device)
        encoder_emb_input = torch.stack(encoder_emb_input)
        self_att_c_all = self.mh_attention1(encoder_emb_input.view(bsz_all, -1, self.hidden_dim), encoder_emb_input.view(bsz_all, -1, self.hidden_dim), encoder_emb_input.view(bsz_all, -1, self.hidden_dim))
        align_c_all = self.alignfc_c(self_att_c_all).view(bsz_all, self.hidden_dim)
        encoder_emb_input = align_c_all.view(align_c_all.shape[0], 1, align_c_all.shape[1])
        decoder_emb_input = align_c.view(align_c.shape[0], 1, align_c.shape[1])
        encoder_emb_input = encoder_emb_input + self.type_encoder(encoder_type_input).view(encoder_emb_input.shape[0], 1, encoder_emb_input.shape[2])
        encoder_emb_input = encoder_emb_input + self.pos_encoder(encoder_emb_input)
        encoder_emb_input = encoder_emb_input + align_g_all.view(encoder_emb_input.shape[0], 1, encoder_emb_input.shape[2])
        decoder_emb_input = decoder_emb_input + self.pos_encoder(decoder_emb_input)
        decoder_emb_input = decoder_emb_input + self.type_encoder(decoder_type_input).view(decoder_emb_input.shape[0], 1, decoder_emb_input.shape[2])
        decoder_emb_input = decoder_emb_input + align_g.view(decoder_emb_input.shape[0], 1, decoder_emb_input.shape[2])
        final_representation = self.transformer(encoder_emb_input, decoder_emb_input)
        return final_representation, dist

    def output(self, c_embed_batch):
        c_embed = c_embed_batch[0, 0, :].view(1, 1, self.out_embed_d)
        c_embed = self.out_dropout(c_embed)
        return self.output_act(self.out_linear(c_embed))

    def forward(self, x, neighbor_order_n_p_u, all_t, return_internals: bool = False):
        """
        原签名兼容：return_internals=False 时仍返回 (predictions, dist)，
        训练脚本无需任何改动即可继续使用。

        return_internals=True 时额外返回 ModelInternals，封装：
          - 假新闻概率 sigmoid 输出
          - 结构冲突 MSE(align_c, align_g)（== loss_dis）及归一化尺度
          - 三模态表示能量 {text,image,time}
          - 参与聚合的邻居规模
        """
        self._capture_internals = return_internals
        self._modality_energy = {}
        self._num_neighbors = 0

        rep, dist = self.transformer_agg(x, neighbor_order_n_p_u, all_t, npu=self.npu)
        predictions = self.output(c_embed_batch=rep)

        if not return_internals:
            return predictions, dist

        align_c, align_g = dist[0], dist[1]
        dist_mse = F.mse_loss(align_c, align_g).item()
        rep_scale = float(0.5 * (align_c.detach().pow(2).mean() + align_g.detach().pow(2).mean()).cpu())
        internals = ModelInternals(
            probability=float(predictions.view(-1)[0].detach().cpu()),
            dist_mse=dist_mse,
            rep_scale=rep_scale,
            modality_energy=dict(self._modality_energy),
            num_neighbors=int(self._num_neighbors),
        )
        return predictions, dist, internals


def build_model(hparams: dict, device: str = "cpu") -> DPSG:
    """按配置构造 DPSG（不加载权重）。"""
    model = DPSG(
        input_dim=hparams["input_dim"],
        n_hidden_dim=hparams["n_hidden_dim"],
        u_hidden_dim=hparams["u_hidden_dim"],
        p_hidden_dim=hparams["p_hidden_dim"],
        out_embed_d=hparams["out_embed_d"],
        d_model=hparams["d_model"],
        attn_heads=hparams["attn_heads"],
        enc_layers=hparams["enc_layers"],
        npu=hparams["npu"],
    )
    return model.to(device)
