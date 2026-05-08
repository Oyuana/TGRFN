#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from os import path
from torch.autograd import Variable
import math
import random
import os
import random
import json
import scipy.sparse as sp
import torch.nn.functional as F
import torch.nn.init as init
import time

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("device:", device)

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# ================= 核心类定义 (保持原样) =================
class Het_Node():
    def __init__(self, node_type, node_id, embed, neighbor_list_news=[], neighbor_list_post=[], neighbor_list_user=[], label=None):
        self.node_type = node_type
        self.node_id = node_id
        self.emb = embed
        self.label = label
        self.neighbors_news = neighbor_list_news
        self.neighbors_post = neighbor_list_post
        self.neighbors_user = neighbor_list_user

def neighbor_loader(pathway):
    neighbor_dict_n_p_u = {}
    neighbor_dict_n = {}
    neighbor_dict_n_p_u_add_time={}
    neighbor_dict_n_add_time = {}
    neighbor_dict_news = {}
    neighbor_dict_post = {}
    neighbor_dict_user = {}
    f = open(pathway)
    Lines = f.readlines()
    for i in range(len(Lines)):
        neighbor_list = Lines[i].split()
        neighbor_dict_n_p_u_add_time[neighbor_list[0][1:-1]] = [item[1:] for item in neighbor_list[1:] if item[1:] != 'PADDING']
        neighbor_dict_n_p_u[(neighbor_list[0][1:-1]).split("t")[0]]=[(item[0], (item[1:]).split("t")[0]) for item in neighbor_list[1:] if item[1:]!='PADDING']
        neighbor_dict_n_add_time[neighbor_list[0][1:-1]] = [(item[0], item[1:]) for item in neighbor_list[1:] if item[0] == 'n' and item[1:] != 'PADDING']
        neighbor_dict_n[(neighbor_list[0][1:-1]).split("t")[0]]=[(item[0], (item[1:]).split("t")[0]) for item in neighbor_list[1:] if item[0] == 'n' and item[1:]!='PADDING']
        neighbor_dict_news[neighbor_list[0][1:-1]] = [item[1:] for item in neighbor_list[1:] if item[0] == 'n' and item[1:] != 'PADDING']
        neighbor_dict_user[neighbor_list[0][1:-1]] = [item[1:] for item in neighbor_list[1:] if item[0] == 'u' and item[1:] != 'PADDING']
        neighbor_dict_post[neighbor_list[0][1:-1]] = [item[1:] for item in neighbor_list[1:] if item[0] == 'p' and item[1:] != 'PADDING']
    key_mapping = dict()
    for i, j in zip(neighbor_dict_n_p_u_add_time.keys(), neighbor_dict_n_p_u.keys()):
        key_mapping[j] = i
    return neighbor_dict_n_p_u, neighbor_dict_n, key_mapping,neighbor_dict_n_p_u_add_time,neighbor_dict_n_add_time,neighbor_dict_news, neighbor_dict_post, neighbor_dict_user

def data_loader(pathway='/home/zhangrq/DPSG/data/processed_data/PHEME/pheme_n5_p5_u100/normalized_news_nodes/', node_type="news"):
    if node_type == "news":
        news_node = []
        news_id = []
        news_label = []
        news_embed = []
        news_n_neigh = []
        news_p_neigh = []
        news_u_neigh = []
        if not os.path.exists(pathway): return []
        for i in range(len(os.listdir(pathway))):
            batch = str(i)
            f = open(pathway + "batch_" + batch + '.txt')
            Lines = f.readlines()
            for j in range(len(Lines)):
                if j % 5 == 0:
                    _, id_, label = Lines[j].split()
                    news_id.append(id_); news_label.append(int(label)); embed = []
                if j % 5 == 1:
                    embed.append(list(map(float, Lines[j].split()))); news_embed.append(embed)
                if j % 5 == 2: news_n_neigh.append(Lines[j].split())
                if j % 5 == 3: news_p_neigh.append(Lines[j].split())
                if j % 5 == 4: news_u_neigh.append(Lines[j].split())
            f.close()
        for i in range(len(news_id)):
            node = Het_Node(node_type="news", node_id=news_id[i], embed=news_embed[i], neighbor_list_news=news_n_neigh[i], neighbor_list_post=news_p_neigh[i], neighbor_list_user=news_u_neigh[i], label=news_label[i])
            news_node.append(node)
        return news_node
    elif node_type == 'post':
        post_node = []
        post_id = []
        post_embed = []
        if not os.path.exists(pathway): return []
        for i in range(len(os.listdir(pathway))):
            batch = str(i)
            f = open(pathway + "batch_" + batch + '.txt')
            Lines = f.readlines()
            for j in range(len(Lines)):
                if j % 2 == 0: _, id_ = Lines[j].split(); post_id.append(id_); embed = []
                if j % 2 == 1: embed.append(list(map(float, Lines[j].split()))); post_embed.append(embed)
            f.close()
        for i in range(len(post_id)):
            node = Het_Node(node_type="post", node_id=post_id[i], embed=post_embed[i]); post_node.append(node)
        return post_node
    else:
        user_node = []
        user_id = []
        user_embed = []
        if not os.path.exists(pathway): return []
        for i in range(len(os.listdir(pathway))):
            batch = str(i)
            f = open(pathway + "batch_" + batch + '.txt')
            Lines = f.readlines()
            for j in range(len(Lines)):
                if j % 3 == 0: id_ = Lines[j].split(); user_id.append(id_[1]); embed = []
                if j % 3 == 1 or j % 3 == 2: embed.append(list(map(float, Lines[j].split())))
                if j % 3 == 2: user_embed.append(embed)
            f.close()
        for i in range(len(user_id)):
            node = Het_Node(node_type="user", node_id=user_id[i], embed=user_embed[i]); user_node.append(node)
        return user_node

# ================= 模型类 (确保完全一致) =================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]; return self.dropout(x)

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features; self.out_features = out_features; self.dropout = dropout; self.alpha = alpha; self.concat = concat
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features))); nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1))); nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W); N = Wh.size()[0]
        a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1), Wh.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1); attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime) if self.concat else h_prime

class Signed_GAT(nn.Module):
    def __init__(self, embedding_dim, nb_heads=1, dropout=0.0, alpha=0.3, max_nodes=10000):
        super(Signed_GAT, self).__init__()
        self.dropout = dropout; self.nb_heads = nb_heads; self.alpha = alpha; self.embedding_dim = embedding_dim; self.max_nodes = max_nodes
        self.user_tweet_embedding = nn.Embedding(self.max_nodes, self.embedding_dim, padding_idx=0)
        self.attentions = nn.ModuleList([GraphAttentionLayer(embedding_dim, embedding_dim, dropout=self.dropout, alpha=self.alpha, concat=True) for _ in range(self.nb_heads)])
        self.out_att = GraphAttentionLayer(embedding_dim * self.nb_heads, embedding_dim, dropout=self.dropout, alpha=self.alpha, concat=False)
    def reset_embedding(self, num_nodes, embedding_dim, device):
        self.user_tweet_embedding = nn.Embedding(num_nodes, embedding_dim, padding_idx=0).to(device); self.embedding_dim = embedding_dim
    def forward(self, node_embedding, cosmatrix, original_adj, X_tid):
        device = node_embedding.device; num_nodes = node_embedding.size(0)
        if self.user_tweet_embedding.weight.size(0) != num_nodes or self.user_tweet_embedding.weight.size(1) != node_embedding.size(1): self.reset_embedding(num_nodes, node_embedding.size(1), device)
        with torch.no_grad(): self.user_tweet_embedding.weight.copy_(node_embedding)
        original_adj = torch.from_numpy(original_adj.astype(np.float32)).to(device)
        potential_adj = torch.where(cosmatrix > 0.5, torch.ones_like(cosmatrix, device=device), torch.zeros_like(cosmatrix, device=device))
        adj = original_adj + potential_adj; adj = torch.where(adj > 0, torch.ones_like(adj, device=device), torch.zeros_like(adj, device=device))
        idx_tensor = torch.arange(num_nodes, device=device)
        X = self.user_tweet_embedding(idx_tensor); x = F.dropout(X, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1); x = F.dropout(x, self.dropout, training=self.training)
        x = torch.sigmoid(self.out_att(x, adj))
        if isinstance(X_tid, list): X_tid = X_tid[0] if len(X_tid) == 1 else torch.LongTensor(X_tid).to(device)
        elif isinstance(X_tid, torch.Tensor): X_tid = X_tid.to(device)
        return x, x[X_tid] if isinstance(X_tid, int) else x.index_select(0, X_tid)

class TransformerBlock(nn.Module):
    def __init__(self, input_size, d_k=16, d_v=16, n_heads=8, is_layer_norm=False, attn_dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.n_heads = n_heads; self.d_k = d_k; self.d_v = d_v; self.is_layer_norm = is_layer_norm
        if is_layer_norm: self.layer_morm = nn.LayerNorm(normalized_shape=input_size)
        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * d_k)); self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * d_v)); self.W_o = nn.Parameter(torch.Tensor(d_v*n_heads, input_size))
        self.linear1 = nn.Linear(input_size, input_size); self.linear2 = nn.Linear(input_size, input_size); self.dropout = nn.Dropout(attn_dropout); self.__init_weights__()
    def __init_weights__(self):
        init.xavier_normal_(self.W_q); init.xavier_normal_(self.W_k); init.xavier_normal_(self.W_v); init.xavier_normal_(self.W_o)
        init.xavier_normal_(self.linear1.weight); init.xavier_normal_(self.linear2.weight)
    def FFN(self, X):
        return self.dropout(self.linear2(F.relu(self.linear1(X))))
    def multi_head_attention(self, Q, K, V):
        bsz, q_len, _ = Q.size(); bsz, k_len, _ = K.size(); bsz, v_len, _ = V.size()
        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k).permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k).permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v).permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_v)
        V_att = self.dropout(F.softmax(torch.einsum("bqd,bkd->bqk", Q_, K_) / (self.d_k ** 0.5 + 1e-6), dim=-1)).bmm(V_).view(bsz, self.n_heads, q_len, self.d_v).permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads*self.d_v)
        return self.dropout(V_att.matmul(self.W_o))
    def forward(self, Q, K, V):
        X = self.layer_morm(Q + self.multi_head_attention(Q, K, V)) if self.is_layer_norm else Q + self.multi_head_attention(Q, K, V)
        return self.layer_morm(self.FFN(X) + X) if self.is_layer_norm else self.FFN(X) + X

class DPSG(nn.Module):
    def __init__(self, input_dim, n_hidden_dim, u_hidden_dim, p_hidden_dim, out_embed_d, outemb_d=1, content_dict={}, attn_heads=8, d_model=104, self_attn_heads=8, enc_layers=1, dec_layers=1,max_nodes=10000):
        super(DPSG, self).__init__()
        self.input_dim = input_dim; self.hidden_dim = out_embed_d; self.embed_d = out_embed_d; self.n_input_dim = out_embed_d; self.n_hidden_dim = n_hidden_dim; self.n_output_dim = out_embed_d
        self.u_input_dim = out_embed_d; self.u_hidden_dim = u_hidden_dim; self.u_output_dim = out_embed_d; self.p_input_dim = out_embed_d; self.p_hidden_dim = p_hidden_dim; self.p_output_dim = out_embed_d
        self.out_embed_d = out_embed_d; self.outemb_d = outemb_d; self.content_dict = content_dict; self.d_model = d_model
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=0.1, max_len=201); self.pos_decoder = PositionalEncoding(self.d_model, dropout=0.1, max_len=201); self.type_encoder = nn.Embedding(3, self.d_model, padding_idx=0)
        self.transformer = nn.Transformer(d_model=self.d_model, nhead=attn_heads, num_encoder_layers=enc_layers, num_decoder_layers=dec_layers, dim_feedforward=512, dropout=0.1, activation='relu')
        self.init_linear_text = nn.Linear(self.input_dim[0], self.hidden_dim); self.init_linear_image = nn.Linear(self.input_dim[1], self.hidden_dim); self.init_linear_time = nn.Linear(1, self.hidden_dim); self.init_linear_feature = nn.Linear(self.input_dim[2], self.hidden_dim)
        self.news_content_attention_text = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout=0.4); self.post_content_attention_text = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout=0.4); self.user_content_attention_text = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout=0.4)
        self.attention_time = nn.MultiheadAttention(self.hidden_dim, attn_heads, dropout=0.4); self.attention_other_user = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout=0.4)
        self.layernorm1 = nn.LayerNorm([1, out_embed_d]); self.layernorm2 = nn.LayerNorm([1, out_embed_d]); self.layernorm3 = nn.LayerNorm([1, out_embed_d]); self.layernorm4 = nn.LayerNorm([1, out_embed_d]); self.layernorm6 = nn.LayerNorm([1, out_embed_d])
        self.n_init_linear = nn.Linear(self.n_input_dim, self.n_hidden_dim); self.n_attention = nn.MultiheadAttention(self.n_hidden_dim, self_attn_heads, dropout=0.4); self.n_linear = nn.Linear(self.n_hidden_dim, self.n_output_dim)
        self.u_init_linear = nn.Linear(self.u_input_dim, self.u_hidden_dim); self.u_attention = nn.MultiheadAttention(self.u_hidden_dim, self_attn_heads, dropout=0.4); self.u_linear = nn.Linear(self.u_hidden_dim, self.u_output_dim)
        self.p_init_linear = nn.Linear(self.p_input_dim, self.p_hidden_dim); self.p_attention = nn.MultiheadAttention(self.p_hidden_dim, self_attn_heads, dropout=0.4); self.p_linear = nn.Linear(self.p_hidden_dim, self.p_output_dim)
        self.act = nn.LeakyReLU(); self.relu = nn.ReLU(); self.softmax = nn.Softmax(dim=1); self.out_dropout = nn.Dropout(p=0.25); self.out_linear = nn.Linear(self.out_embed_d, self.outemb_d); self.output_act = nn.Sigmoid()
        self.mh_attention1 = TransformerBlock(input_size=self.hidden_dim, n_heads=8, attn_dropout=0); self.mh_attention = TransformerBlock(input_size=self.hidden_dim, n_heads=8, attn_dropout=0,is_layer_norm=True)
        self.alignfc_g = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim); self.alignfc_c = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim); self.fc3 = nn.Linear(2 * self.hidden_dim, self.hidden_dim); self.dropout = nn.Dropout(0.6)
        self.gat_relation = Signed_GAT(embedding_dim=out_embed_d, nb_heads=1, dropout=0.0, alpha=0.3, max_nodes=max_nodes)
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter): nn.init.xavier_normal_(m.weight.data); m.bias.data.fill_(0.1)
    def build_symmetric_adjacency_matrix(self,edges, shape):
        if edges.shape[0] == 0: adj = sp.coo_matrix((shape[0], shape[1]), dtype=np.float32)
        else: adj = sp.coo_matrix(arg1=(edges[:, 2], (edges[:, 0], edges[:, 1])), shape=shape, dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj); rowsum = np.array(adj.sum(1)); r_inv_sqrt = np.power(rowsum, -0.5).flatten(); r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.; r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
        return (adj.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt) + sp.eye(adj.shape[0])).tocoo()
    def calculate_cos_matrix(self, node_embedding):
        if node_embedding.dim() > 2: node_embedding = node_embedding.view(node_embedding.size(0), -1)
        elif node_embedding.dim() == 1: node_embedding = node_embedding.unsqueeze(0)
        a = node_embedding; b = node_embedding.transpose(0, 1); c = torch.mm(a, b); aa = a * a; bb = b * b
        return c / (torch.mm(torch.sqrt(torch.sum(aa, dim=1, keepdim=True)), torch.sqrt(torch.sum(bb, dim=0, keepdim=True))) + 1e-8)
    def Bi_RNN(self, neighbor_id, neighbor_time,node_type, post_emb_dict, user_emb_dict, news_emb_dict):
        new_id = []
        if node_type == "news":
            input_content = []
            for i in neighbor_id: 
                if ("news", i) not in self.content_dict: input_content.append(news_emb_dict[i][0]); new_id.append(i)
            if not input_content: return torch.zeros(1, self.hidden_dim).to(device)
            input_content = torch.Tensor(input_content).to(device); input_time = torch.Tensor(neighbor_time).to(device).view(-1, 1)
            l_c = self.init_linear_text(input_content).view(len(input_content), 1, -1); l_t = self.init_linear_time(input_time).view(len(input_time), 1, -1)
            att_c, _ = self.news_content_attention_text(l_c,l_c,l_c); att_t, _ = self.attention_time(l_t, l_t,l_t)
            concate = self.mh_attention(self.layernorm6(att_t), self.layernorm1(att_c), self.layernorm1(att_c))
        elif node_type == "user":
            input_a, input_b = [], []
            for i in neighbor_id:
                if i in user_emb_dict: input_a.append(user_emb_dict[i][0]); input_b.append(user_emb_dict[i][1]); new_id.append(i)
            if not input_a: return torch.zeros(1,self.hidden_dim).to(device)
            input_a = torch.Tensor(input_a).to(device); input_b = torch.Tensor(input_b).to(device)
            l_a = self.init_linear_feature(input_a).view(len(input_a), 1, -1); l_b = self.init_linear_text(input_b).view(len(input_b), 1, -1)
            att_a, _ = self.attention_other_user(l_a, l_a, l_a); att_b, _ = self.user_content_attention_text(l_b,l_b,l_b)
            concate = torch.cat((self.layernorm3(att_b), self.layernorm2(att_a)), 1)
        elif node_type == "post":
            input_a = []
            for i in neighbor_id: 
                if ("post", i) not in self.content_dict: input_a.append(post_emb_dict[i][0]); new_id.append(i)
            if not input_a: return torch.zeros(1,self.hidden_dim).to(device)
            input_a = torch.Tensor(input_a).to(device); input_time = torch.Tensor(neighbor_time).to(device).view(-1, 1)
            l_c = self.init_linear_text(input_a).view(len(input_a), 1, -1); l_t = self.init_linear_time(input_time).view(len(input_time), 1, -1)
            att_c, _ = self.post_content_attention_text(l_c,l_c,l_c); att_t, _ = self.attention_time(l_t, l_t,l_t)
            concate = self.mh_attention(self.layernorm6(att_t), self.layernorm4(att_c), self.layernorm4(att_c))
        return torch.mean(concate, 1).unsqueeze(0)
    def SameType_Agg_Bi_RNN(self, neighbor_id, neighbor_time, node_type):
        aft_rnn_dict = {}
        for i, nid in enumerate(neighbor_id): aft_rnn_dict[nid] = self.Bi_RNN([nid], [neighbor_time[i]], node_type, post_emb_dict, user_emb_dict, news_emb_dict).squeeze(0)
        return aft_rnn_dict
    def transformer_agg(self, het_node, neighbor_order_n_p_u, neighbor_order_n, all_t0, npu=15, n=5):
        all_t = [float(i.split('t')[-1]) for i in all_t0]
        all_tt_exp = F.softmax(-(torch.tensor(all_t[1:]) - all_t[0]), dim=0).to(device) if len(all_t)>1 else torch.tensor([]).to(device)
        c_agg = self.Bi_RNN([het_node.node_id], [0], het_node.node_type, post_emb_dict, user_emb_dict, news_emb_dict).squeeze(0)
        node_embedding = [c_agg]; node_embedding_n = [c_agg]; now_node = ["n" + het_node.node_id]; now_node_n = ["n" + het_node.node_id]
        p_neigh, p_time, u_neigh, u_time, n_neigh, n_time = [], [], [], [], [], []
        limit = min(len(neighbor_order_n_p_u), npu, len(all_tt_exp))
        for idx, item in enumerate(neighbor_order_n_p_u[:limit]):
            t = all_tt_exp[idx]
            if item[0]=='p': p_neigh.append(item[1]); p_time.append(t)
            elif item[0]=='u': u_neigh.append(item[1]); u_time.append(t)
            elif item[0]=='n': n_neigh.append(item[1]); n_time.append(t)
        n_rnn = self.SameType_Agg_Bi_RNN(n_neigh, n_time, "news") if n_neigh else {}
        u_rnn = self.SameType_Agg_Bi_RNN(u_neigh, u_time, "user") if u_neigh else {}
        p_rnn = self.SameType_Agg_Bi_RNN(p_neigh, p_time, "post") if p_neigh else {}
        for id, v in n_rnn.items(): now_node.append("n"+id); now_node_n.append("n"+id); node_embedding.append(v); node_embedding_n.append(v)
        for id, v in u_rnn.items(): now_node.append("u"+id); node_embedding.append(v)
        for id, v in p_rnn.items(): now_node.append("p"+id); node_embedding.append(v)
        for i in range(len(node_embedding)):
             if node_embedding[i].dim() == 2 and node_embedding[i].size(0) == 1: node_embedding[i] = node_embedding[i].squeeze(0)
        node_embedding = torch.stack(node_embedding); node_embedding_n = torch.stack(node_embedding_n)
        if node_embedding.dim() == 3 and node_embedding.size(1) == 1: node_embedding = node_embedding.squeeze(1)
        if node_embedding_n.dim() == 3 and node_embedding_n.size(1) == 1: node_embedding_n = node_embedding_n.squeeze(1)
        node_idx = {id: i for i, id in enumerate(now_node)}
        relation = []
        for id in now_node:
            if id in adj_list:
                for neighbor in list(set(adj_list[id]) & set(now_node)): relation.append([node_idx[id], node_idx[neighbor], 1])
        rel_arr = np.array(relation) if relation else np.zeros((0, 3))
        adj = self.build_symmetric_adjacency_matrix(rel_arr, (len(now_node), len(now_node)))
        orig_adj = np.zeros(adj.shape)
        
        # === 修复 Bug 的关键行 ===
        if len(relation) > 0: 
            # 使用 rel_arr 而不是 relation 进行切片
            orig_adj[rel_arr[:,0].astype(int), rel_arr[:,1].astype(int)] = 1
            
        self.cosmatrix = self.calculate_cos_matrix(node_embedding)
        X_tid = torch.LongTensor([node_idx[i] for i in now_node_n]).to(device)
        gx, graph_feature = self.gat_relation(node_embedding, self.cosmatrix, orig_adj, X_tid)
        feat_all = [gx[0]]
        for item in neighbor_order_n_p_u[:limit]:
            key = item[0]+item[1]
            if key in node_idx: feat_all.append(gx[node_idx[key]])
        feat_all = torch.stack(feat_all)
        bsz, bsz_all = len(now_node_n), feat_all.size(0)
        att_c = self.mh_attention(node_embedding_n.view(bsz,-1,self.hidden_dim), node_embedding_n.view(bsz,-1,self.hidden_dim), node_embedding_n.view(bsz,-1,self.hidden_dim))
        att_g = self.mh_attention(graph_feature.view(bsz,-1,self.hidden_dim), graph_feature.view(bsz,-1,self.hidden_dim), graph_feature.view(bsz,-1,self.hidden_dim))
        att_g_all = self.mh_attention1(feat_all.view(bsz_all,-1,self.hidden_dim), feat_all.view(bsz_all,-1,self.hidden_dim), feat_all.view(bsz_all,-1,self.hidden_dim))
        align_c = self.alignfc_c(att_c).view(bsz, self.hidden_dim); align_g = self.alignfc_g(att_g).view(bsz, self.hidden_dim); align_g_all = self.alignfc_g(att_g_all).view(bsz_all, self.hidden_dim)
        enc_input = [c_agg]; enc_type = [0]
        for item in neighbor_order_n_p_u[:limit]:
            if item[0]=='p' and item[1] in p_rnn: enc_input.append(p_rnn[item[1]]); enc_type.append(1)
            elif item[0]=='u' and item[1] in u_rnn: enc_input.append(u_rnn[item[1]]); enc_type.append(2)
            elif item[0]=='n' and item[1] in n_rnn: enc_input.append(n_rnn[item[1]]); enc_type.append(0)
        enc_input = torch.stack(enc_input); enc_type = torch.LongTensor(enc_type).to(device)
        dec_input = [c_agg]; dec_type = [0]
        for nid in n_neigh: 
            if nid in n_rnn: dec_input.append(n_rnn[nid]); dec_type.append(0)
        dec_input = torch.stack(dec_input); dec_type = torch.LongTensor(dec_type).to(device)
        bsz_enc = enc_input.size(0)
        att_c_all = self.mh_attention1(enc_input.view(bsz_enc,-1,self.hidden_dim), enc_input.view(bsz_enc,-1,self.hidden_dim), enc_input.view(bsz_enc,-1,self.hidden_dim))
        enc_input = self.alignfc_c(att_c_all).view(bsz_enc, 1, self.hidden_dim)
        dec_input = align_c.view(align_c.shape[0], 1, self.hidden_dim)
        if align_g_all.size(0) == bsz_enc: enc_input += align_g_all.view(bsz_enc, 1, self.hidden_dim)
        enc_input += self.type_encoder(enc_type).view(bsz_enc, 1, self.hidden_dim) + self.pos_encoder(enc_input)
        dec_input += self.type_encoder(dec_type).view(dec_input.shape[0], 1, self.hidden_dim) + self.pos_encoder(dec_input) + align_g.view(dec_input.shape[0], 1, self.hidden_dim)
        return self.transformer(enc_input, dec_input), [align_c, align_g]
    def output(self, c_embed_batch):
        c_embed = c_embed_batch[0, 0, :].view(1, 1, self.out_embed_d)
        c_embed = self.out_dropout(c_embed)
        return self.output_act(self.out_linear(c_embed))
    def forward(self, x, neighbor_order_n_p_u, neighbor_order_n,all_t):
        x,dist = self.transformer_agg(x, neighbor_order_n_p_u, neighbor_order_n,all_t)
        return self.output(c_embed_batch=x), dist

def BCELoss(predictions, true_label):
    loss = nn.BCELoss(); predictions = predictions.view(1)
    tensor_label = torch.FloatTensor(np.array([true_label])).to(device)
    return loss(predictions, tensor_label)

def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path, map_location='cpu')
    filtered_state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if not k.startswith('gat_relation.user_tweet_embedding')}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    return model, optimizer, checkpoint['epoch'], checkpoint['val_acc']

def train_test(data_real, data_fake, test_size, PATH):
    y_real = range(len(data_real)); y_fake = range(len(data_fake))
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(data_real, y_real, test_size=test_size, random_state=42)
    X_train_fake, X_test_fake, y_train_fake, y_test_fake = train_test_split(data_fake, y_fake, test_size=test_size, random_state=42)
    return X_train_real, X_test_real, X_train_fake, X_test_fake

def early_truncate(neighbor_list, time_list, T_seconds):
    if not time_list or not neighbor_list: return [], []
    try: t0 = float(time_list[0].split('t')[-1])
    except: t0 = 0.0
    new_neigh = []; new_time = [time_list[0]]
    limit = min(len(neighbor_list), len(time_list) - 1)
    for i in range(limit):
        try: t_val = float(time_list[i+1].split('t')[-1])
        except: continue
        if t_val - t0 <= T_seconds:
            new_neigh.append(neighbor_list[i]); new_time.append(time_list[i+1])
    return new_neigh, new_time

# ================= 数据加载 =================
with open("/home/zhangrq/DPSG/data/rwr_results/pheme_n5_p5_u100/original_adj",'r',encoding='utf-8') as f: adj_list=json.load(f)
neighbor_dict = neighbor_loader('/home/zhangrq/DPSG/data/rwr_results/pheme_n5_p5_u100/n_neighbors.txt')
post_nodes = data_loader(pathway='/home/zhangrq/DPSG/data/processed_data/PHEME/pheme_n5_p5_u100/normalized_post_nodes/', node_type="post")
news_nodes = data_loader(pathway='/home/zhangrq/DPSG/data/processed_data/PHEME/pheme_n5_p5_u100/normalized_news_nodes/', node_type="news")
user_nodes = data_loader(pathway='/home/zhangrq/DPSG/data/processed_data/PHEME/pheme_n5_p5_u100/normalized_user_nodes/', node_type="user")

news_emb_dict = {}; post_emb_dict = {}; user_emb_dict = {}
for user in user_nodes: user_emb_dict[user.node_id] = user.emb
for post in post_nodes: post_emb_dict[post.node_id] = post.emb
for news in news_nodes: news_emb_dict[news.node_id] = news.emb

news_nodes_real = [n for n in news_nodes if n.label == 1]
news_nodes_fake = [n for n in news_nodes if n.label == 0]
ratio = len(news_nodes_real) / len(news_nodes_fake) if len(news_nodes_fake) > 0 else 1.0

# ================= 仅测试逻辑 =================
lr = 0.01; batch_size = 8; out_dim = 112; hidden_dim = 32; num_folds = 2; alp=2.15; belt=1.55
# checkpoints = [20/60, 40/60, 60/60, 80/60, 100/60, 120/60, 4, 8, 24] # 分钟/小时
checkpoints = [20/60, 40/60, 60/60, 80/60, 100/60, 120/60, 4, 8, 24]

base_PATH = '/home/zhangrq/DPSG/models/ablation/pheme/' # 使用之前训练的路径
f_path = base_PATH + 'result-earlytest.txt'

for repeat in range(5):
    PATH = base_PATH + 'lr' + str(lr) +'bs'+str(batch_size)+'od'+str(out_dim)+'hd'+str(hidden_dim)+'al'+str(alp)+'be'+str(belt)+'/repeat' + str(repeat)+ '/'
    if not os.path.exists(PATH): continue
    print(f"Repeat {repeat}")

    # 复现数据划分 (必须)
    X_train_real, X_test_real, X_train_fake, X_test_fake = train_test(news_nodes_real, news_nodes_fake, 0.1, PATH)
    test_set = X_test_real + X_test_fake
    
    # 加载模型 (Fold)
    best_models = []
    net = DPSG(input_dim=[768,512, 6], n_hidden_dim=hidden_dim, u_hidden_dim=hidden_dim, p_hidden_dim=hidden_dim, out_embed_d=out_dim, outemb_d=1, attn_heads=8, d_model=out_dim, self_attn_heads=8, enc_layers=1, dec_layers=1,max_nodes=10000)
    optimizer = optim.SGD(net.parameters(), lr=lr)

    loaded = False
    for fold in range(num_folds):
        cur_PATH = PATH + 'best_model' + '_' + str(fold) + '.tar'
        if os.path.exists(cur_PATH):
            # 这里必须每次重新初始化一个干净的 Net，否则 load_checkpoint 可能不干净
            fold_net = DPSG(input_dim=[768,512, 6], n_hidden_dim=hidden_dim, u_hidden_dim=hidden_dim, p_hidden_dim=hidden_dim, out_embed_d=out_dim, outemb_d=1, attn_heads=8, d_model=out_dim, self_attn_heads=8, enc_layers=1, dec_layers=1,max_nodes=10000).to(device)
            fold_net, _, _, _ = load_checkpoint(fold_net, optimizer, cur_PATH)
            fold_net.eval()
            best_models.append(fold_net)
            loaded = True
    
    if not loaded: continue
    
    # 测试循环
    with open(f_path, "a") as f: f.write(f"\nRepeat {repeat}\nTime\tAcc\tRealF1\tFakeF1\n")
    
    for T_hour in checkpoints:
        T_seconds = T_hour * 3600
        time_label = f"{int(round(T_hour*60))}min" if T_hour < 1.0 else f"{T_hour:.1f}h"
        
        t, real_true, fake_true, real_count, fake_count = 0, 0, 0, 0, 0
        
        for k in range(len(test_set)):
            full_neighbors = neighbor_dict[0][test_set[k].node_id]
            key = neighbor_dict[2][test_set[k].node_id]
            full_all_t = [key] + neighbor_dict[3][key]
            
            trunc_neighbors, trunc_all_t = early_truncate(full_neighbors, full_all_t, T_seconds)
            
            fold_preds = []
            for net in best_models:
                # 传入完整的 neighbor_dict[1] (neighbor_order_n) 作为辅助，核心是 trunc_neighbors
                out, _ = net(test_set[k], trunc_neighbors, neighbor_dict[1][test_set[k].node_id], trunc_all_t)
                fold_preds.append(out.item())
            
            output = sum(fold_preds)/len(fold_preds)
            
            pred = 1 if output / (1 - output + 1e-9) >= ratio else 0
            label = test_set[k].label
            
            if pred == label: t += 1
            if label == 1:
                real_count += 1
                if pred == 1: real_true += 1
            else:
                fake_count += 1
                if pred == 0: fake_true += 1
        
        acc = t / len(test_set)
        real_p = real_true / (real_true + fake_count - fake_true + 1e-9)
        real_r = real_true / (real_count + 1e-9)
        real_f1 = 2 * real_p * real_r / (real_p + real_r + 1e-9)
        
        fake_p = fake_true / (fake_true + real_count - real_true + 1e-9)
        fake_r = fake_true / (fake_count + 1e-9)
        fake_f1 = 2 * fake_p * fake_r / (fake_p + fake_r + 1e-9)

        print(f"[{time_label}] Acc: {acc:.4f} | Real F1: {real_f1:.4f} | Fake F1: {fake_f1:.4f}")
        with open(f_path, "a") as f: f.write(f"{time_label}\t{acc:.4f}\t{real_f1:.4f}\t{fake_f1:.4f}\n")

print("Finished.")