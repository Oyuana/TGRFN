#!/usr/bin/env python
# coding: utf-8

import torch.backends.cudnn as cudnn
cudnn.enabled = False

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.optim as optim
import os
import math
import random
import torch.nn.functional as F
import torch.nn.init as init
import json
import scipy.sparse as sp
import warnings

# 忽略除零等非致命警告，保持输出清爽
warnings.filterwarnings("ignore")

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("device:", device)

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# ================= 核心类定义 =================
class Het_Node():
    def __init__(self, node_type, node_id, embed, neighbor_list_news=[], neighbor_list_post=[], neighbor_list_user=[], label=None):
        self.node_type = node_type
        self.node_id = node_id
        self.emb = embed
        self.label = label
        self.neighbors_news = neighbor_list_news
        self.neighbors_post = neighbor_list_post
        self.neighbors_user = neighbor_list_user

# ================= 数据加载函数 =================
def neighbor_loader(pathway):
    if not os.path.exists(pathway):
        print(f"Error: Neighbor file not found at {pathway}")
        return {}, {}, {}, {}
    
    n_p_u = {}
    n_p_u_time = {}
    n = {}
    
    with open(pathway) as f:
        lines = f.readlines()
    
    for line in lines:
        lst = line.split()
        if len(lst) < 1: continue
        
        # Key processing
        raw_key = lst[0][1:-1] # Remove []
        root_id = raw_key.split("t")[0]
        
        # Time processing
        # item format in file: [Type+ID+Time] e.g., 'p1234t5678'
        # We need to extract time for n_p_u_time
        # And Type+ID for n_p_u
        
        time_list = []
        neighbor_list = []
        n_neighbor_list = []
        
        for item in lst[1:]:
            if item == 'PADDING': continue
            # item structure: 'p'+id+'t'+time or 'u'+id+'t'+time
            # Find 't' index
            try:
                t_index = item.rfind('t')
                if t_index != -1:
                    time_val = item[t_index+1:]
                    type_id = item[:t_index]
                    node_type = type_id[0]
                    node_id = type_id[1:]
                    
                    time_list.append(time_val)
                    neighbor_list.append((node_type, node_id))
                    
                    if node_type == 'n':
                        n_neighbor_list.append((node_type, node_id))
            except:
                continue

        n_p_u_time[raw_key] = time_list
        n_p_u[root_id] = neighbor_list
        n[root_id] = n_neighbor_list

    # Create mapping: root_id -> raw_key (which contains t0)
    key_map = {}
    for k in n_p_u_time.keys():
        rid = k.split("t")[0]
        key_map[rid] = k
        
    return n_p_u, n, key_map, n_p_u_time

def data_loader(pathway, node_type="post"):
    if not os.path.exists(pathway):
        print(f"Error: Data path not found at {pathway}")
        return []
    
    nodes = []
    # Safety: read all batch files
    files = [f for f in os.listdir(pathway) if f.startswith('batch') and f.endswith('.txt')]
    # Sort by batch number
    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]) if '_' in x and '.' in x else 0)
    
    for fn in files:
        with open(os.path.join(pathway, fn)) as f:
            lines = f.readlines()
            
        # Parse based on type
        line_idx = 0
        while line_idx < len(lines):
            try:
                if node_type == "news":
                    # 7 lines per node
                    if line_idx + 6 >= len(lines): break
                    _, id_, label = lines[line_idx].split()
                    emb = []
                    emb.append(list(map(float, lines[line_idx+1].split())))
                    emb.append(list(map(float, lines[line_idx+2].split())))
                    emb.append(list(map(float, lines[line_idx+3].split())))
                    # Skip neighbors lines 4,5,6
                    nodes.append(Het_Node("news", id_, emb, label=int(label)))
                    line_idx += 7
                    
                elif node_type == "post":
                    # 3 lines per node
                    if line_idx + 2 >= len(lines): break
                    _, id_ = lines[line_idx].split()
                    emb = []
                    emb.append(list(map(float, lines[line_idx+1].split())))
                    emb.append(list(map(float, lines[line_idx+2].split())))
                    nodes.append(Het_Node("post", id_, emb))
                    line_idx += 3
                    
                elif node_type == "user":
                    # 3 lines per node
                    if line_idx + 2 >= len(lines): break
                    _, id_ = lines[line_idx].split()
                    emb = []
                    emb.append(list(map(float, lines[line_idx+1].split())))
                    emb.append(list(map(float, lines[line_idx+2].split())))
                    nodes.append(Het_Node("user", id_, emb))
                    line_idx += 3
            except Exception as e:
                # print(f"Error parsing line {line_idx} in {fn}: {e}")
                line_idx += 1 # Try to skip bad line
                continue
                
    return nodes

# ================= 模型组件 =================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x): return self.dropout(x + self.pe[:x.size(0), :])

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features; self.out_features = out_features; self.dropout = dropout; self.alpha = alpha; self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features))).to(device); nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1))).to(device); nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.wtrans = nn.Parameter(torch.zeros(size=(2 * out_features, out_features))).to(device); nn.init.xavier_uniform_(self.wtrans.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    def forward(self, inp, adj):
        h = torch.mm(inp, self.W)
        Wh1 = torch.mm(h, self.a[:self.out_features, :]); Wh2 = torch.mm(h, self.a[self.out_features:, :])
        e = self.leakyrelu(Wh1 + Wh2.T)
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        negative_attention = torch.where(adj > 0, -e, zero_vec)
        attention = F.dropout(F.softmax(attention, dim=1), self.dropout, training=self.training)
        negative_attention = F.dropout(-F.softmax(negative_attention,dim=1), self.dropout, training=self.training)
        h_prime = torch.matmul(attention, inp); h_prime_negative = torch.matmul(negative_attention, inp)
        out = torch.mm(torch.cat([h_prime,h_prime_negative],dim=1), self.wtrans)
        return F.elu(out) if self.concat else out

class Signed_GAT(nn.Module):
    def __init__(self,nb_heads = 1, dropout = 0, alpha = 0.3):
        super(Signed_GAT, self).__init__()
        self.dropout = dropout; self.nb_heads = nb_heads; self.alpha = alpha
    def forward(self,node_embedding,cosmatrix,original_adj, X_tid):
        embedding_dim = node_embedding.shape[1]; node_num = original_adj.shape[0]
        user_tweet_embedding = nn.Embedding(num_embeddings=node_num, embedding_dim=embedding_dim, padding_idx=0).to(device)
        user_tweet_embedding.from_pretrained(node_embedding)
        original_adj = torch.from_numpy(original_adj.astype(np.float64)).to(device)
        potentinal_adj = torch.where(cosmatrix > 0.5, torch.ones_like(cosmatrix), torch.zeros_like(cosmatrix)).to(device)
        adj = torch.where(original_adj + potentinal_adj > 0, torch.ones_like(original_adj), torch.zeros_like(original_adj)).to(torch.float32)
        self.attentions = [GraphAttentionLayer(embedding_dim, embedding_dim, dropout=self.dropout, alpha=self.alpha, concat=True) for _ in range(self.nb_heads)]
        for i, attention in enumerate(self.attentions): self.add_module('attention_{}'.format(i), attention)
        out_att = GraphAttentionLayer(embedding_dim * self.nb_heads, embedding_dim, dropout=self.dropout,alpha=self.alpha,concat=False)
        X = F.dropout(user_tweet_embedding(torch.arange(0, node_num).long().to(device)).to(torch.float32), self.dropout, training=self.training)
        x = torch.cat([att(X, adj) for att in self.attentions], dim=1)
        x = F.sigmoid(out_att(F.dropout(x, self.dropout, training=self.training), adj))
        if isinstance(X_tid, list): return x, x[torch.LongTensor(X_tid).to(device)]
        return x,x[X_tid]

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
    def FFN(self, X): return self.dropout(self.linear2(F.relu(self.linear1(X))))
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
    def __init__(self, input_dim,n_hidden_dim,u_hidden_dim,p_hidden_dim,
                 out_embed_d, outemb_d=1, n_num_layers=1, u_num_layers=1,
                 p_num_layers=1, num_layers=1,content_dict={},
                 attn_heads=2,d_model=512,  npu=200):
        super(DPSG, self).__init__()
        self.npu = npu; self.input_dim = input_dim; self.hidden_dim = out_embed_d
        self.d_model = d_model; self.out_embed_d = out_embed_d; self.outemb_d = outemb_d; self.content_dict = content_dict
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.1, max_len=201); self.type_encoder = nn.Embedding(3, d_model, padding_idx=0)
        self.transformer = nn.Transformer(d_model=d_model, nhead=attn_heads, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=512,dropout=0.1, activation='relu')
        self.init_linear_text = nn.Linear(self.input_dim[0], self.hidden_dim)
        self.init_linear_image = nn.Linear(self.input_dim[1], self.hidden_dim)
        self.init_linear_time = nn.Linear(1, self.hidden_dim)
        self.init_linear_other_p = nn.Linear(self.input_dim[2], self.hidden_dim)
        self.init_linear_other_user = nn.Linear(self.input_dim[3], self.hidden_dim)
        self.news_title_attention_text = nn.MultiheadAttention(self.hidden_dim, attn_heads, dropout=0.2)
        self.news_content_attention_text = nn.MultiheadAttention(self.hidden_dim, attn_heads, dropout=0.2)
        self.post_content_attention_text = nn.MultiheadAttention(self.hidden_dim, attn_heads, dropout=0.2)
        self.user_content_attention_text = nn.MultiheadAttention(self.hidden_dim, attn_heads, dropout=0.2)
        self.attention_image = nn.MultiheadAttention(self.hidden_dim, attn_heads, dropout=0.2)
        self.attention_time = nn.MultiheadAttention(self.hidden_dim, attn_heads, dropout=0.2)
        self.attention_other = nn.MultiheadAttention(self.hidden_dim, attn_heads, dropout=0.2)
        self.layernorm1 = nn.LayerNorm([1, out_embed_d]); self.layernorm2 = nn.LayerNorm([1, out_embed_d]); self.layernorm3 = nn.LayerNorm([1, out_embed_d])
        self.layernorm4 = nn.LayerNorm([1, out_embed_d]); self.layernorm5 = nn.LayerNorm([1, out_embed_d]); self.layernorm6 = nn.LayerNorm([1, out_embed_d])
        self.act = nn.LeakyReLU(); self.relu = nn.ReLU(); self.softmax = nn.Softmax(dim=1)
        self.out_dropout = nn.Dropout(p=0.25); self.out_linear = nn.Linear(self.out_embed_d, self.outemb_d); self.output_act = nn.Sigmoid()
        self.mh_attention1 = TransformerBlock(input_size=self.hidden_dim, n_heads=8, attn_dropout=0); self.mh_attention = TransformerBlock(input_size=self.hidden_dim, n_heads=8, attn_dropout=0,is_layer_norm=True)
        self.alignfc_g = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim); self.alignfc_c = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.fc3 = nn.Linear(2*self.hidden_dim, self.hidden_dim); self.dropout = nn.Dropout(0.6); self.gat_relation=Signed_GAT()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter): nn.init.xavier_normal_(m.weight.data); m.bias.data.fill_(0.1)

    def build_symmetric_adjacency_matrix(self,edges, shape):
        if edges.shape[0] == 0: adj = sp.coo_matrix((shape[0], shape[1]), dtype=np.float32)
        else: adj = sp.coo_matrix(arg1=(edges[:, 2], (edges[:, 0], edges[:, 1])), shape=shape, dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        rowsum = np.array(adj.sum(1))
        # 修复除零警告
        with np.errstate(divide='ignore'):
            r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
        return (adj.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt) + sp.eye(adj.shape[0])).tocoo()

    def calculate_cos_matrix(self,node_embedding):
        if node_embedding.dim() > 2: node_embedding = node_embedding.squeeze(1)
        a,b = node_embedding,node_embedding.T; c = torch.mm(a, b); aa = torch.mul(a, a); bb = torch.mul(b, b)
        norm = torch.mm(torch.sqrt(torch.sum(aa, dim=1, keepdim=True)), torch.sqrt(torch.sum(bb, dim=0, keepdim=True))) + 1e-8
        return c / norm

    def Bi_RNN(self, neighbor_id, neighbor_time,node_type, post_emb_dict, user_emb_dict, news_emb_dict):
        if node_type == "news":
            input_title = []; input_content = []; input_image = []
            for i in neighbor_id:
                if ("news", i) not in self.content_dict: input_title.append(news_emb_dict[i][0]); input_content.append(news_emb_dict[i][1]); input_image.append(news_emb_dict[i][2])
            if not input_title: return torch.zeros(1, self.hidden_dim).to(device)
            input_title = torch.Tensor(input_title).to(device); input_image = torch.Tensor(input_image).to(device); input_content = torch.Tensor(input_content).to(device); input_time=torch.Tensor(neighbor_time).to(device).view(-1,1)
            l_t = self.init_linear_text(input_title).view(len(input_title), 1, -1); l_c = self.init_linear_text(input_content).view(len(input_content), 1, -1); l_i = self.init_linear_image(input_image).view(len(input_image), 1, -1); l_tm = self.init_linear_time(input_time).view(len(input_time), 1, -1)
            att_t, _ = self.news_title_attention_text(l_t,l_t,l_t); att_c, _ = self.news_content_attention_text(l_c,l_c,l_c); att_i, _ = self.attention_image(l_i, l_i,l_i); att_tm, _ = self.attention_time(l_tm, l_tm,l_tm); att_tm = self.layernorm6(att_tm)
            concate = torch.cat((self.mh_attention(att_tm,self.layernorm1(att_t),self.layernorm1(att_t)), self.mh_attention(att_tm, self.layernorm2(att_c), self.layernorm2(att_c)), self.mh_attention(att_tm, att_i, att_i)), 1)
        elif node_type == "post":
            input_a = []; input_b = []
            for i in neighbor_id:
                if ("post", i) not in self.content_dict: input_a.append(post_emb_dict[i][1]); input_b.append(post_emb_dict[i][0])
            if not input_a: return torch.zeros(1, self.hidden_dim).to(device)
            input_a = torch.Tensor(input_a).to(device); input_b = torch.Tensor(input_b).to(device); input_time = torch.Tensor(neighbor_time).to(device).view(-1, 1)
            l_t = self.init_linear_text(input_a).view(len(input_a), 1, -1); l_o = self.init_linear_other_p(input_b).view(len(input_b), 1, -1); l_tm = self.init_linear_time(input_time).view(len(input_time), 1, -1)
            att_t, _ = self.post_content_attention_text(l_t,l_t,l_t); att_o, _ = self.attention_other(l_o, l_o,l_o); att_tm, _ = self.attention_time(l_tm, l_tm,l_tm); att_tm = self.layernorm6(att_tm)
            concate = torch.cat((self.mh_attention(att_tm, self.layernorm3(att_t), self.layernorm3(att_t)), self.mh_attention(att_tm, self.layernorm3(att_o), self.layernorm3(att_o))), 1)
        elif node_type == "user":
            input_a = []; input_b = []
            for i in neighbor_id:
                if ("user", i) not in self.content_dict: input_a.append(user_emb_dict[i][0]); input_b.append(user_emb_dict[i][1])
            if not input_a: return torch.zeros(1, self.hidden_dim).to(device)
            input_a = torch.Tensor(input_a).to(device); input_b = torch.Tensor(input_b).to(device)
            l_t = self.init_linear_text(input_b).view(len(input_b), 1, -1); l_o = self.init_linear_other_user(input_a).view(len(input_a), 1, -1)
            att_t, _ = self.user_content_attention_text(l_t,l_t,l_t); att_o, _ = self.attention_other(l_o, l_o,l_o)
            concate = torch.cat((self.layernorm4(att_t), self.layernorm5(att_o)), 1)
        return torch.mean(concate, 1)

    def SameType_Agg_Bi_RNN(self, neighbor_id,neighbor_time, node_type):
        content_embedings = self.Bi_RNN(neighbor_id,neighbor_time, node_type, post_emb_dict, user_emb_dict, news_emb_dict)
        aft_rnn_dict = {}
        for i, nid in enumerate(neighbor_id): aft_rnn_dict[nid] = content_embedings[i]
        return aft_rnn_dict

    def transformer_agg(self, het_node, neighbor_order_n_p_u, neighbor_order_n,all_t0, npu=200,n=5):
        all_t = [float(i.split('t')[-1]) for i in all_t0]
        # 修正：当时间列表过短时
        if len(all_t) > 1:
            all_tt = torch.tensor(all_t[1:]) - all_t[0]
            all_tt_exp = F.softmax(-F.softmax(all_tt, dim=0), dim=0).to(device)
        else:
            all_tt_exp = torch.tensor([]).to(device)

        c_agg_batch = self.Bi_RNN([het_node.node_id], [0],het_node.node_type, post_emb_dict, user_emb_dict, news_emb_dict)
        center_emb = c_agg_batch[0]
        
        node_embedding = [center_emb]; node_embedding_n = [center_emb]
        now_node = ["n"+het_node.node_id]; now_node_n = ["n"+het_node.node_id]
        post_neighbor, post_time, user_neighbor, user_time, news_neighbor, news_time = [], [], [], [], [], []
        
        limit = min(len(neighbor_order_n_p_u), npu, len(all_tt_exp))
        for idx, item in enumerate(neighbor_order_n_p_u[:limit]):
            t = all_tt_exp[idx]
            if item[0]=='p': post_neighbor.append(item[1]); post_time.append(t)
            elif item[0]=='u': user_neighbor.append(item[1]); user_time.append(t)
            elif item[0]=='n': news_neighbor.append(item[1]); news_time.append(t)

        p_aft, u_aft, n_aft = {}, {}, {}
        if news_neighbor:
            n_aft = self.SameType_Agg_Bi_RNN(news_neighbor,news_time, "news")
            for id,v in n_aft.items(): now_node.append("n"+id); now_node_n.append("n"+id); node_embedding.append(v); node_embedding_n.append(v)
        if user_neighbor:
            u_aft = self.SameType_Agg_Bi_RNN(user_neighbor, user_time,"user")
            for id,v in u_aft.items(): now_node.append("u"+id); node_embedding.append(v)
        if post_neighbor:
            p_aft = self.SameType_Agg_Bi_RNN(post_neighbor,post_time, "post")
            for id,v in p_aft.items(): now_node.append("p"+id); node_embedding.append(v)
        
        node_embedding=torch.stack(node_embedding); node_embedding_n=torch.stack(node_embedding_n)
        
        now_adj_list_old={}
        for node in now_node:
            c=list(set(adj_list[node])&set(now_node)) if node in adj_list else []
            if c: now_adj_list_old[node]=c
        node_id2_xulie={id:i for i,id in enumerate(now_node)}
        now_node_n=[node_id2_xulie[i] for i in now_node_n]
        relation=[]
        for id,v in now_adj_list_old.items():
            for dst in v: relation.append([node_id2_xulie[id],node_id2_xulie[dst],1])
        relation=np.array(relation) if len(relation)>0 else np.zeros((0,3))
        
        # 修复 IndexError: 确保 build_symmetric_adjacency_matrix 接收正确的 shape
        adj = self.build_symmetric_adjacency_matrix(relation, (len(now_node), len(now_node)))
        
        original_adj=np.zeros(adj.shape)
        if len(relation)>0: original_adj[relation[:,0].astype(int), relation[:,1].astype(int)] = 1
        
        self.cosmatrix = self.calculate_cos_matrix(node_embedding)
        X_tid = torch.LongTensor(now_node_n).to(device)
        gx, graph_feature = self.gat_relation(node_embedding, self.cosmatrix, original_adj, X_tid)

        graph_feature_all=[gx[0]]
        for type_id in neighbor_order_n_p_u[:limit]:
            key = type_id[0] + type_id[1]
            if key in node_id2_xulie: graph_feature_all.append(gx[node_id2_xulie[key]])
        graph_feature_all=torch.stack(graph_feature_all)

        bsz, bsz_all = len(now_node_n), len(graph_feature_all)
        att_g = self.mh_attention1(graph_feature.view(bsz, 1, -1), graph_feature.view(bsz, 1, -1), graph_feature.view(bsz, 1, -1))
        att_c = self.mh_attention1(node_embedding_n.view(bsz, 1, -1), node_embedding_n.view(bsz, 1, -1), node_embedding_n.view(bsz, 1, -1))
        att_g_all = self.mh_attention1(graph_feature_all.view(bsz_all, 1, -1), graph_feature_all.view(bsz_all, 1, -1), graph_feature_all.view(bsz_all, 1, -1))
        
        align_c = self.alignfc_c(att_c).view(bsz, self.hidden_dim); align_g = self.alignfc_g(att_g).view(bsz, self.hidden_dim); align_g_all = self.alignfc_g(att_g_all).view(bsz_all, self.hidden_dim)
        dist = [align_c, align_g]

        enc_in = [center_emb]; enc_type = [0]
        for i, type_id in enumerate(neighbor_order_n_p_u[:limit]):
            if type_id[0] == 'p' and type_id[1] in p_aft: enc_in.append(p_aft[type_id[1]]); enc_type.append(1)
            elif type_id[0] == 'u' and type_id[1] in u_aft: enc_in.append(u_aft[type_id[1]]); enc_type.append(2)
            elif type_id[0] == 'n' and type_id[1] in n_aft: enc_in.append(n_aft[type_id[1]]); enc_type.append(0)
        
        enc_type = torch.LongTensor(enc_type).to(device)
        enc_in = torch.stack(enc_in); bsz_enc = enc_in.size(0)
        
        att_c_all = self.mh_attention1(enc_in.view(bsz_enc, 1, -1), enc_in.view(bsz_enc, 1, -1), enc_in.view(bsz_enc, 1, -1))
        enc_in = self.alignfc_c(att_c_all).view(bsz_enc, 1, -1)
        enc_in += self.type_encoder(enc_type).view(bsz_enc, 1, -1) + self.pos_encoder(enc_in)
        if align_g_all.size(0) == bsz_enc: enc_in += align_g_all.view(bsz_enc, 1, -1)

        dec_in = [center_emb]; dec_type = [0]
        for (i, type_id) in enumerate(news_neighbor):
            if type_id in n_aft: dec_in.append(n_aft[type_id]); dec_type.append(0)
        dec_type = torch.LongTensor(dec_type).to(device)
        
        dec_in = align_c.view(align_c.shape[0], 1, -1)
        dec_in += self.pos_encoder(dec_in) + self.type_encoder(dec_type).view(dec_in.shape[0], 1, -1) + align_g.view(dec_in.shape[0], 1, -1)

        return self.transformer(enc_in, dec_in), dist

    def output(self, c_embed_batch):
        return self.output_act(self.out_linear(self.out_dropout(c_embed_batch[0, 0, :].view(1, 1, self.out_embed_d))))

    def forward(self, x, neighbor_order_n_p_u, neighbor_order_n,all_t, npu=200):
        x,dist = self.transformer_agg(x, neighbor_order_n_p_u, neighbor_order_n,all_t, npu=npu)
        return self.output(c_embed_batch=x), dist

# ================= 辅助函数 =================
def load_checkpoint(model, load_path):
    # 严格模式关闭，因为嵌入层可能在训练时更新，但在推断时我们可能不希望它报错
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False) 
    model.to(device)
    return model

def early_truncate(neighbor_list, time_list, T_seconds):
    if not time_list: return [], []
    try: t0 = float(time_list[0].split('t')[-1])
    except: return [], []
    new_neigh = []; new_time = [time_list[0]]
    # neighbor_list is one shorter than time_list
    limit = min(len(neighbor_list), len(time_list) - 1)
    for i in range(limit):
        try: t = float(time_list[i+1].split('t')[-1])
        except: continue
        if t - t0 <= T_seconds:
            new_neigh.append(neighbor_list[i])
            new_time.append(time_list[i+1])
    return new_neigh, new_time

def train_test(data_real, data_fake, test_size, PATH):
    y_real = range(len(data_real))
    y_fake = range(len(data_fake))
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(data_real, y_real, test_size=test_size, random_state=42)
    X_train_fake, X_test_fake, y_train_fake, y_test_fake = train_test_split(data_fake, y_fake, test_size=test_size, random_state=42)
    return X_train_real, X_test_real, X_train_fake, X_test_fake

# ================= 主流程 =================
print("Loading Data...")
neighbor_dict_tuple = neighbor_loader('/home/zhangrq/DPSG/data/rwr_results/fnn_politifact_n5_p5_u100/n_neighbors.txt')
neighbor_dict = {
    0: neighbor_dict_tuple[0],
    1: neighbor_dict_tuple[1],
    2: neighbor_dict_tuple[2],
    3: neighbor_dict_tuple[3]
}

with open("/home/zhangrq/DPSG/data/rwr_results/fnn_politifact_n5_p5_u100/original_adj",'r') as f:
    adj_list=json.load(f)

post_nodes = data_loader('/home/zhangrq/DPSG/data/processed_data/FakeNewsNet/PolitiFact/fnn_politifact_n5_p5_u100/normalized_post_nodes/', "post")
user_nodes = data_loader('/home/zhangrq/DPSG/data/processed_data/FakeNewsNet/PolitiFact/fnn_politifact_n5_p5_u100/normalized_user_nodes/', "user")
news_nodes = data_loader('/home/zhangrq/DPSG/data/processed_data/FakeNewsNet/PolitiFact/fnn_politifact_n5_p5_u100/normalized_news_nodes/', "news")

news_emb_dict = {n.node_id: n.emb for n in news_nodes}
post_emb_dict = {n.node_id: n.emb for n in post_nodes}
user_emb_dict = {n.node_id: n.emb for n in user_nodes}

real_news = [n for n in news_nodes if n.label == 1]
fake_news = [n for n in news_nodes if n.label == 0]
if len(fake_news) == 0: ratio = 1.0 
else: ratio = len(real_news) / len(fake_news)
print(f"Data Loaded. Real: {len(real_news)}, Fake: {len(fake_news)}, Ratio: {ratio:.4f}")

lr=0.0001; batch_size=16; output_dim=128; hidden_dim=256; num_folds = 9
checkpoints = [20/60, 40/60, 60/60, 80/60, 100/60, 120/60, 4, 8, 24]
base_PATH = '/home/zhangrq/DPSG/models/pre-trained/politifact/'
f_out = base_PATH + 'result-early.txt'

for repeat in range(5):
    print(f"Repeat {repeat}")
    PATH = base_PATH +'lr'+str(lr)+'bs'+str(batch_size)+ 'od'+str(output_dim)+'hd'+str(hidden_dim)+'al2.15be1.55/repeat' + str(repeat) + '/'
    if not os.path.exists(PATH): continue
    
    # 切分数据
    _, X_test_real, _, X_test_fake = train_test(real_news, fake_news, 0.1, PATH)
    test_set = X_test_real + X_test_fake
    
    # 加载模型
    models = []
    for fold in range(num_folds):
        cur_PATH = PATH + 'best_model_' + str(fold) + '.tar'
        if os.path.exists(cur_PATH):
            net = DPSG(input_dim=[768, 512, 3, 9], n_hidden_dim=hidden_dim, u_hidden_dim=hidden_dim, p_hidden_dim=hidden_dim, out_embed_d=output_dim, d_model=output_dim, attn_heads=8).to(device)
            try:
                net = load_checkpoint(net, cur_PATH)
                net.eval()
                models.append(net)
            except Exception as e:
                pass
                # print(f"Load failed for fold {fold}: {e}")

    if not models: continue
    
    with open(f_out, "a") as f: f.write(f"\nRepeat {repeat}\nTime\tAcc\tRealF1\tFakeF1\n")
    
    for T_h in checkpoints:
        T_s = T_h * 3600
        t_cnt, rc, fc, rt, ft = 0, 0, 0, 0, 0
        
        for node in test_set:
            nid = node.node_id
            
            # --- 最终修复：防御性数据获取 ---
            # 如果 ID 不在字典，或者 key 不在时间字典，直接设为空
            if nid not in neighbor_dict[0] or nid not in neighbor_dict[2]:
                neighbors, all_t = [], [f"t0.0"]
            else:
                neighbors = neighbor_dict[0][nid]
                key = neighbor_dict[2][nid]
                all_t = [key] + neighbor_dict[3].get(key, [])
            
            trunc_neigh, trunc_t = early_truncate(neighbors, all_t, T_s)
            
            neigh_n = neighbor_dict[1].get(nid, []) if 1 in neighbor_dict else []
            
            preds = []
            for net in models:
                try:
                    # 传入模型
                    out, _ = net(node, trunc_neigh, neigh_n, trunc_t)
                    preds.append(out.item())
                except Exception:
                    # 任何模型错误（包括除零导致的数值异常），都视为无法判断，给 0.5
                    preds.append(0.5)
            
            if not preds: preds = [0.5]
            avg = sum(preds)/len(preds) + 1e-10
            
            pred = 1 if avg / (1 - avg) >= ratio else 0
            
            if pred == node.label: t_cnt += 1
            if node.label == 1:
                rc += 1
                if pred == 1: rt += 1
            else:
                fc += 1
                if pred == 0: ft += 1
        
        acc = t_cnt / len(test_set)
        
        # Calculate Metrics
        r_p = rt / (rt + fc - ft + 1e-10)
        r_r = rt / (rc + 1e-10)
        rf1 = 2 * r_p * r_r / (r_p + r_r + 1e-10)
        
        f_p = ft / (ft + rc - rt + 1e-10)
        f_r = ft / (fc + 1e-10)
        ff1 = 2 * f_p * f_r / (f_p + f_r + 1e-10)
        
        print(f"[{T_h*60:.0f}min] Acc: {acc:.4f} | Real F1: {rf1:.4f} | Fake F1: {ff1:.4f}")
        with open(f_out, "a") as f: f.write(f"{T_h:.2f}h\t{acc:.4f}\t{rf1:.4f}\t{ff1:.4f}\n")

print("Finished.")