"""
pheme{
    "n": 6425,5802
    "p": 98929,97410
    "u": 51043,49778}
politifact{
    "n": 870,884,733
    "p": 440467,1317371,1167624,
    "u": 468210,686925,627191}

gossipcop{
    "n": 21068,21202,20955,
    "p": 1192766,1920964,1890915,
    "u": 429628,630611,618440}

"""
import os
import json

#统计每个数据集中不同类型的节点（如新闻、推文、用户）及其关系（如新闻与新闻、新闻与推文等）的数量

#统计节点和边的数量
def stats(in_dir, node_types, edge_files):
    nodes = {t : [] for t in node_types}
    #遍历 edge_files 字典，其中每个键是一个元组 (t0, t1)，表示从节点类型 t0 到节点类型 t1 的边，fname 是文件名
    for (t0, t1), fname in edge_files.items():
        with open(os.path.join(in_dir, fname), 'r') as f:
            for line in f.readlines():
                # 去掉每行的空白字符并按空格分割为列表 info，假设每行有两个节点 ID
                info = line.strip().split()
                #将当前行的第一个节点 ID（info[0]）添加到 nodes[t0] 列表中
                nodes[t0].append(info[0])
                #将当前行的第二个节点 ID（info[1]）添加到 nodes[t1] 列表中
                nodes[t1].append(info[1])
    #统计每种节点类型的唯一数量
    counts = {k : len(set(v)) for k, v in nodes.items()}
    return counts

if __name__ == "__main__":
    for dataset in ['politifact', 'gossipcop','pheme']:
        if dataset in ['politifact', 'gossipcop']:
            #为文件命名时的前缀，格式为 fnn_数据集名_
            prefix = f'fnn_{dataset}_'
            in_dir = f'FakeNewsNet-Dataset/FakeNewsNet_Dataset/graph_def/{dataset}'
            edge_dir = os.path.join(in_dir, dataset)
            node_types = ['n', 'p', 'u']
            edge_files = {
                ('n', 'n'): 'news-news edges.txt',
                ('n', 'p'): 'news-post edges.txt',
                ('p', 'u'): 'post-user edges.txt',
                ('u', 'u'): 'user-user edges.txt',
            }
            edges_to_enforce = {('p', 'u'),}
        elif dataset == 'pheme':
            prefix = 'pheme_'
            in_dir = 'PHEME'
            edge_dir = in_dir
            node_types = ['n', 'p', 'u']
            edge_files = {
                ('n', 'p'): 'PhemeNewsPost.txt',
                ('n', 'u'): 'PhemeNewsUser.txt',
                ('p', 'p'): 'PhemePostPost.txt',
                ('p', 'u'): 'PhemePostUser.txt',
                ('u', 'u'): 'PhemeUserUser.txt',
            }
            edges_to_enforce = {('n', 'u'), ('p', 'u'),}
        #统计并输出每个数据集的节点数量
        counts = stats(in_dir, node_types, edge_files)
        print(dataset)
        print(json.dumps(counts, indent=4))