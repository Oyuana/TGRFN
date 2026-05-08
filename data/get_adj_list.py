
import os
import json
from tqdm import tqdm

#将生成的图结构转为邻接列表形式，读取 FakeNewsNet 数据集中的边文件，构建节点之间的邻接关系，并将结果保存为 JSON 格式的邻接表（adj_list）

#将两个节点（m 和 n）之间的邻接关系添加到 adj_list 字典中
def add_adjacent(m, n):
    if m not in adj_list.keys():
        adj_list[m] = []
    adj_list[m].append(n)


in_dir = 'FakeNewsNet-Dataset/FakeNewsNet_Dataset/graph_def'
# edge_dir = os.path.join(in_dir, 'politifact')
edge_dir = os.path.join(in_dir, 'gossipcop')

edge_files = {
    ('n', 'n'): 'news-news edges.txt',
    ('n', 'p'): 'news-post edges.txt',
    ('p', 'u'): 'post-user edges.txt',
    ('u', 'u'): 'user-user edges.txt',
}
output_dir = f"rwr_results/fnn_gossipcop_n5_p5_u100"

#初始化邻接列表
adj_list = dict()
#遍历边文件并读取数据
for (main_type, neig_type), edge_f in edge_files.items():
    with open(os.path.join(edge_dir, edge_f), "r") as f:
        for l in tqdm(f.readlines(), desc='read ' + main_type+' '+neig_type):
            l = l.strip().split()
            add_adjacent(main_type + l[0], neig_type + l[1])
            add_adjacent(neig_type + l[1], main_type + l[0])


with open(output_dir+"/original_adj",'w') as f:
    print(len(adj_list))
    json.dump(adj_list,f)