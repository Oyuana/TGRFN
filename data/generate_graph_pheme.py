from itertools import product
from tqdm import tqdm
from collections import defaultdict
from os import listdir
from os.path import join
from json import load
import json
from datetime import datetime,timezone,timedelta

#处理 PHEME 数据集，构建社交媒体上的 图结构，即社交网络中的 节点（用户、新闻、帖子） 及其 边（关系）

in_dir = 'PHEME'
out_dir = 'PHEME/graph_def'
node_files = {
}
#定义了 6 种边类型
edge_files = {
    ('n', 'n'): 'PhemeNewsNews.txt',
    ('n', 'p'): 'PhemeNewsPost.txt',
    ('n', 'u'): 'PhemeNewsUser.txt',
    ('p', 'p'): 'PhemePostPost.txt',
    ('p', 'u'): 'PhemePostUser.txt',
    ('u', 'u'): 'PhemeUserUser.txt',
}
#将 Twitter 时间格式转换为 UTC 时间戳,用于唯一标识 推文 ID，避免不同用户的推文时间重叠
def changetime(time_str):
    dt_object = datetime.strptime(time_str, "%a %b %d %H:%M:%S %z %Y")
    dt_object_utc = dt_object.astimezone(timezone.utc)
    timestamp = int(dt_object_utc.timestamp())
    return str(timestamp)

#处理数据集 并 生成图的边文件
def process():
    #写入边文件
    # d：边的字典（格式：{(源节点, 目标节点): 边权重}）
    # fn：输出文件名，如 PhemeNewsPost.txt
    def write_edge(d, fn):
        with open(join(out_dir, fn), 'w') as f:
            for (s, t), v in d.items():
                f.write(f'{s}\t{t}\t{v}\n')
    
    #递归解析 structure_add_time.json 构造 推文之间的边
    def write_edges_from_structure(root, tree, level):
        if len(tree) == 0:
            return
        for child, subtree in tree.items():
            #层级 0（根）：新闻 (n) → 推文 (p) 关系
            if level == 0:
                edges[('n', 'p')][(root, child)] += 1
            #层级 1+（评论链）：推文 **(p) → 推文 (p) 关系（双向）
            else:
                edges[('p', 'p')][(root, child)] += 1
                edges[('p', 'p')][(child, root)] += 1
            write_edges_from_structure(child, subtree, level + 1)
    
    #解析用户关系
    def write_user_edges(folder, tweet_fname, tweet_type):
        tweet = load(open(join(news_root, folder, tweet_fname), 'r'))
        tweet_id = tweet["id_str"] + "t" + changetime(tweet['created_at'])
        user_id = tweet["user"]["id_str"] + "t" + "0"
        # 推文→用户 n-u or p-u
        edges[(tweet_type, 'u')][tweet_id, user_id] += 1
        # 用户→用户 u-u
        another_user_id = tweet["in_reply_to_user_id_str"]
        if another_user_id != None:
            another_user_id = str(tweet["in_reply_to_user_id_str"]) + "t" + "0"
            edges[('u', 'u')][user_id, another_user_id] += 1
            edges[('u', 'u')][another_user_id, user_id] += 1

    edges = {k : defaultdict(int) for k in edge_files.keys()}
    #遍历数据集
    for event_raw in listdir(in_dir):
        #筛选 -all-rnr-threads 结尾的事件,提取事件名称
        if not event_raw.endswith('-all-rnr-threads'): continue
        # {event}-all-rnr-threads
        event = event_raw[:-16]
        same_event_news = set()
        # 解析每个新闻贴，构建 新闻推文的图结构
        for rumority in ['non-rumours', 'rumours']:
            for news_id in tqdm(listdir(join(in_dir, event_raw, rumority)), desc=f'{event_raw}-{rumority}'):
                if news_id == '.DS_Store': continue
                news_root = join(in_dir, event_raw, rumority, news_id)
                source_tweet = load(open(join(news_root, 'source-tweet', news_id + '.json')))
                news_id_time = news_id + "t" + changetime(source_tweet['created_at'])
                same_event_news.add(news_id_time)
                # n-p, p-p
                structure = load(open(join(news_root, 'structure_add_time.json'), 'r'))
                write_edges_from_structure(news_id_time, structure[news_id_time], 0)
                # n-u
                write_user_edges('source-tweet', f'{news_id}.json', 'n')
                # p-u, u-u
                for tweet_file_name in listdir(join(news_root, 'reactions')):
                    if tweet_file_name == '.DS_Store':
                        continue
                    write_user_edges('reactions', tweet_file_name, 'p')
        for news_id_1 in same_event_news:
            for news_id_2 in same_event_news:
                edges[('n', 'n')][news_id_1, news_id_2] += 1
    # 遍历 edges，将不同类型的边写入 对应的 txt 文件
    for k, v in edge_files.items():
        write_edge(edges[k], v)
# 
def getstruct():
    #递归地在 struct（树形结构）中查找 re（被回复的推文 ID），并在找到 re 后，将 v（当前推文 ID）作为其子节点插入
    def getkv(re,v,struct):
        for key in struct:
            if len(struct[key])!=0:
                if re in struct[key]:
                    if len(struct[key][re]) == 0:
                        struct[key][re] = {v: {}}
                    else:
                        struct[key][re][v] = []
                    break
                else:
                    getkv(re, v, struct[key])
    #遍历 in_dir 目录中的所有事件文件夹
    for event_raw in listdir(in_dir):
        #只处理文件名以 -all-rnr-threads 结尾的文件夹
        if not event_raw.endswith('-all-rnr-threads'): continue
        for rumority in ['non-rumours', 'rumours']:
            #遍历 rumours 或 non-rumours 目录中的所有 news_id
            for news_id in tqdm(listdir(join(in_dir, event_raw, rumority)), desc=f'{event_raw}-{rumority}'):
                # 读取源推文，创建根节点
                #构造 news_root 路径，指向该新闻推文的文件夹
                news_root = join(in_dir, event_raw, rumority, news_id)
                #读取 source-tweet/{news_id}.json 文件，解析出源推文数据
                source_tweet = load(open(join(news_root, 'source-tweet', news_id + '.json'), 'r'))
                # news_id_time 由 news_id + "t" + 推文时间戳 组成，确保唯一性
                news_id_time = news_id + "t" + changetime(source_tweet['created_at'])
                # struct 初始化为 {news_id_time: {}}，表示树形结构的根节点
                struct = {news_id_time: {}}
                #遍历所有回复推文，构建树形结构
                #进入 reactions 目录，遍历所有回复推文文件
                for tweet_file_name in listdir(join(news_root, 'reactions')):
                    tweet = load(open(join(news_root, 'reactions', tweet_file_name), 'r'))
                    #v：当前推文的 ID + "t" + 时间戳
                    v = str(tweet["id"]) + "t" + changetime(tweet['created_at'])
                    #re：该推文回复的父推文 ID + "t" + 源推文时间戳
                    re = str(tweet["in_reply_to_status_id"]) + "t" + changetime(source_tweet['created_at'])
                    #如果 re 已在 struct 结构中，则直接添加 v 作为子节点
                    if re in struct:
                        struct[re][v]=[]
                    #如果 re 不在 struct 顶层，需要递归查找 re
                    else:
                        for key in struct:
                            #先检查 struct[key] 是否非空
                            if len(struct[key]) != 0:
                                #如果 re 存在于 struct[key]
                                if re in struct[key]:
                                    #如果 re 还没有子节点，则创建 {v: {}}
                                    if len(struct[key][re])==0:
                                        struct[key][re]={v:{}}
                                    #否则，将 v 作为 re 的子节点，并赋值为空列表 []
                                    else:
                                        struct[key][re][v] = []
                                #如果 re 仍未找到，递归调用 getkv(re, v, struct[key]) 继续查找
                                else:
                                    getkv(re, v, struct[key])
                #遍历完所有 reactions 之后，将 struct 结构写入 structure_add_time.json 文件，该文件记录了该新闻推文及其所有回复的树状结构
                with open(join(news_root, "structure_add_time.json"), 'w') as f:
                    json.dump(struct,f)

if __name__ == '__main__':
    # getstruct()
    process()