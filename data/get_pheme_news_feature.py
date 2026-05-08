#!/usr/bin/env python
# coding: utf-8


import os
from tqdm import tqdm
import json

#从 PHEME 数据集中提取每条源推文（新闻节点）的 点赞数（favorite_count） 作为结构性节点特征
#前面是fakenewsnet的提取post/user节点的retweet、favorite、quote  多个 profile 字段 作为结构性节点特征

data_path = 'PHEME/'
output_dir = 'processed_data/PHEME/'

event_list = ['charliehebdo-all-rnr-threads/','ferguson-all-rnr-threads/',
          'germanwings-crash-all-rnr-threads/', 'ottawashooting-all-rnr-threads/', 'sydneysiege-all-rnr-threads/']

news_features = list()

for event in tqdm(event_list):
    for news_type in ['rumours/', 'non-rumours/']:

        event_path = data_path + event + news_type
        post_list = os.listdir(event_path)
        try:
            post_list.remove('.DS_Store')
        except:
            pass
        for post in tqdm(post_list):
            news_path = event_path + post + '/source-tweet/'
            news_list = os.listdir(news_path)
            try:
                news_list.remove('.DS_Store')
            except:
                pass
            news = news_list[0].rstrip('.json')
            with open(news_path + news_list[0]) as json_file:
                news_dict = json.load(json_file)
            favorite_count = str(news_dict['favorite_count'])
            news_features.append([news, favorite_count])


if not os.path.exists(output_dir):
    os.mkdir(output_dir)
            
with open(output_dir + 'news_features_pheme.txt', 'w') as f:
    for [news, favorite_count] in news_features:
        f.write('%s: %s\n' %(news, favorite_count))
