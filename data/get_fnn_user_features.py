#!/usr/bin/env python
# coding: utf-8


import json
import os
# import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing
import numpy as np

#为 FakeNewsNet 数据集中的 用户（user）节点 和 推文（post）节点 提取结构性数值特征，并输出为结构化文本文件，是多模态表示中的用户与帖子行为统计特征提取部分。

def get_user_post_features(dataset):
    uid = list()
    user_features = list()
    pid = list()
    post_features = list()
    
    print('reading %s...' % dataset)
    pathway = 'FakeNewsNet-Dataset/FakeNewsNet_Dataset/%s/' % dataset
    output_dir = 'processed_data/FakeNewsNet/%s/' % dataset
    for post_type in ['fake/', 'real/']:
        
        print('reading user info from %s file...' %post_type)
        news_list = os.listdir(pathway + post_type)
        
        for file in tqdm(news_list, desc='reading each news'):
            with open(pathway + post_type + file + '/retweets.json' ) as f:  # read json files
                retweet_dict = json.load(f)
            for tweet_id,retweet_list in retweet_dict.items():
                if len(retweet_list)!=0:
                    for re in retweet_list:
                        tweet_post_data=re["retweeted_status"]
                        if tweet_post_data['id_str'] not in pid:
                            pid.append(tweet_post_data['id_str'])
                            num_retweet = str(tweet_post_data['retweet_count'])

                            num_favorite = str(tweet_post_data['favorite_count'])
                            is_quote_status = str(tweet_post_data['is_quote_status'] - 0)

                            post_features.append([num_retweet, num_favorite, is_quote_status])

                        retweet_post_data=re
                        if retweet_post_data['id_str'] not in pid:
                            pid.append(retweet_post_data['id_str'])
                            num_retweet = str(retweet_post_data['retweet_count'])

                            # skip num word description
                            # skip num word name
                            num_favorite = str(retweet_post_data['favorite_count'])
                            is_quote_status = str(retweet_post_data['is_quote_status'] - 0)

                            post_features.append([num_retweet, num_favorite, is_quote_status])

                        tweet_user_data=re["retweeted_status"]["user"]
                        if tweet_user_data["id_str"] not in uid:
                            uid.append(tweet_user_data["id_str"])
                            num_friends = str(tweet_user_data['friends_count']) #用户关注数
                            # skip num word description
                            # skip num word name
                            num_followers = str(tweet_user_data['followers_count'])#用户粉丝量
                            num_statuses = str(tweet_user_data['statuses_count'])#该用户累计发布的推文数量
                            verified = str(tweet_user_data['verified'] - 0)#是否为认证用户（布尔值 → 转为 0/1）
                            geo_position = str(tweet_user_data['geo_enabled'] - 0)#是否开启地理定位功能（布尔值）
                            # skip time
                            num_favorite = str(tweet_user_data['favourites_count'])#用户点赞过的 tweet 数量（即“收藏”数量）
                            profile_background = str(tweet_user_data['profile_use_background_image'] - 0) #用户是否启用背景图（布尔）
                            profile = str(tweet_user_data['default_profile'] - 0) #	用户是否使用默认主页样式
                            profile_image = str(tweet_user_data['default_profile_image'] - 0) #用户是否使用默认头像
                            #所有布尔字段都被强制转为 0/1 (bool - 0)，成为结构特征
                            user_features.append([num_friends, num_followers, num_statuses,
                                                  verified, geo_position, num_favorite, profile_background,
                                                  profile, profile_image])

                        retweet_user_data=re["user"]
                        if retweet_user_data['id_str'] not in uid:
                            uid.append(retweet_user_data['id_str'])
                            num_friends = str(retweet_user_data['friends_count'])
                            # skip num word description
                            # skip num word name
                            num_followers = str(retweet_user_data['followers_count'])
                            num_statuses = str(retweet_user_data['statuses_count'])
                            verified = str(retweet_user_data['verified'] - 0)
                            geo_position = str(retweet_user_data['geo_enabled'] - 0)
                            # skip time
                            num_favorite = str(retweet_user_data['favourites_count'])
                            profile_background = str(retweet_user_data['profile_use_background_image'] - 0)
                            profile = str(retweet_user_data['default_profile'] - 0)
                            profile_image = str(retweet_user_data['default_profile_image'] - 0)

                            user_features.append([num_friends, num_followers, num_statuses,
                                                  verified, geo_position, num_favorite, profile_background,
                                                  profile, profile_image])
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(output_dir + 'user_features.txt', 'w') as f:
        for i in tqdm(range(len(uid)), 'writing user features'):
            f.write('u%s: ' %uid[i])
            f.write(' '.join(user_features[i]))
            f.write('\n')

    with open(output_dir + 'post_features.txt', 'w') as f:
        for i in tqdm(range(len(pid)), 'writing post features'):
            f.write('p%s: ' %pid[i])
            f.write(' '.join(post_features[i]))
            f.write('\n')

    
if __name__ == '__main__':
    #原始数据示例（post_features.txt）p123456789: 10 5 0
    with open('processed_data/FakeNewsNet/politifact/post_features.txt','r') as f:
        post_features=f.readlines()
    post_features=[line.split() for line in post_features]

    post_features = np.array(post_features)
    #提取点赞数和是否引用（即后两个字段）进行归一化
    matrix = post_features[:, [1,2]]
    #将每列值线性缩放到 0,1 区间，变成比例值。
    nor = preprocessing.MinMaxScaler() 
    nor_matrix = nor.fit_transform(matrix)

    post_features = post_features.tolist()
    nor_matrix = nor_matrix.tolist()
    #统一数值范围，但保留了 retweet_count 原始值。
    with open('processed_data/FakeNewsNet/politifact/post_features_onehot.txt', 'w') as f:
        for i, j in zip(post_features, nor_matrix):
            f.write('%s: %s %s %s\n' % (i[0], j[0], j[1],i[3],))
        f.write('%s: %s %s %s\n' % ('uPADDING', '0', '0', '0' ))
#提供一个默认值节点 uPADDING，用于图模型训练时填充（padding）不完整节点邻居的特征值，防止维度对齐问题
