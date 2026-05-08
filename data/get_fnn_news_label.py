#!/usr/bin/env python
# coding: utf-8

import os
from tqdm import tqdm

#从 FakeNewsNet 数据集中读取新闻的真假类型（fake/ 或 real/），并为每篇新闻生成一个标签（0 或 1），保存成 news_label.txt 文件。

#负责为指定数据集（如 politifact 或 gossipcop）生成 ID → label 文件。输出格式：1234 0
def get_news_label(dataset):
    
    news_label = list()
    print('reading %s...' %dataset)
    pathway = 'FakeNewsNet-Dataset/FakeNewsNet_Dataset/%s/' % dataset
    output_dir = 'processed_data/FakeNewsNet/%s/' % dataset
    
    for news_type in ['fake/', 'real/']:
        print('reading news info from %s file...' %news_type)
        if news_type == 'fake/':
            label = '0'
        else:
            label = '1'
        news_list = os.listdir(pathway + news_type)
        for n in news_list:
            n_id=n[len("politifact"):] if n[:len("politifact")] == "politifact" else n[len("gossipcop-"):]
            news_label.append([n_id, label])

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(output_dir + 'news_label.txt', 'w') as f:
        for news in news_label:
            f.write(' '.join(news) + '\n')

if __name__ == '__main__':
    
    get_news_label('politifact')
    get_news_label('gossipcop')
