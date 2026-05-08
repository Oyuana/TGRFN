import os
import json
from tqdm import tqdm

#遍历 FakeNewsNet 数据集，统计新闻、推文、转推、图片等各项指标，最终以 LaTeX 形式 输出统计结果。


ds_path = 'FakeNewsNet-Dataset/FakeNewsNet_Dataset'
datasets = ['politifact_fake','politifact_real', 'gossipcop_fake', 'gossipcop_real']

keys = [
    '# News articles',
    '# News with images downloaded',
    '# Users posting tweets',
    '# Tweets posting news', 
    '# Tweets with retweets',
    '# News with retweets', '# News with retweets downloaded',
    '# News with tweets', '# News with tweets downloaded',
    '# News images processed'
]

#用于存储所有发推文的用户 ID
uids = set()
#存储被转发的推文 ID
tweets_with_retweets = set()
#存储每个类别的数据统计信息
all_stats = {k1 : {} for k1 in datasets}

#分别用于存储 gossipcop 和 politifact 的新闻图片名称
goimgs=set()
poimgs=set()
#分别列出 gossipcop_images 和 politifact_images 目录下的所有图片文件名
goimg_list=os.listdir('FakeNewsNet-Dataset/NewsImages/gossipcop_images')
poimg_list=os.listdir('FakeNewsNet-Dataset/NewsImages/politifact_images')

#遍历 gossipcop 和 politifact 目录下的图片文件，并提取不带文件扩展名的 nid（新闻 ID）,并将 nid 存入 goimgs 和 poimgs 集合
for goimg in tqdm(goimg_list,desc="goimg_list"):
    goimg_nid=goimg.split(".")[0]
    if goimg_nid == '.DS_Store': continue
    goimgs.add(goimg_nid)
for poimg in tqdm(poimg_list,desc="poimg_list"):
    poimg_nid=poimg.split(".")[0]
    if poimg_nid == '.DS_Store': continue
    poimgs.add(poimg_nid)

#遍历四个数据集
for ds in datasets:
    print(ds,  'starts')
    stats = {key : 0 for key in keys} #存储统计数据
    uids = set()
    img_processed_nid = set() #存储该类别已处理图片的新闻 ID
    #统计当前类别的新闻数量
    news_list = os.listdir(os.path.join(ds_path, ds))
    stats['# News articles'] += len(news_list)
    #遍历新闻文件
    for nidpath in tqdm(news_list, desc=ds):
        #提取新闻 ID（去掉 politifact_ 或 gossipcop- 前缀）
        if nidpath[:len("politifact")]=="politifact":
            nid=nidpath[len("politifact"):]
        else:
            nid=nidpath[len("gossipcop-"):]
        #检查是否有对应的图片，如果有，则加入 img_processed_nid 集合
        if nid in goimgs or nid in poimgs:
            img_processed_nid.add(nid)
        #定义当前新闻的 JSON 文件路径
        news_content_path = os.path.join(ds_path, ds, nidpath, 'news_content.json') #新闻内容
        tweet_path = os.path.join(ds_path, ds, nidpath, 'tweets.json') #与该新闻相关的推文
        retweet_path = os.path.join(ds_path, ds, nidpath, 'retweets.json') #与该新闻相关的转推
        #统计该新闻的推文数，提取发推用户 ID 并存入 uids 集合
        if os.path.isfile(tweet_path):
            stats['# News with tweets downloaded'] += 1
            with open(tweet_path,'r') as tf:
                tweet=json.load(tf)
                stats['# Tweets posting news'] += len(tweet['tweets'])
                for tweet_f in tweet['tweets']:
                    uids.add(tweet_f['user_id'])
        #统计该新闻的转推数，记录被转推的推文 ID
        if os.path.isfile(retweet_path):
            stats['# News with retweets'] += 1
            with open(retweet_path,'r') as rtf:
                retweet=json.load(rtf)
                for tweet_id,re in retweet.items():
                    if len(retweet[tweet_id])!=0:
                        stats['# News with retweets downloaded'] += 1
                        tweets_with_retweets.add(tweet_id)
                        for rt in retweet[tweet_id]:
                            uids.add(rt["user"]["id"])
    #计算 用户数、被转发推文数、已处理图片数。
    stats['# Users posting tweets'] = len(uids)
    stats['# Tweets with retweets'] = len(tweets_with_retweets)
    stats['# News images processed'] = len(img_processed_nid)
    print(ds, 'ends')
    all_stats[ds] = stats

print('#' * 10 + ' Overall Stats ' + '#' * 10)

print('\\toprule')

print(' ' * 40, end = "")
for ds in datasets:
    print(' & {:10} & {:10}'.format(ds, ds), end='')


print('\\\\')
print(' ' * 40, end = "")
for i in range(2):
    print(' & {:10} & {:10}'.format('Fake', 'Real'), end='')

    
print('\\\\')

print('\\midrule')

for k in keys:
    print('{:40}'.format(k.replace('#', "\\#")), end='')
    for ds in datasets:
        # for ss in subset:
        print(' & {:10}'.format(int(all_stats[ds][k])), end='')
    print('\\\\')

print('\\bottomrule')
    