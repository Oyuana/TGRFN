import os
import json

root_dir = '/home/zhangrq/DPSG/data/PHEME/pheme-rnr-dataset'  # 相对路径：data/PHEME

stats = {
    'non_rumours': 0,
    'false_rumours': 0,
    'posts': 0,
    'users': set(),
    'images': 0
}

def count_tweets_in_thread(path):
    count = 0
    for dirpath, _, filenames in os.walk(path):
        for fname in filenames:
            if fname.endswith('.json'):
                full_path = os.path.join(dirpath, fname)
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        count += 1
                        # 用户
                        if 'user' in data and 'id_str' in data['user']:
                            stats['users'].add(data['user']['id_str'])
                        # 图片（简单版本）
                        if 'entities' in data and 'media' in data['entities']:
                            stats['images'] += 1
                except Exception as e:
                    print(f"[跳过无效文件] {full_path}: {e}")
    return count

# 遍历每个事件
for event in os.listdir(root_dir):
    event_path = os.path.join(root_dir, event)
    if not os.path.isdir(event_path):
        continue
    for label in ['non-rumours', 'rumours']:
        label_path = os.path.join(event_path, label)
        if not os.path.exists(label_path):
            continue
        for thread in os.listdir(label_path):
            thread_path = os.path.join(label_path, thread)
            if not os.path.isdir(thread_path):
                continue
            if label == 'non-rumours':
                stats['non_rumours'] += 1
            else:
                stats['false_rumours'] += 1

            # 🔁 只统计 reactions 目录（排除 source-tweet）
            reactions_path = os.path.join(thread_path, 'reactions')
            if os.path.exists(reactions_path):
                stats['posts'] += count_tweets_in_thread(reactions_path)


# 输出统计结果
print("\n=== PHEME 数据集统计结果 ===")
print(f"Non-rumours 数量：{stats['non_rumours']}")
print(f"False rumours 数量：{stats['false_rumours']}")
print(f"总 posts 数量（源推 + 回复）：{stats['posts']}")
print(f"唯一用户数（user_id）：{len(stats['users'])}")
print(f"包含图片的推文数：{stats['images']}")
