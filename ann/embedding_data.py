import pandas as pd
from ann.preprocess import Preprocess

# Load Data
train_df = pd.read_csv("./data/train500_tag.csv")
dic = {}
preprocess_lst = []
for i, data in train_df.iterrows():
    preprocess = Preprocess(str(data.loc['id']), str(data.loc['message']), str(data.loc['title']),
                            str(data.loc['anchor']))
    # 留中英文及數字、繁轉簡、去標點、分詞
    preprocess.find_chinese().tw_to_simple().remove_punctuation().participle()
    preprocess_lst.append(preprocess)

    # 統計次數
    for word in preprocess.message_seg_list + preprocess.title_seg_list + preprocess.anchor_seg_list:
        if word in dic:
            dic[word] += 1
        else:
            dic[word] = 1

count_in_range = 0
stopwords = set()
for word, count in dic.items():
    if 10 < count < 3000:
        count_in_range += 1
        print(word + ':' + str(count))
    else:
        stopwords.add(word)
print('-----------------------------')
print('stop word:%d' % len(stopwords))
print('train words:%d' % count_in_range)
total = len(dic)
print('total:%d' % total)

# stopwords寫入
path1 = './embedding_data/stopwords.txt'
with open(path1, 'w', encoding='UTF-8') as f1:
    f1.write('\n'.join(stopwords))
    f1.close()

# 去除stopwords
train_sentences = []
max_message_sentence_long = 0
max_title_sentence_long = 0
max_anchor_sentence_long = 0
for preprocess in preprocess_lst:
    new_words = []
    message_sentence_long = 0
    for word in preprocess.message_seg_list:
        if word not in stopwords:
            new_words.append(word)
            message_sentence_long += 1
    if message_sentence_long > max_message_sentence_long:
        max_message_sentence_long = message_sentence_long
    train_sentences.append(' '.join(new_words))

    new_words = []
    title_sentence_long = 0
    for word in preprocess.title_seg_list:
        if word not in stopwords:
            new_words.append(word)
            title_sentence_long += 1
    if title_sentence_long > max_title_sentence_long:
        max_title_sentence_long = title_sentence_long
    train_sentences.append(' '.join(new_words))

    new_words = []
    anchor_sentence_long = 0
    for word in preprocess.anchor_seg_list:
        if word not in stopwords:
            new_words.append(word)
            anchor_sentence_long += 1
    if anchor_sentence_long > max_anchor_sentence_long:
        max_anchor_sentence_long = anchor_sentence_long
    train_sentences.append(' '.join(new_words))

print('max_message_sentence_long:%d' % max_message_sentence_long)
print('max_title_sentence_long:%d' % max_title_sentence_long)
print('max_anchor_sentence_long:%d' % max_anchor_sentence_long)

# 訓練詞(清除stopwords)寫入
path2 = './embedding_data/sentences_no_stopwords.txt'
with open(path2, 'w', encoding='UTF-8') as f2:
    f2.write('\n'.join(train_sentences))
    f2.close()

# 訓練詞寫入
path3 = './embedding_data/sentences.txt'
with open(path3, 'w', encoding='UTF-8') as f3:
    for preprocess in preprocess_lst:
        f3.write('%s\n' % ' '.join(preprocess.message_seg_list))
        f3.write('%s\n' % ' '.join(preprocess.title_seg_list))
        f3.write('%s\n' % ' '.join(preprocess.anchor_seg_list))
    f3.close()
