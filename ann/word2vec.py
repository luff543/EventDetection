"""
參考網站
https://www.twblogs.net/a/5ef22878ae25b7655256fd8e
"""
import math
from gensim.models import KeyedVectors, word2vec, Word2Vec
import multiprocessing
from sklearn.cluster import DBSCAN
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.pyplot as mp
import numpy as np
from sklearn.decomposition import PCA

# 全部有效words數量
TOTAL_EFFECT_WORDS = 3244

# 可以用BrownCorpus,Text8Corpus或lineSentence來構建sentences，一般大語料使用這個
sentences = list(word2vec.LineSentence('./embedding_data/sentences_no_stopwords.txt'))
# sentences = list(word2vec.Text8Corpus('data.txt'))

# vector_size大小為log(V)
vector_size = math.floor(math.log(TOTAL_EFFECT_WORDS, 2))
print("vector_size: %d" % vector_size)

# 訓練方式1
model = Word2Vec(sentences, vector_size=vector_size, window=5, sg=0,
                 workers=multiprocessing.cpu_count())  # sg：用於設置訓練算法，默認為0，對應CBOW算法；sg=1則採用skip-gram算法
# print('model:%s' % model)
# 訓練方式2
# #加載一個空模型
# model2 = Word2Vec(size=256,min_count=1)
# # 加載詞表
# model2.build_vocab(sentences)
# # 訓練
# model2.train(sentences, total_examples=model2.corpus_count, epochs=10)
# print(model2)


# 模型保存
# 方式一
model.save('./embedding_data/word2vec.model')
# 方式二
model.wv.save_word2vec_format('./embedding_data/word2vec.vector')

# 加載模型
# 方式一
# model = Word2Vec.load('word2vec.model')
# print(model)
# # 方式二
# model = KeyedVectors.load_word2vec_format('word2vec.vector')
# print(len(model.vectors))

# #增量訓練word2vec
# model.build_vocab(sentences_cut,update=True) #注意update = True 這個參數很重要
# model.train(sentences_cut,total_examples=model.corpus_count,epochs=10)
# print(model)

all_words = model.wv.key_to_index.keys()

words = ['活动']
related_1 = model.wv.most_similar('活动', topn=30)
# related_2 = model.wv.most_similar('时间', topn=10)
for related in related_1:
    words.append(related[0])
# for related in related_2:
#     words.append(related[0])
print(related_1)
# print(related_2)


def display_pca_scatterplot(model, words):
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False

    # Take word vectors
    word_vectors = np.array([model.wv[w] for w in words])

    # PCA, take the first 2 principal components
    twodim = PCA().fit_transform(word_vectors)[:, :2]

    # Draw
    plt.figure(figsize=(6, 6))
    plt.scatter(twodim[:, 0], twodim[:, 1], edgecolors='k', c='r')
    for word, (x, y) in zip(words, twodim):
        plt.text(x + 0.05, y + 0.05, word)
    plt.show()


display_pca_scatterplot(model, all_words)
display_pca_scatterplot(model, words)

# # 詞向量類聚
# vectors = [model.wv[word] for word in model.wv.index_to_key]
# labels = DBSCAN(eps=0.24, min_samples=3).fit(vectors).labels_
#
# # 詞向量可視化
# mp.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 顯示中文
# matplotlib.rcParams['axes.unicode_minus'] = False  # 顯示負號
# fig = mp.figure()
# ax = mplot3d.Axes3D(fig)  # 創建3D座標軸
# colors = ['red', 'blue', 'green', 'black']
# show_words = ['个展', '日期', '活动', '時間']
# for word, vector, label in zip(show_words, vectors, labels):
#     ax.scatter(vector[0], vector[1], vector[2], c=colors[label], s=500, alpha=0.4)
#     ax.text(vector[0], vector[1], vector[2], word, ha='center', va='center')
# mp.show()
