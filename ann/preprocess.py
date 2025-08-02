import jieba
import re
from opencc import OpenCC
import jieba.posseg as pseg
import torch
from transformers import AutoTokenizer, AutoModel
from keras_preprocessing.sequence import pad_sequences

MAX_MESSAGE_SENTENCE_LONG = 3187
MAX_TITLE_SENTENCE_LONG = 32
MAX_ANCHOR_SENTENCE_LONG = 1052
VECTOR_SIZE = 11


class Preprocess:
    def __init__(self, id=None, message=None, title=None, anchor=None):
        self.url = id
        # HTML的title
        self.title = title
        # HTML的anchor
        self.anchor = anchor
        # boilerPlateRemoval的文字
        self.message = message
        # 詞性標注
        self.word_flag_pair = None
        # 分詞
        self.message_seg_list = None
        self.title_seg_list = None
        self.anchor_seg_list = None
        self.embedding_message = None
        self.embedding_title = None
        self.embedding_anchor = None

    # 只留中文、英文、數字
    def find_chinese(self):
        pattern = re.compile(r'[^\u4e00-\u9fa5^a-z^A-Z^0-9]')
        self.message = re.sub(pattern, ' ', self.message)
        self.title = re.sub(pattern, ' ', self.title)
        self.anchor = re.sub(pattern, ' ', self.anchor)
        return self

    # 繁轉簡
    def tw_to_simple(self):
        cc = OpenCC('tw2s')
        self.message = cc.convert(self.message)
        self.title = cc.convert(self.title)
        self.anchor = cc.convert(self.anchor)
        return self

    # 簡轉繁
    def simple_to_tw(self):
        cc = OpenCC('s2tw')
        self.message = cc.convert(self.message)
        self.title = cc.convert(self.title)
        self.anchor = cc.convert(self.anchor)
        return self

    # 去標點符號
    def remove_punctuation(self):
        # sentenceClean = []
        remove_chars = '[̶●「」。／·’!"\#$%&\'()＃！（）*+,-./:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~]+'
        self.message = re.sub(remove_chars, "", self.message)
        self.title = re.sub(remove_chars, "", self.title)
        self.anchor = re.sub(remove_chars, "", self.anchor)
        return self

    # 詞性標注
    def part_of_speech_tagging(self):
        words_pair = pseg.cut(self.message)
        self.word_flag_pair = " ".join(["{0}/{1}".format(word, flag) for word, flag in words_pair])
        # print(self.word_flag_pair)
        return self

    # 分詞
    def participle(self):
        message_seg_list = jieba.cut(self.message, cut_all=False, HMM=True)
        title_seg_list = jieba.cut(self.title, cut_all=False, HMM=True)
        anchor_seg_list = jieba.cut(self.anchor, cut_all=False, HMM=True)
        self.message_seg_list = [x.strip() for x in list(message_seg_list) if x.strip() != '']
        self.title_seg_list = [x.strip() for x in list(title_seg_list) if x.strip() != '']
        self.anchor_seg_list = [x.strip() for x in list(anchor_seg_list) if x.strip() != '']
        # print("message:" + '/'.join(self.message_seg_list))
        # print("title:" + '/'.join(self.title_seg_list))
        # print("anchor:" + '/'.join(self.anchor_seg_list))
        return self

    def embedding(self, model):
        temp_lst = []
        for message in self.message_seg_list:
            if message in model.wv:
                temp_lst.append(model.wv[message])
        # sentence長度
        message_len = MAX_MESSAGE_SENTENCE_LONG - len(temp_lst)
        # flatten
        self.embedding_message = [item for sublist in temp_lst for item in sublist]
        # 補0到最長的sentence
        self.embedding_message += [0 for i in range(message_len * VECTOR_SIZE)]

        temp_lst = []
        for title in self.title_seg_list:
            if title in model.wv:
                temp_lst.append(model.wv[title])
        # sentence長度
        title_len = MAX_TITLE_SENTENCE_LONG - len(temp_lst)
        # flatten
        self.embedding_title = [item for sublist in temp_lst for item in sublist]
        # 補0到最長的sentence
        self.embedding_title += [0 for i in range(title_len * VECTOR_SIZE)]

        temp_lst = []
        for anchor in self.anchor_seg_list:
            if anchor in model.wv:
                temp_lst.append(model.wv[anchor])
        # sentence長度
        anchor_len = MAX_ANCHOR_SENTENCE_LONG - len(temp_lst)
        # flatten
        self.embedding_anchor = [item for sublist in temp_lst for item in sublist]
        # 補0到最長的sentence
        self.embedding_anchor += [0 for i in range(anchor_len * VECTOR_SIZE)]
        return self

    def bert_embedding(self, tokenizer, bert_model):
        max_message_embedding_long = 8620
        max_title_embedding_long = 69
        max_anchor_embedding_long = 1521

        # Preprocess
        sent = self.message
        # a = tokenizer(sent)
        sent_token = tokenizer.encode(sent)
        sent_token_padding = pad_sequences([sent_token], maxlen=max_message_embedding_long, padding='post', dtype='int')
        self.embedding_message = sent_token_padding[0].tolist()

        sent = self.title
        sent_token = tokenizer.encode(sent)
        sent_token_padding = pad_sequences([sent_token], maxlen=max_title_embedding_long, padding='post', dtype='int')
        masks = [[float(value > 0) for value in values] for values in sent_token_padding]

        # Convert
        inputs = torch.tensor(sent_token_padding)
        masks = torch.tensor(masks)
        embedded, _ = bert_model(inputs, attention_mask=masks)

        self.embedding_title = sent_token_padding[0].tolist()

        sent = self.anchor
        sent_token = tokenizer.encode(sent)
        sent_token_padding = pad_sequences([sent_token], maxlen=max_anchor_embedding_long, padding='post',
                                           dtype='int')
        self.embedding_anchor = sent_token_padding[0].tolist()

        return self

