import statistics
import pandas as pd
from torch.utils.data import Subset, SubsetRandomSampler
from ann.preprocess import Preprocess
import gensim
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import sklearn.metrics as sm

HIDDEN_SIZE = 1024
HIDDEN_SIZE_2 = 32

# 評估指標
def print_evaluate(alllabels,allpreds):
    # 混淆矩陣
    matrixes = sm.confusion_matrix(alllabels,allpreds)
    print("confusion matrix:")
    print(matrixes)

    # 準確率
    correct_count = sum([int(i == j) for i, j in zip(allpreds, alllabels)])
    total_data = len(allpreds)
    acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
    print("correct count :", correct_count, "total data :", total_data, "accuracy :", acc)

    # 計算評估直
    precision, recall, macro_f1, _ = sm.precision_recall_fscore_support(y_true=alllabels, y_pred=allpreds, average='macro')
    micro_f1 = sm.f1_score(y_true=alllabels, y_pred=allpreds, average='micro')
    print("precision :", precision)   
    print("recall :", recall) 
    print("macro-f1 :", macro_f1) 
    print("micro-f1 :", micro_f1)
    return matrixes,correct_count, total_data, acc, precision, recall, macro_f1, micro_f1


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear1 = nn.Linear(input_dim, HIDDEN_SIZE)
        self.linear2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE_2)
        self.linear3 = nn.Linear(HIDDEN_SIZE_2, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid_(x)
        x = self.linear2(x)
        x = torch.sigmoid_(x)
        return torch.relu(self.linear3(x))


def main():
    # Load Data
    train_df = pd.read_csv("./data/train500_tag.csv")

    # Load word2vec pre-train model
    embedding_model = gensim.models.Word2Vec.load('./embedding_data/word2vec.model')

    preprocess_lst = []

    for i, data in train_df.iterrows():
        preprocess = Preprocess(str(data.loc['id']), str(data.loc['message']), str(data.loc['title']),
                                str(data.loc['anchor']))
        # 留中文、繁轉簡、去標點、分詞
        preprocess = preprocess.find_chinese().tw_to_simple().participle()
        preprocess.embedding(embedding_model)
        preprocess_lst.append(preprocess)

    # 設定 seed
    torch.manual_seed(1234)
    np.random.seed(1234)

    # 選擇要預測的欄位順序
    output_predict_index = train_df.columns.get_loc('label')

    train = train_df.to_numpy()
    train_size = len(preprocess_lst)
    feature_size = len(preprocess_lst[0].embedding_message) + len(preprocess_lst[0].embedding_title) + len(
        preprocess_lst[0].embedding_anchor)

    train_x = np.empty([train_size, feature_size], dtype=float)
    train_y = np.empty([train_size, 1], dtype=float)

    for idx in range(train_size):
        message_arr = np.array(preprocess_lst[idx].embedding_message)
        title_arr = np.array(preprocess_lst[idx].embedding_title)
        anchor_arr = np.array(preprocess_lst[idx].embedding_anchor)
        train_x[idx, :] = np.hstack([message_arr, title_arr, anchor_arr])
        train_y[idx, 0] = train[idx][output_predict_index]

    # 交叉驗證紀錄者
    class Recorder:
        def __init__(self):
            self.train_loss = []
            self.validation_loss = []
            self.train_accuracy = []
            self.validation_accuracy = []

        def mean(self):
            self.train_loss = statistics.mean(self.train_loss)
            self.validation_loss = statistics.mean(self.validation_loss)
            self.train_accuracy = statistics.mean(self.train_accuracy)
            self.validation_accuracy = statistics.mean(self.validation_accuracy)

    # 交叉驗證
    k_fold = 5
    recorder = Recorder()

    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=1234)
    for train_idx, valid_idx in skf.split(train_x, train_y):
        val_x_ = torch.tensor(Subset(train_x, valid_idx), dtype=torch.float32)
        val_y_ = torch.tensor(Subset(train_y, valid_idx), dtype=torch.float32)
        train_x_ = torch.tensor(Subset(train_x, train_idx), dtype=torch.float32)
        train_y_ = torch.tensor(Subset(train_y, train_idx), dtype=torch.float32)

        # 轉型
        # train_x = torch.from_numpy(train_x.astype(np.float32))
        # train_y = torch.from_numpy(train_y.astype(np.float32))
        # 切割Validation
        # train_x_, val_x_, train_y_, val_y_ = train_test_split(train_x, train_y, random_state=1234, train_size=0.8)

        print("train size:", end="")
        print(len(train_x_))
        print("validation size:", end="")
        print(len(val_x_))
        print("feature size:", end="")
        print(feature_size)

        model = LinearRegression(feature_size, 1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

        loss_dict = []
        val_loss_dict = []
        val_accuracy_dict = []
        epochs = 400
        for epoch in range(epochs):
            train_correct = 0
            val_correct = 0

            # forward pass and loss
            y_predicted = model(train_x_)

            predicted = []
            for i, x in enumerate(y_predicted.detach().numpy()):
                if x[0] >= 0.5:
                    predicted.append([1.])
                else:
                    predicted.append([0.])
            predicted = torch.FloatTensor(predicted)

            # 計算train_acc
            train_correct += torch.eq(predicted, train_y_).sum().float().item()

            loss = criterion(y_predicted, train_y_)
            # init optimizer
            optimizer.zero_grad()
            # backward pass
            loss.backward()
            # update
            optimizer.step()

            # Validation forward loss
            val_y_predicted = model(val_x_)

            val_predicted = []
            for i, x in enumerate(val_y_predicted.detach().numpy()):
                if x[0] >= 0.5:
                    val_predicted.append([1.])
                else:
                    val_predicted.append([0.])
            val_predicted = torch.FloatTensor(val_predicted)

            # 計算val_acc
            val_correct += torch.eq(val_predicted, val_y_).sum().float().item()
            val_accuracy_dict.append(val_correct / len(val_x_))

            val_loss = criterion(val_y_predicted, val_y_)

            

            # 保存loss
            loss_dict.append(loss.item())
            val_loss_dict.append(val_loss.item())

            if (epoch + 1) % 10 == 0:
                print(
                    f'epoch[{epoch + 1}/{epochs}] train_loss = {loss.item(): .4f}        , validation_loss = {val_loss.item(): .4f}')
                print(
                    f'               train_accuracy = {train_correct / len(train_x_): .4f} , validation_accuracy = {val_correct / len(val_x_): .4f}')
        print_evaluate(val_y_,val_predicted)
        
        # Recorder紀錄
        recorder.train_loss.append(loss.item())
        print('record train_loss:', loss.item())
        recorder.validation_loss.append(val_loss.item())
        print('record validation_loss:', val_loss.item())
        recorder.train_accuracy.append(train_correct / len(train_x_))
        print('record train_accuracy:', train_correct / len(train_x_))
        recorder.validation_accuracy.append(val_correct / len(val_x_))
        print('record val_accuracy:', val_correct / len(val_x_))

    recorder.mean()
    print("平均train_loss : ", recorder.train_loss)
    print("平均validation_loss : ", recorder.validation_loss)
    print("平均train_accuracy : ", recorder.train_accuracy)
    print("平均validation_accuracy : ", recorder.validation_accuracy)

    # 畫loss變化圖
    plt.plot(val_loss_dict, label='LOSS iteration')
    plt.legend()
    plt.show()

    # 畫accuracy變化圖
    plt.plot(val_accuracy_dict, label='ACCURACY iteration')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
