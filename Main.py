import os
from BertFineTune import BertFineTune
from datasets import load_dataset
from CatchFeatures import catch_features
from HardPrompt import HardPrompt
from Hyperparameters import Hyperparameters
from P_tuning_v1 import P_tuning_v1
from tool.Evaluator import Evaluator
from tool.LeaningCurveDrawer import LeaningCurveDrawer

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
hyperparameters = Hyperparameters()

# Load Data
def load_data(train_size = hyperparameters.train_size):
    datasets = load_dataset("csv", data_files="./data/train500_tag.csv", split='train')
    datasets = datasets.train_test_split(test_size=hyperparameters.test_size+train_size, train_size=hyperparameters.val_size, shuffle=True, seed=1234)
    val = datasets['train']
    datasets = datasets["test"].train_test_split(test_size=hyperparameters.test_size, train_size=train_size, shuffle=True, seed=1234)
    datasets['validation'] = val
    return datasets
"""
prompt_tune模型
"""
def prompt():
    # Load Data
    datasets = load_data()

    # prompt_tune
    # prompt = HardPrompt("bert", "bert-base-chinese")
    prompt = P_tuning_v1("t5", "shibing624/prompt-t5-base-chinese")
    prompt.datasets_process(datasets)

    # 設定參數
    prompt.training_args.learn_rate = hyperparameters.learn_rate
    prompt.training_args.epoch = hyperparameters.epoch
    prompt.training_args.weight_decay = hyperparameters.weight_decay
    prompt.training_args.batch_size = hyperparameters.batch_size
    prompt.training_args.staging_point = './temp/p_tuning_t5.pth'
    prompt.training_args.early_stopping_patience = 7
    hyperparameters.print("hard prompt")
    
    # Zero-Shot
    labels,predic = prompt.zero_shot()
    evaluator_z = Evaluator(labels, predic)
    evaluator_z.matrixes()
    evaluator_z.accuracy()
    evaluator_z.precision_recall_fscore()
    
    # 訓練
    prompt.set_device()
    prompt.train()

    # 預測
    print("--------test---------")
    labels,predic = prompt.label_predict()
    evaluator = Evaluator(labels, predic)
    evaluator.matrixes()
    evaluator.accuracy()
    evaluator.precision_recall_fscore()
    evaluator.save("output/hard_prompt_")

"""
fine_tune模型
"""
def fine_tune():

    hyperparameters.print()

    # Load Data
    datasets = load_data()

    # BertFineTune
    bertFineTune = BertFineTune(hyperparameters.checkpoint)
    bertFineTune.datasets_process(datasets)
    # 設定參數
    bertFineTune.training_args.learning_rate = hyperparameters.learn_rate
    bertFineTune.training_args.num_train_epochs = hyperparameters.epoch
    bertFineTune.training_args.weight_decay = hyperparameters.weight_decay
    bertFineTune.training_args.per_device_train_batch_size = hyperparameters.batch_size
    bertFineTune.training_args.per_device_eval_batch_size = hyperparameters.batch_size

    # 訓練
    bertFineTune.train()
    bertFineTune.trainer.save_model("final_checkpoint.ckpt")

    # 預測
    labels,predic = bertFineTune.label_predict()
    evaluator = Evaluator(labels, predic)
    evaluator.matrixes()
    evaluator.accuracy()
    evaluator.precision_recall_fscore()

"""
畫learn_curve
"""
def learn_curve():

    # 設定training_set
    training_set = [1,50,100,150,200,250,300,350,400]

    evaluator = Evaluator()

    for size in training_set:
        train_size = size

        # Load Data
        datasets = load_data(train_size)

        # BertFineTune
        bertFineTune = BertFineTune(hyperparameters.checkpoint)

        bertFineTune.datasets_process(datasets)
        # 設定參數
        bertFineTune.training_args.learning_rate = hyperparameters.learn_rate
        bertFineTune.training_args.num_train_epochs = hyperparameters.epoch
        bertFineTune.training_args.weight_decay = hyperparameters.weight_decay
        bertFineTune.training_args.per_device_train_batch_size = hyperparameters.batch_size
        bertFineTune.training_args.per_device_eval_batch_size = hyperparameters.batch_size

        # 訓練
        bertFineTune.train()

        # 預測
        labels,predic = bertFineTune.label_predict()
        evaluator.labels = labels
        evaluator.preds = predic
        evaluator.matrixes()
        evaluator.accuracy()
        evaluator.precision_recall_fscore()
    
    # 畫出learn_curve
    leaning_curve_drawer = LeaningCurveDrawer(training_set,[(evaluator,"fine_tune")])
    leaning_curve_drawer.accuracy_curve()
    leaning_curve_drawer.recall_curve()
    leaning_curve_drawer.precision_curve()
    leaning_curve_drawer.macro_f1_curve()

if __name__ == '__main__':
    # catch_features("./data/train10.csv","./data/train10_catch.json", "json")
    fine_tune()
    # prompt()
    # learn_curve()