# class Hyperparameters():
    
#     checkpoint = "bert-base-chinese"
#     train_size = 400
#     val_size = 20
#     test_size = 80
#     epoch = 30
#     batch_size = 8
#     learn_rate = 1e-5
#     weight_decay = 0
    
#     @staticmethod
#     def print():
#         print("-------------------------")
#         print("TRAIN_SIZE :",Hyperparameters.train_size)
#         print("VALIDATION_SIZE :",Hyperparameters.val_size)
#         print("TEST_SIZE :",Hyperparameters.test_size)    
#         print("EPOCH :",Hyperparameters.epoch)
#         print("BATCH_SIZE :",Hyperparameters.batch_size)
#         print("LEARN_RATE :",Hyperparameters.learn_rate)
#         print("WEIGHT_DECAY :",Hyperparameters.weight_decay)
#         print("-------------------------")

class Hyperparameters():
    def __init__(self):
        self.checkpoint = "bert-base-chinese"
        self.train_size = 300
        self.val_size = 100
        self.test_size = 100
        self.epoch = 30
        self.batch_size = 8
        self.learn_rate = 1e-7
        self.weight_decay = 0
        self.max_seq_length = 256

    def print(self, title=""):
        print(title)
        print("-------------------------")
        print("TRAIN_SIZE :",self.train_size)
        print("VALIDATION_SIZE :",self.val_size)
        print("TEST_SIZE :",self.test_size)    
        print("EPOCH :",self.epoch)
        print("BATCH_SIZE :",self.batch_size)
        print("LEARN_RATE :",self.learn_rate)
        print("WEIGHT_DECAY :",self.weight_decay)
        print("-------------------------")

    