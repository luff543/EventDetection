import argparse
import json
from datasets import Dataset, DatasetDict
from BertFineTune import BertFineTune
from tool.Evaluator import Evaluator
# import chardet
def inference(args):

    # 載入json list
    with open(args.path, 'rt', encoding='utf-8', errors='ignore') as f:
        data = json.load(f)
        
    converted_data = {"id": [], "message": [], "title": [], "anchor": [], "anchor_link": []}
    for example in data:
        converted_data["id"].append(example["id"])
        converted_data["message"].append(example["message"])
        converted_data["title"].append(example["title"])
        converted_data["anchor"].append(example["anchor"])
        converted_data["anchor_link"].append(example["anchor_link"])

    # 将转换后的数据创建为Dataset对象
    dataset = Dataset.from_dict(converted_data)

    # BertFineTune
    bertFineTune = BertFineTune('./final_checkpoint.ckpt', dataset ,inference=True)
    
    predictions = bertFineTune.predictions()
    print(predictions)
    return predictions

def inference_json(input_json,bertFineTune:BertFineTune):

    # 載入json list
    data = json.loads(input_json)
    
    converted_data = {"id": [], "message": [], "title": [], "anchor": [], "anchor_link": []}
    for example in data:
        converted_data["id"].append(example["id"])
        converted_data["message"].append(example["message"])
        converted_data["title"].append(example["title"])
        converted_data["anchor"].append(example["anchor"])
        converted_data["anchor_link"].append(example["anchor_link"])

    # 将转换后的数据创建为Dataset对象
    dataset = Dataset.from_dict(converted_data)

    # bertFineTune = BertFineTune('./final_checkpoint.ckpt', dataset ,inference=True)

    bertFineTune.datasets_process(dataset)
    predictions, scores = bertFineTune.predictions()
    print(predictions)
    return predictions, scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='input json path')
    args = parser.parse_args()
    inference(args)
