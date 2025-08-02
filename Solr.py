# coding: utf-8
import json
import sys
import os
from BertFineTune import BertFineTune

from Inference import inference_json
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


import pysolr
from pysolr import SolrError
import time

from datetime import datetime
import requests

# https://yonik.com/solr/atomic-updates/
def updateIsEvent(result, prediction, score):
    if prediction is not None:
        update_doc = {
            "id": result['id'],
            "isJudgeEvent": {"set": True},
            "isEvent": {"set": prediction},
            "isEventScore": {"set": score[1]},
            "isNotEventScore": {"set": score[0]}
        }
    else:
        update_doc = {
            "id": result['id'],
            "isJudgeEvent": {"set": True}
        }
    try:
        solr.add([update_doc])
    except SolrError:
        pass

solr = pysolr.Solr('http://140.115.54.61/solr/WebEventExtraction/', always_commit=True, timeout=100)

query = '-isJudgeEvent:* AND boilerPlateRemovalMessage:*'
# query = 'isJudgeEvent:* AND boilerPlateRemovalMessage:* AND isEvent:* AND -isEventScore:*'
# query = 'isJudgeEvent:false AND boilerPlateRemovalMessage:*'

# Load Model
bertFineTune = BertFineTune('./final_checkpoint.ckpt' ,inference=True)

condition = True

while condition == True:

    try:
        results = solr.search(query,
                                #   fl='id,boilerPlateRemovalMessage', rows=100)
                        fl='id,boilerPlateRemovalMessage, htmlTitle, htmlAnchors, htmlAnchorLinks', rows=100)

    except requests.exceptions.ReadTimeout:
        print("pysolr.SolrError: Connection to server")
        pass
    except pysolr.SolrError:
        print("pysolr.SolrError")
        pass

    for result in results:
        result['message'] = result["boilerPlateRemovalMessage"]
        # 刪除舊的key
        del result["boilerPlateRemovalMessage"]

        if 'htmlTitle' not in result:
            result['title'] = ''
        else:
            result['title'] = result['htmlTitle']
            del result["htmlTitle"]
        if 'htmlAnchors' not in result:
            result['anchor'] = ''
        else:
            result['anchor'] = result['htmlAnchors']
            del result["htmlAnchors"]

        if 'htmlAnchorLinks' not in result:
            result['anchor_link'] = ''
        else:
            result['anchor_link'] = result['htmlAnchorLinks']
            del result["htmlAnchorLinks"]
    
    # 轉為json檔
    json_results = json.dumps(results.docs, ensure_ascii=False)
    
    # 取得預測
    predictions, scores = inference_json(json_results,bertFineTune)

    for i in range(len(results.docs)):
        update_doc = {
            "id": results.docs[i]['id'],
            "isJudgeEvent": {"set": False},
            "isEvent": {"set": None},
            "isEventScore": {"set": 0},
            "isNotEventScore": {"set": 0}
        }
        try:
            solr.add([update_doc])
        except SolrError:
            pass
        try:
            solr.commit()
        except SolrError:
            pass
        
        updateIsEvent(results.docs[i], predictions[i], scores[i])

    print("time.sleep(60)")
    time.sleep(60)

