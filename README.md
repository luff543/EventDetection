# Event Detection

一個基於BERT的網頁事件檢測系統，用於判斷網頁內容是否為活動相關頁面。

## 項目概述

本系統使用深度學習技術，特別是BERT模型，來自動識別網頁內容是否包含事件信息。系統可以處理網頁的多個特徵，包括文本內容、標題、錨點文字等，並提供高準確度的事件檢測能力。

## 主要功能

- **網頁特徵提取**: 自動從URL抓取網頁的標題、錨點文字、錨點鏈接等特徵
- **BERT模型訓練**: 支持BERT模型的微調訓練
- **模型推理**: 使用訓練好的模型對新數據進行預測
- **批量處理**: 與Apache Solr集成，支持大規模數據的自動化處理
- **學習曲線分析**: 提供模型訓練過程的可視化分析

## 系統要求

- Python 3.7+
- CUDA支持的GPU (推薦)
- Apache Solr (用於Solr.py功能)

## 安裝

1. 克隆項目：
```bash
git clone <repository-url>
cd EventDetection
```

2. 安裝依賴：
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 特徵提取 (CatchFeatures.py)

從CSV文件中的URL提取網頁特徵：

```python
from CatchFeatures import catch_features

# 提取特徵並保存為CSV
catch_features("./data/input.csv", "./data/output.csv", "csv")

# 提取特徵並保存為JSON (用於推理)
catch_features("./data/input.csv", "./data/output.json", "json")
```

**輸入要求：**
- CSV文件必須包含 `id` (網站URL) 和 `message` (PlateRemoval擷取的文字) 欄位
- 如需移除額外欄位，可在程式碼第13行處添加drop語句

**輸出：**
- 包含 `id`, `message`, `title`, `anchor`, `anchor_link` 五個欄位的CSV/JSON文件

### 2. 模型訓練 (Main.py)

訓練BERT模型：

```python
python Main.py
```

**主要功能：**
- `train_model()`: 訓練模型，將最佳模型保存為 `final_checkpoint.ckpt`
- `learn_curve()`: 生成學習曲線圖表
- 全局變量設定訓練參數

### 3. 模型推理 (Inference.py)

使用訓練好的模型進行預測：

```bash
python Inference.py ./data/train10_catch.json
```

**輸入格式：**
```json
[
  {
    "id": "id1",
    "message": "message 1",
    "title": "title 1",
    "anchor": "anchor 1",
    "anchor_link": "anchor link 1"
  },
  {
    "id": "id2",
    "message": "message 2",
    "title": "title 2",
    "anchor": "anchor 2",
    "anchor_link": "anchor link 2"
  }
]
```

**輸出：**
- 返回預測結果列表，例如 `[1, 1, 1, 0, 1, 0, 0, 1, 1, 1]`
- `1` 代表預測為活動網頁，`0` 代表非活動網頁

### 4. Solr自動化處理 (Solr.py)

與Apache Solr集成的自動化事件檢測服務：

```bash
python -m Solr.py
```

**功能說明：**
- 自動從Solr搜索引擎查詢未處理的網頁數據
- 對每批數據進行事件檢測預測
- 將預測結果更新回Solr數據庫
- 支持連續運行，每60秒檢查一次新數據

**Solr集成特性：**
- 連接到指定的Solr服務器 (`http://140.115.54.61/solr/WebEventExtraction/`)
- 查詢條件：`-isJudgeEvent:* AND boilerPlateRemovalMessage:*`
- 更新字段：`isJudgeEvent`, `isEvent`, `isEventScore`, `isNotEventScore`
- 錯誤處理：包含超時和連接錯誤的容錯機制

## 文件結構

```
EventDetection/
├── CatchFeatures.py          # 網頁特徵提取
├── Main.py                   # 模型訓練主程序
├── Inference.py              # 模型推理
├── Solr.py                   # Solr自動化處理
├── BertFineTune.py           # BERT模型微調
├── HardPrompt.py             # Hard Prompt方法
├── P_tuning_v1.py           # P-tuning方法
├── Hyperparameters.py        # 超參數設定
├── requirements.txt          # 依賴包列表
├── data/                     # 數據文件夾
│   ├── train10_catch.json   # 範例推理數據
│   ├── train500_tag.csv     # 訓練數據
│   └── ...
├── output/                   # 輸出結果
├── ann/                      # ANN相關模組
├── log/                      # 日誌文件
└── final_checkpoint.ckpt     # 訓練好的模型
```

## 模型輸出說明

- **預測值**: 0 (非事件網頁) 或 1 (事件網頁)
- **置信度分數**: 提供每個類別的預測置信度
- **批量處理**: 支持單個樣本或批量數據的預測

## 注意事項

1. 運行前請確保已正確安裝所有依賴包
2. 使用GPU訓練時，請設置適當的CUDA環境變量
3. Solr.py需要有效的Solr服務器連接
4. 訓練數據需要包含正確的標籤格式

## 故障排除

- **CUDA相關錯誤**: 檢查CUDA驅動和環境配置
- **Solr連接錯誤**: 確認Solr服務器地址和端口正確
- **內存不足**: 調整批量大小或使用更小的模型
- **依賴包問題**: 確保使用正確的Python版本和虛擬環境
