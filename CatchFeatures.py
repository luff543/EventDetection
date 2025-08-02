import pandas as pd
from bs4 import BeautifulSoup
import requests

# 範例catch_features("./data/train.csv","./data/train.json", "json")
def catch_features(data_path, oup_path, oup_type="csv"):

    # Load Data
    train_df = pd.read_csv(data_path)

    # 處理掉不需要的欄位
    # train_df.drop(columns=['index'], inplace=True)
    # train_df.drop(columns=['uuid'], inplace=True)
    # train_df.drop(columns=['label'], inplace=True)

    # 抓取HTML
    title_lst = []
    anchor_lst = []
    anchor_link_lst = []
    for url in train_df['id']:
        try:
            if url[0:4] != "http":
                    url = "https://" + url
                    response = requests.get(url, timeout=10, verify=False)
            else:
                response = requests.get(url, timeout=10, verify=False)
            response.encoding='utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')

            # 取Title
            title = soup.title
            if title is not None:
                title_str = title.string            
            else:
                title_str = ' '

            # 取Anchor文字
            temp_anchor_lst = []
            temp_anchor_link_lst = []
            page_anchor = " "
            page_anchor_link = " "
            if soup.find_all('a'):
                for link in soup.find_all('a'):
                    # 取得anchor文字
                    anchor = link.get_text()
                    # 取得anchor link
                    anchor_link = link.get('href')
                    if anchor:
                        # 去空白
                        anchor = anchor.strip()
                        temp_anchor_lst.append(anchor)
                    if anchor_link:
                        anchor_link = anchor_link.strip()
                        temp_anchor_link_lst.append(anchor_link)
                page_anchor = " ".join(temp_anchor_lst)
                page_anchor_link = " ".join(temp_anchor_link_lst)
            # else:
            #     page_anchor = " "
            #     page_anchor_link = " "
        except(Exception) as e:
            print(e)

            if title_str is None:
                title_str = ' '
            if page_anchor is None:
                page_anchor = ' '
            if page_anchor_link is None:
                page_anchor_link = ' '
        # print('title_str: ', title_str)
        # print('page_anchor: ', page_anchor)
        # print('page_anchor_link: ', page_anchor_link)
        title_lst.append(title_str)
        anchor_lst.append(page_anchor)
        anchor_link_lst.append(page_anchor_link)

    train_df.insert(loc=len(train_df.columns), column='title', value=title_lst)
    train_df.insert(loc=len(train_df.columns), column='anchor', value=anchor_lst)
    train_df.insert(loc=len(train_df.columns), column='anchor_link', value=anchor_link_lst)
    
    # 存檔
    if oup_type == "csv":
        train_df.to_csv(oup_path, index=False)
    elif oup_type == "json":
        train_df.to_json(oup_path,force_ascii=False,orient='records')
