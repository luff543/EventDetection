import requests
import csv

url = "https://eventgo.widm.csie.ncu.edu.tw:3006"
contents = "/activity?"
query = "sort=start_time&asc=true&from=1514736000000&to=1706630400000&type=Web%20Post&gps=25.033718,121.56481&p=2&num=500"

response = requests.get(url + contents + query)
my_json = response.json()

# print(my_json)
count = int(my_json['count'])
print(count)
print(my_json['events'][0].keys())

header = ['index', 'id', 'message']
data = []
for i, event in enumerate(my_json['events']):
    data.append([i + 1, event["id"], event["description"]])

with open('./data/train230213.csv', "w", encoding='UTF8', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)
