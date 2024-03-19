import json
import datetime
import os
import sys
from mpi4py import MPI

begin_time = datetime.datetime.now()

# TODO 重构，重新设计一下。一个需要的结果对应一个dict有点奇怪。
happy_hour_dict = {}
happy_day_dict = {}
active_hour_dict = {}
active_day_dict = {}

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# find the largest json file
def find_largest_json_file(directory="."): 
    json_files = [file for file in os.listdir(directory) if file.endswith(".json")] #loop for the json file
    if not json_files:
        return None  # if there is no json file, return none

    largest_file = max(json_files, key=lambda file: os.path.getsize(os.path.join(directory, file)))#find the largest json file
    return largest_file

# TODO 变量名 小驼峰， 函数名 大驼峰
def get_params(tweet: json):
    data = tweet.get('doc').get('data')
    hour = data.get('created_at').split(':')[0]
    day = data.get('created_at').split('T')[0]
    sentiment = data.get('sentiment', None)
    return hour, day, sentiment

# TODO 需要找到目录下最大的json文件，
file_path = find_largest_json_file()

total_bytes = os.path.getsize(file_path)
each_bytes = total_bytes // size
begin_position = rank * each_bytes
end_position = (rank + 1) * each_bytes
current_position = begin_position

# TODO 逻辑需要重写。
with open(file_path, 'r', encoding='utf-8') as tweet_file:
    tweet_str = ''
    tweet_file.seek(begin_position)
    tweet_file.readline()

    while (tweet_str := tweet_file.readline()) != '{}]}\n':
        if current_position > end_position:
            break

        tweet = json.loads(tweet_str[:-2])
        hour, day, sentiment = get_params(tweet)

        if hour not in happy_hour_dict:
            active_hour_dict[hour] = 0
            happy_hour_dict[hour] = 0

        if day not in happy_day_dict:
            active_day_dict[day] = 0
            happy_day_dict[day] = 0

        active_hour_dict[hour] += 1
        active_day_dict[day] += 1

        if sentiment and isinstance(sentiment, float):
            happy_day_dict[hour] = happy_day_dict.get(hour, 0) + sentiment
            happy_hour_dict[day] = happy_day_dict.get(hour, 0) + sentiment

        # TODO 改分组的逻辑
        current_position += len(tweet_str)

local_maxes = {
    "happiest hour ever": max(happy_hour_dict.items(), key=lambda x: x[1]),
    "happyiest day ever": max(happy_day_dict.items(), key=lambda x: x[1]),
    "most active hour ever": max(active_hour_dict.items(), key=lambda x: x[1]),
    "most active day ever": max(active_day_dict.items(), key=lambda x: x[1]),
}

all_maxes = comm.gather(local_maxes, root=0)

if rank == 0:
    global_maxes = {category: ("", 0) for category in local_maxes.keys()}

    for maxes in all_maxes:
        for category, (key, value) in maxes.items():
            if value > global_maxes[category][1]:
                global_maxes[category] = (key, value)

    for category, (key, value) in global_maxes.items():
        print(f" {category}: {key}, {value}")

    finish_time = datetime.datetime.now()
    print(finish_time - begin_time)