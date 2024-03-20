import json
import datetime
import os
import sys
from mpi4py import MPI

begin_time = datetime.datetime.now()

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
        print("No json file found!")
        sys.exit()

    largest_file = max(json_files, key=lambda file: os.path.getsize(os.path.join(directory, file)))#find the largest json file
    return largest_file

def merge_dicts(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1:
            dict1[key] += value
        else:
            dict1[key] = value
    return dict1

def get_params(tweet: json):
    data = tweet.get('doc').get('data')
    hour = data.get('created_at').split(':')[0]
    day = data.get('created_at').split('T')[0]
    sentiment = data.get('sentiment', None)
    return hour, day, sentiment

file_path = find_largest_json_file()

total_bytes = os.path.getsize(file_path)
each_bytes = total_bytes // size
begin_position = rank * each_bytes
end_position = (rank + 1) * each_bytes
current_position = begin_position

with open(file_path, 'r', encoding='utf-8') as tweet_file:
    tweet_str = ''
    tweet_file.seek(begin_position)
    tweet_file.readline()

    while (tweet_str := tweet_file.readline()) != '{}]}\n':
        if current_position > end_position:
            break

        tweet = json.loads(tweet_str[:-2])
        hour, day, sentiment = get_params(tweet)
        print(f"rank {rank}: hour{hour}")

        if hour not in happy_hour_dict:
            active_hour_dict[hour] = 0
            happy_hour_dict[hour] = 0

        if day not in happy_day_dict:
            active_day_dict[day] = 0
            happy_day_dict[day] = 0

        active_hour_dict[hour] += 1
        active_day_dict[day] += 1

        if sentiment and isinstance(sentiment, float):
            happy_day_dict[day] = happy_day_dict.get(day, 0) + sentiment
            happy_hour_dict[hour] = happy_hour_dict.get(hour, 0) + sentiment

        # 这里可能不准确，会导致bug数据错误
        current_position += len(tweet_str)

dict_list_list = comm.gather([happy_hour_dict, happy_day_dict, active_hour_dict, active_day_dict], root=0)

happy_hour_dict = {}
happy_day_dict = {}
active_hour_dict = {}
active_day_dict = {}

if rank == 0:
    for dict_list in dict_list_list:
        happy_hour_dict = merge_dicts(happy_hour_dict, dict_list[0])
        happy_day_dict = merge_dicts(happy_day_dict, dict_list[1])
        active_hour_dict = merge_dicts(active_hour_dict, dict_list[2])
        active_day_dict = merge_dicts(active_day_dict, dict_list[3])

    print(list(dict(sorted(happy_hour_dict.items(), key=lambda item: item[1])).items())[-1])
    print(list(dict(sorted(happy_day_dict.items(), key=lambda item: item[1])).items())[-1])
    print(list(dict(sorted(active_hour_dict.items(), key=lambda item: item[1])).items())[-1])
    print(list(dict(sorted(active_day_dict.items(), key=lambda item: item[1])).items())[-1])

    finish_time = datetime.datetime.now()
    print(finish_time - begin_time)