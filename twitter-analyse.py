import json
import datetime
import os
import sys
from mpi4py import MPI

begin_time = datetime.datetime.now()

# {day : (cnt, sentiment)}
# {hour  : (cnt, sentiment)}
hour_stats_dict = {}
day_stats_dict = {}

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

# merge dicts
def merge_dicts(main_dict, merging_dict):
    for key, (new_count, new_sentiment_sum) in merging_dict.items():
        if key in main_dict:
            current_count, current_sentiment_sum = main_dict[key]
            main_dict[key] = (current_count + new_count, current_sentiment_sum + new_sentiment_sum)
        else:
            main_dict[key] = (new_count, new_sentiment_sum)
    return main_dict

def get_params(tweet: json):
    data = tweet.get('doc').get('data')
    hour = data.get('created_at').split(':')[0]
    day = data.get('created_at').split('T')[0]
    sentiment = data.get('sentiment', None)
    return hour, day, sentiment

file_path = 'twitter-50mb.json'

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

        hour_cnt = 0
        hour_sentiment = 0.0
        day_cnt = 0
        day_sentiment = 0.0

        if hour in hour_stats_dict:
            hour_cnt, hour_sentiment = hour_stats_dict[hour]

        if day in day_stats_dict:
            day_cnt, day_sentiment = day_stats_dict[day]

        hour_cnt += 1
        day_cnt += 1

        if sentiment and isinstance(sentiment, float):
            hour_sentiment += sentiment
            day_sentiment += sentiment

        hour_stats_dict[hour] = (hour_cnt, hour_sentiment)
        day_stats_dict[day] = (day_cnt, day_sentiment)

        # 这里可能不准确，会导致bug数据错误
        current_position += len(tweet_str)

all_hour_stats = comm.gather(hour_stats_dict, root=0)
all_day_stats = comm.gather(day_stats_dict, root=0)

if rank == 0:
    combined_hour_stats = {}
    combined_day_stats = {}

    # merge hour dict
    for hour_stats in all_hour_stats:
        combined_hour_stats = merge_dicts(combined_hour_stats, hour_stats)

    # merge day dict
    for day_stats in all_day_stats:
        combined_day_stats = merge_dicts(combined_day_stats, day_stats)

    # Identify the hour and day with the highest sentiment sum and most records
    max_sentiment_hour = max(combined_hour_stats.items(), key=lambda item: item[1][1])
    max_records_hour = max(combined_hour_stats.items(), key=lambda item: item[1][0])
    max_sentiment_day = max(combined_day_stats.items(), key=lambda item: item[1][1])
    max_records_day = max(combined_day_stats.items(), key=lambda item: item[1][0])

    print(f"Hour with highest sentiment: {max_sentiment_hour[0]} (Sentiment: {max_sentiment_hour[1][1]})")
    print(f"Day with highest sentiment: {max_sentiment_day[0]} (Sentiment: {max_sentiment_day[1][1]})")
    print(f"Hour with most records: {max_records_hour[0]} (Records: {max_records_hour[1][0]})")
    print(f"Day with most records: {max_records_day[0]} (Records: {max_records_day[1][0]})")

    finish_time = datetime.datetime.now()
    print(f"Total processing time: {finish_time - begin_time}")
