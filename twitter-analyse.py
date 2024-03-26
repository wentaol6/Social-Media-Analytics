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


def find_largest_json_file(directory="."):
    json_files = [file for file in os.listdir(directory) if file.endswith(".json")]
    if not json_files:
        print("No json file found!")
        sys.exit()

    largest_file = max(json_files, key=lambda file: os.path.getsize(os.path.join(directory, file)))
    return largest_file


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


file_path = find_largest_json_file()

total_bytes = os.path.getsize(file_path)
each_bytes = total_bytes // size
begin_position = rank * each_bytes
end_position = (rank + 1) * each_bytes if rank < size - 1 else total_bytes

with open(file_path, 'r', encoding='utf-8') as tweet_file:
    if rank != 0:
        # Skip to the first complete record for this rank
        tweet_file.seek(begin_position)
        while True:
            char = tweet_file.read(1)
            begin_position += 1
            if char == '{':
                tweet_file.seek(begin_position - 1)  # Go back to the start of the JSON object
                break
            if begin_position >= end_position:
                break

    current_position = tweet_file.tell()
    while current_position < end_position:
        tweet_str = tweet_file.readline()
        current_position += len(tweet_str.encode('utf-8'))

        if tweet_str.strip() in ('', '{}]}\n'):
            break

        try:
            tweet = json.loads(tweet_str.rstrip(',\n'))
            hour, day, sentiment = get_params(tweet)

            if isinstance(sentiment, (float, int)): 
                hour_cnt, hour_sentiment = hour_stats_dict.get(hour, (0, 0.0))
                day_cnt, day_sentiment = day_stats_dict.get(day, (0, 0.0))

                hour_stats_dict[hour] = (hour_cnt + 1, hour_sentiment + sentiment)
                day_stats_dict[day] = (day_cnt + 1, day_sentiment + sentiment)
            else:
                # fail to get sentiment
                hour_stats_dict[hour] = hour_stats_dict.get(hour, (0, 0.0))
                day_stats_dict[day] = day_stats_dict.get(day, (0, 0.0))
        except json.JSONDecodeError:
            # Handle incomplete or malformed JSON
            pass

all_hour_stats = comm.gather(hour_stats_dict, root=0)
all_day_stats = comm.gather(day_stats_dict, root=0)

if rank == 0:
    combined_hour_stats = {}
    combined_day_stats = {}

    for hour_stats in all_hour_stats:
        combined_hour_stats = merge_dicts(combined_hour_stats, hour_stats)

    for day_stats in all_day_stats:
        combined_day_stats = merge_dicts(combined_day_stats, day_stats)

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
