import pickle
from collections import defaultdict
import pandas as pd

post_dict = defaultdict()
comment_dict = defaultdict()

key_sequence_dict = defaultdict()
id_sequence_dict = defaultdict()

sentence_dict = defaultdict()

def makeDict(subreddit):
    POSTPATH = './data/raw_data/posts_{}.csv'.format(subreddit)
    COMMENTPATH = './data/raw_data/comments_{}.csv'.format(subreddit)

    post_dict.clear()
    comment_dict.clear()

    key_sequence_dict.clear()
    id_sequence_dict.clear()

    sentence_dict.clear()

    #post_dict
    with open(POSTPATH, 'r', encoding='utf-8') as f:
        post_train = f.read().rstrip("\n;\n").split("\n;\n")

    for i in range(len(post_train)):
        data = post_train[i].split('\t;\t')
        post_id = int(data[0])
        post_key = data[1]
        title = data[4]
        selftext = data[7]

        post_dict[post_key] = data
        key_sequence_dict.update({post_key:[]})
        id_sequence_dict.update({post_id:[]})

        body = title + ' ' + selftext
        sentence_dict.update({post_id:body})

    #comment_dict
    with open(COMMENTPATH, 'r', encoding='utf-8') as f:
        comment = f.read().rstrip("\n;\n").split("\n;\n")

    for i in range(len(comment)):
        data = comment[i].split('\t;\t')
        comment_id = int(data[0])
        comment_key = data[1]
        link_key = data[4]
        parent_key = data[5]
        body = data[6]

        comment_dict[comment_key] = data
        sentence_dict.update({comment_id:body})

        if subreddit == 'news':
            time_stamp = int(data[8])
        else:
            time_stamp = int(data[7])

    #key_sequence_dict
        if link_key in key_sequence_dict:
            key_sequence_dict[link_key] += [(comment_key, parent_key, time_stamp)]

    #sort key_sequence_dict by time_stamp
    for i in list(key_sequence_dict.keys()):
        key_sequence_dict[i].sort(key=lambda x: (x[2]))

def makeDf(subreddit):
    u_list, i_list, ts_list, label_list, idx_list = [], [], [], [], []

    idx = 1

    for post_key in list(key_sequence_dict.keys()):
        sequence = key_sequence_dict[post_key]

        for j in range(len(sequence)):
            comment_key = sequence[j][0]
            parent_key = sequence[j][1]
            time_stamp = sequence[j][2]

            comment_id = int(comment_dict[comment_key][0])

            link_key = comment_dict[comment_key][4]

            if parent_key in post_dict:
                parent_id = int(post_dict[parent_key][0])

            elif parent_key in comment_dict:
                parent_id = int(comment_dict[parent_key][0])

            else:
                continue

            u = comment_id
            i = parent_id

            if j == len(sequence)-1:
                label = 0
            else:
                label = 1

            u_list.append(u)
            i_list.append(i)
            ts_list.append(time_stamp)
            label_list.append(label)
            idx_list.append(idx)

            link_id = int(post_dict[link_key][0])
            id_sequence_dict[link_id] += [(comment_id, parent_id, time_stamp, label, idx)]
            idx += 1

    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': idx_list})

def save(df, subreddit):

    df.to_csv(OUT_CSV_train)
    print('\nSaved ml_{}.csv'.format(subreddit))

    with open(OUT_SENTENCE_DICT, 'wb') as f:
        pickle.dump(sentence_dict, f, pickle.HIGHEST_PROTOCOL)
    print('\nSaved {}_sentence_dict.pickle'.format(subreddit))

    # with open(OUT_SEQUENCE_DICT, 'wb') as f:
    #     pickle.dump(id_sequence_dict, f, pickle.HIGHEST_PROTOCOL)
    # print('\nSaved {}_sequence_dict.pickle'.format(subreddit))


subredditlist = ['news', 'iama', 'showerthoughts']
for subreddit in subredditlist:
    print('\nProcessing subreddit - {}...\n'.format(subreddit))

    OUT_CSV_train = './processed/ml_{}.csv'.format(subreddit)
    OUT_SENTENCE_DICT = './data/{}_sentence_dict.pickle'.format(subreddit)
    OUT_SEQUENCE_DICT = './processed/{}_seq_dict.pickle'.format(subreddit)

    makeDict(subreddit)
    df = makeDf(subreddit)
    save(df, subreddit)
    print('-' * 50)

print('\nDone')
