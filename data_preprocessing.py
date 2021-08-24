import pickle
from collections import defaultdict
import pandas as pd

train_post_dict = defaultdict()
test_post_dict = defaultdict()
comment_dict = defaultdict()

train_key_sequence_dict = defaultdict()
test_key_sequence_dict = defaultdict()

train_id_sequence_dict = defaultdict()
test_id_sequence_dict = defaultdict()

sentence_dict = defaultdict()

def makeDict(subreddit):

    train_post_dict.clear()
    test_post_dict.clear()
    comment_dict.clear()
    sentence_dict.clear()

    train_key_sequence_dict.clear()
    test_key_sequence_dict.clear()

    train_id_sequence_dict.clear()
    test_id_sequence_dict.clear()

    #post_train_dict
    with open('./data/raw_data/posts_' + subreddit + '_train.csv', 'r', encoding='utf-8') as f:
        post_train = f.read().rstrip("\n;\n").split("\n;\n")

    for i in range(len(post_train)):
        data = post_train[i].split('\t;\t')
        post_id = int(data[0])
        post_key = data[1]
        title = data[4]
        selftext = data[7]

        train_post_dict[post_key] = data
        train_key_sequence_dict.update({post_key:[]})
        train_id_sequence_dict.update({post_id:[]})

        body = title + ' ' + selftext
        sentence_dict.update({post_id:body})

    #post_test_dict
    with open('./data/raw_data/posts_' + subreddit + '_test.csv', 'r', encoding='utf-8') as f:
        post_test = f.read().rstrip("\n;\n").split("\n;\n")

    for i in range(len(post_test)):
        data = post_test[i].split('\t;\t')
        post_id = int(data[0])
        post_key = data[1]
        title = data[4]
        selftext = data[7]

        test_post_dict[post_key] = data
        test_key_sequence_dict.update({post_key:[]})
        test_id_sequence_dict.update({post_id: []})

        body = title + ' ' + selftext
        sentence_dict.update({post_id:body})

    #comment_dict
    with open('./data/raw_data/comments_' + subreddit + '.csv', 'r', encoding='utf-8') as f:
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
        if link_key in train_key_sequence_dict:
            train_key_sequence_dict[link_key] += [(comment_key, parent_key, time_stamp)]

        if link_key in test_key_sequence_dict:
            test_key_sequence_dict[link_key] += [(comment_key, parent_key, time_stamp)]

    #sort key_sequence_dict by time_stamp
    for i in list(train_key_sequence_dict.keys()):
        train_key_sequence_dict[i].sort(key=lambda x: (x[2]))

    for i in list(test_key_sequence_dict.keys()):
        test_key_sequence_dict[i].sort(key=lambda x: (x[2]))


def makeTrainDataframe(subreddit):
    u_list, i_list, ts_list, label_list, idx_list = [], [], [], [], []

    idx = 1

    for post_key in list(train_key_sequence_dict.keys()):
        sequence = train_key_sequence_dict[post_key]

        for j in range(len(sequence)):
            comment_key = sequence[j][0]
            parent_key = sequence[j][1]
            time_stamp = sequence[j][2]

            comment_id = int(comment_dict[comment_key][0])

            link_key = comment_dict[comment_key][4]

            if parent_key in train_post_dict:
                parent_id = int(train_post_dict[parent_key][0])

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

            link_id = int(train_post_dict[link_key][0])
            train_id_sequence_dict[link_id] += [(comment_id, parent_id, time_stamp, label, idx)]

            idx += 1

    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': idx_list})

def makeTestDataframe(subreddit):
    u_list, i_list, ts_list, label_list, idx_list = [], [], [], [], []

    idx = 1

    for post_key in list(test_key_sequence_dict.keys()):

        sequence = test_key_sequence_dict[post_key]
        for j in range(len(sequence)):

            comment_key = sequence[j][0]
            parent_key = sequence[j][1]
            time_stamp = sequence[j][2]

            comment_id = int(comment_dict[comment_key][0])

            link_key = comment_dict[comment_key][4]

            if parent_key in test_post_dict:
                parent_id = int(test_post_dict[parent_key][0])

            elif parent_key in comment_dict:
                parent_id = int(comment_dict[parent_key][0])

            else:
                continue

            if j == len(sequence)-1:
                label = 0
            else:
                label = 1

            u_list.append(comment_id)
            i_list.append(parent_id)
            ts_list.append(time_stamp)
            label_list.append(label)
            idx_list.append(idx)

            link_id = int(test_post_dict[link_key][0])

            test_id_sequence_dict[link_id] += [(comment_id, parent_id, time_stamp, label, idx)]

            idx += 1

    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': idx_list})

def save(train_df, test_df, subreddit):

    train_df.to_csv(OUT_CSV_train)
    print('\nSaved {}_train.csv'.format(subreddit))
    test_df.to_csv(OUT_CSV_test)
    print('\nSaved {}_test.csv'.format(subreddit))

    with open(OUT_SENTENCE_DICT, 'wb') as f:
        pickle.dump(sentence_dict, f, pickle.HIGHEST_PROTOCOL)
    print('\nSaved {}_sentence_dict.pickle'.format(subreddit))

    with open(OUT_SEQUENCE_DICT_TRAIN, 'wb') as f:
        pickle.dump(train_id_sequence_dict, f, pickle.HIGHEST_PROTOCOL)
    print('\nSaved {}_sequence_dict_train.pickle'.format(subreddit))

    with open(OUT_SEQUENCE_DICT_TEST.format(subreddit), 'wb') as f:
        pickle.dump(test_id_sequence_dict, f, pickle.HIGHEST_PROTOCOL)
    print('\nSaved {}_sequence_dict_test.pickle'.format(subreddit))

subredditlist = ['news', 'iama', 'showerthoughts']
for subreddit in subredditlist:
    print('\nProcessing subreddit - {}...\n'.format(subreddit))

    OUT_CSV_train = './processed/{}_train.csv'.format(subreddit)
    OUT_CSV_test = './processed/{}_test.csv'.format(subreddit)
    OUT_SENTENCE_DICT = './data/{}_sentence_dict.pickle'.format(subreddit)
    OUT_SEQUENCE_DICT_TRAIN = './processed/{}_seq_dict_train.pickle'.format(subreddit)
    OUT_SEQUENCE_DICT_TEST = './processed/{}_seq_dict_test.pickle'.format(subreddit)

    makeDict(subreddit)
    train_df = makeTrainDataframe(subreddit)
    test_df = makeTestDataframe(subreddit)

    save(train_df, test_df, subreddit)
    print('-' * 50)

print('\nDone')
