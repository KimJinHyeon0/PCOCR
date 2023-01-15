import pandas as pd
import numpy as np

subredditlist = ['showerthoughts']
for subreddit in subredditlist:
    print('\nProcessing subreddit - {}...\n'.format(subreddit))

    POSTPATH = './data/{}_posts.csv'.format(subreddit)
    COMMENTPATH = './data/{}_comments.csv'.format(subreddit)
    OUT_STRUCTURE_PATH = './processed/{}_structure.csv'.format(subreddit)
    OUT_SENTENCE_PATH = './data/{}_sentence.csv'.format(subreddit)

    with open(POSTPATH, 'r', encoding='utf-8') as f:
        post_train = f.read().rstrip("\n;\n").split("\n;\n")

    total_key_set = []
    total_sentence_dict = {}
    node_map = {}

    g_list, g_ts_list, u_list, i_list, ts_list = [], [], [], [], []

    for i in range(len(post_train)):
        data = post_train[i].split('\t;\t')
        post_id = int(data[0])
        post_key = data[1]
        title = data[4]
        selftext = data[7]
        created_utc = int(data[8])
        body = title + ' ' + selftext

        node_map[post_key] = (post_id, created_utc)
        total_key_set.append(post_key)
        total_sentence_dict[post_id] = body

    with open(COMMENTPATH, 'r', encoding='utf-8') as f:
        comment = f.read().rstrip("\n;\n").split("\n;\n")
    for i in range(len(comment)):
        data = comment[i].split('\t;\t')
        comment_id = int(data[0])
        comment_key = data[1]
        link_key = data[4]
        parent_key = data[5]
        body = data[6]
        time_stamp = int(data[7])

        if link_key in node_map:
            node_map[comment_key] = (comment_id, time_stamp)
            total_key_set.append(comment_key)
            total_sentence_dict[comment_id] = body

            g_list.append(link_key)
            g_ts_list.append(node_map[link_key][1])
            u_list.append(comment_key)
            i_list.append(parent_key)
            ts_list.append(time_stamp - node_map[link_key][1])



    assert len(total_key_set) == len(set(total_key_set))
    sentence_df = pd.DataFrame.from_dict(total_sentence_dict, columns=['raw_text'], orient='index')
    df = pd.DataFrame({'g_num': g_list,
                       'g_ts': g_ts_list,
                       'u': u_list,
                       'i': i_list,
                       'ts': ts_list})

    # Remove nodes disconnected from posts
    print('Removing nodes disconnected from posts...')
    temp = 0
    rn = 0
    while temp != len(node_map):
        temp = len(node_map)
        useless_nodes, useless_indices = np.array([]), np.array([])
        idx_l = df.index.values
        src_l = df.u.values
        dst_l = df.i.values

        for k in set(dst_l):
            if k not in node_map:
                non_flag = dst_l == k
                useless_nodes = np.append(useless_nodes, src_l[non_flag])
                useless_indices = np.append(useless_indices, idx_l[non_flag])

        assert len(useless_nodes) == len(useless_indices)
        rn += len(useless_nodes)

        for i in useless_nodes:
            del node_map[i]

        df = df.drop(useless_indices)
    print('done.')
    print(f'Removed {rn} useless nodes')

    # Remove graphs where #nodes is under 2
    print('Removing graphs where #nodes is under 2...')
    g_num = df.g_num.values
    before_filt = len(g_num)
    isin_filter = df['g_num'].isin(df['g_num'].value_counts()[df['g_num'].value_counts() > 1].index)

    df = df[isin_filter]
    after_filt = len(df)
    print(f'Removed {before_filt - after_filt} useless graphs')
    print('done.')

    df['g_num'] = df.g_num.apply(lambda x: node_map[x][0])
    df['u'] = df.u.apply(lambda x: node_map[x][0])
    df['i'] = df.i.apply(lambda x: node_map[x][0])
    dst_l = df.i.values
    df['label'] = df.u.apply(lambda x: 1 if x in dst_l else 0)
    df['idx'] = df['u']

    # Remove tangled graph
    if subreddit == 'showerthoughts':
        print('Removing Tangled graphs...')
        tangled_g_num = [9630, 18352, 51637, 59047, 84858]
        df = df.drop(df[df['g_num'].isin(tangled_g_num)].index)
        print('done.')

    df.to_csv(OUT_STRUCTURE_PATH)
    print('\nSaved {}_structure.csv'.format(subreddit))
    total_node_set = np.sort(np.unique(np.hstack((df.g_num.values, df.u.values, df.i.values))))
    sentence_df = sentence_df.loc[total_node_set]
    sentence_df.to_csv(OUT_SENTENCE_PATH)
    print('\nSaved {}_sentence.csv'.format(subreddit))

print('\nDone.')