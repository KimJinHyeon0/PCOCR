import torch
import numpy as np
import pandas as pd

cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def embeddingExtract(subreddit):
    sentence_df = pd.read_csv(SENTENCE_DF_PATH, index_col=0)
    max_idx = sentence_df.index.max()

    print('Total Idx = ', max_idx)
    feature = torch.ones((max_idx + 1, 768), device='cpu')


    return feature



for subreddit in ['iama', 'showerthoughts']:
    print('\nExtracting Embedding | subreddit : {}\n'.format(subreddit))
    SENTENCE_DF_PATH = './data/{}_sentence.csv'.format(subreddit)
    OUT_EDGE_FEAT = './processed/{}_np_ones_edge_feat.npy'.format(subreddit)
    OUT_NODE_FEAT = './processed/{}_np_ones_node_feat.npy'.format(subreddit)
    output = embeddingExtract(subreddit).cpu()

    np.save(OUT_NODE_FEAT, output)
    print('\nSaved {}_np_ones_node_feat.npy'.format(subreddit))

    empty_feat = np.ones_like(output)
    np.save(OUT_EDGE_FEAT, empty_feat)
    print('Saved {}_np_ones_edge_feat.npy'.format(subreddit))

    del output
    torch.cuda.empty_cache()
    print('-' * 50)

print('Done')
