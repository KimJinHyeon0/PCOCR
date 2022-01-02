import torch
import pickle5
from transformers import BertTokenizer, BertModel
import time
import numpy as np

cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.to(cuda)


def embeddingExtract(subreddit):
    with open('./data/{}_sentence_dict.pickle'.format(subreddit), 'rb') as f:
        sentence_dict = pickle5.load(f)

    dict_l = list(sentence_dict.keys())
    max_idx = max(dict_l)

    print('Total Idx = ', max_idx)
    feature = torch.zeros((max_idx + 1, 768), device='cpu')

    for key in sentence_dict:
        if int(key) % 1000 == 0:
            print('{} | {}/{}'.format(subreddit, key, max_idx))
        text = sentence_dict[key]
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(cuda)
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        embedding = torch.squeeze(torch.mean(last_hidden_states, dim=1))
        feature[key] = embedding

    return feature


subredditList = ['news', 'iama', 'showerthoughts']

for subreddit in subredditList:
    print('\nExtracting Embedding | subreddit : {}\n'.format(subreddit))
    OUT_EDGE_FEAT = './processed/{}_edge_feat_2.npy'.format(subreddit)
    OUT_NODE_FEAT = './processed/{}_node_feat_2.npy'.format(subreddit)
    output = embeddingExtract(subreddit).cpu()

    np.save(OUT_NODE_FEAT, output)
    print('\nSaved {}_node_feat.npy'.format(subreddit))

    empty_feat = np.ones_like(output)
    np.save(OUT_EDGE_FEAT, empty_feat)
    print('Saved {}_edge_feat.npy'.format(subreddit))

    del output
    torch.cuda.empty_cache()
    print('-' * 50)

print('Done')
