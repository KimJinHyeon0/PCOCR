import torch
import pickle5
from transformers import BertTokenizer, BertModel
import time
import numpy as np

cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer_Bert = BertTokenizer.from_pretrained('bert-base-uncased')
model_Bert = BertModel.from_pretrained("bert-base-uncased").cuda()

def embeddingExtract(subreddit):
    with open('./data/{}_sentence_dict.pickle'.format(subreddit), 'rb') as f:
        sentence_dict = pickle5.load(f)

    dict_l = list(sentence_dict.keys())
    max_idx = max(dict_l)

    print('Total Idx = ', max_idx)
    feature = torch.zeros((max_idx+1, 768), device=cuda)
    cnt = 0
    start = time.time()

    for key in sentence_dict:
        if cnt % 10000 == 0:
            end = time.time()
            print('subreddit = {} | cnt = {}/{} | avg_time = {}'.format(subreddit, cnt, max_idx, (end - start)/10000))
            start = time.time()
        text = sentence_dict[key]
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = tokenizer_Bert.tokenize(marked_text)

        if len(tokenized_text) > 512:
            tokenized_text = tokenized_text[:511]
            tokenized_text.append('[SEP]')

        segments_ids = [key] * len(tokenized_text)
        indexed_tokens = tokenizer_Bert.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens]).cuda()
        segments_tensors = torch.tensor([segments_ids]).cuda()

        with torch.no_grad():
            output_Bert = model_Bert(tokens_tensor, segments_tensors)
        embedding = output_Bert.last_hidden_state
        embedding = torch.squeeze(torch.mean(embedding, dim=1))
        feature[key] = embedding
        cnt += 1
    return feature

subredditList = ['iama', 'showerthoughts', 'news']

for subreddit in subredditList:
    print('\nExtracting Embedding | subreddit : {}\n'.format(subreddit))
    OUT_EDGE_FEAT = './processed/{}_edge_feat.npy'.format(subreddit)
    OUT_NODE_FEAT = './processed/{}_node_feat.npy'.format(subreddit)
    output = embeddingExtract(subreddit).cpu()

    np.save(OUT_NODE_FEAT, output)
    print('\nSaved {}_node_feat.npy'.format(subreddit))

    empty_feat = np.ones_like(output)
    np.save(OUT_EDGE_FEAT, empty_feat)
    print('Saved {}_edge_feat.npy'.format(subreddit))

    del output
    torch.cuda.empty_cache()
    print('-'*50)

print('Done')
