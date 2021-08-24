import torch
import pickle
from transformers import BertTokenizer, BertModel
import time
import numpy as np


cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer_Bert = BertTokenizer.from_pretrained('bert-base-uncased')
model_Bert = BertModel.from_pretrained("bert-base-uncased").cuda()

def embeddingExtract(subreddit, case):
    with open('./data/{}_sentence_dict.pickle'.format(subreddit), 'rb') as f:
        sentence_dict = pickle.load(f)

    dict_l = list(sentence_dict.keys())
    max_idx = dict_l[-1]
    half_idx = int(max_idx / 2)

    print('Total Idx = ', max_idx)
    if case == 0:
        print('Processing fore half idx ({} ~ {})'.format('1', half_idx))
        feature = torch.zeros((half_idx, 1, 768), device=cuda)
        cnt = 0
        start = time.time()

        for key in sentence_dict:
            if key < half_idx:
                if cnt % 10000 == 0:
                    end = time.time()
                    print('cnt = ', cnt, '\tavg_time = ', (end - start)/10000)
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

                embedding = torch.mean(embedding, dim=1)
                feature[key] = embedding
                cnt += 1

    if case == 1:
        print('Processing left half idx ({} ~ {})'.format(half_idx+1, max_idx))
        feature = torch.zeros((half_idx+1, 1, 768), device=cuda)
        cnt = 0
        start = time.time()

        for key in sentence_dict:
            if key >= half_idx:
                if cnt % 10000 == 0:
                    end = time.time()
                    print('cnt = ', cnt, '\tavg_time = ', (end - start) / 10000)
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

                embedding = torch.mean(embedding, dim=1)
                feature[key-half_idx] = embedding
                cnt += 1

    return feature

subredditList = ['news', 'iama', 'showerthoughts']

for subreddit in subredditList:
    print('\nExtracting Embedding | subreddit : {}\n'.format(subreddit))
    OUT_EDGE_FEAT = './processed/{}_edge_feat.npy'.format(subreddit)
    OUT_NODE_FEAT = './processed/{}.node_feat.npy'.format(subreddit)
    for case in [0, 1]:
        output = embeddingExtract(subreddit, case).cpu()
        if case == 0:
            output_0 = output.numpy()
        if case == 1:
            output_1 = output.numpy()

        del output
        torch.cuda.empty_cache()

    tot = np.concatenate((output_0, output_1), axis=0)
    np.save(OUT_NODE_FEAT, tot)
    print('\nSaved {}_node_feat.npy'.format(subreddit))

    empty_feat = np.ones((tot.shape[0], tot.shape[2]))
    np.save(OUT_EDGE_FEAT, empty_feat)
    print('Saved {}_node_feat.npy'.format(subreddit))
    print('-'*50)

print('Done')
