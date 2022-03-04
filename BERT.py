import torch
import pickle5
from transformers import BertTokenizer, BertModel
import time
import numpy as np
import re
from bs4 import BeautifulSoup

cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.to(cuda)


def text_cleaning(text):
    '''
    Cleans text into a basic form for NLP. Operations include the following:-
    1. Remove special charecters like &, #, etc
    2. Removes extra spaces
    3. Removes embedded URL links
    4. Removes HTML tags
    5. Removes emojis

    text - Text piece to be cleaned.
    '''
    template = re.compile(r'https?://\S+|www\.\S+')  # Removes website links
    text = template.sub(r'', text)

    soup = BeautifulSoup(text, 'lxml')  # Removes HTML tags
    only_text = soup.get_text()
    text = only_text

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    text = re.sub(r"[^a-zA-Z\d]", " ", text)  # Remove special Charecters
    text = re.sub(' +', ' ', text)  # Remove Extra Spaces
    text = text.strip()  # remove spaces at the beginning and at the end of string

    return text


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
        raw_text = sentence_dict[key]
        cleaned_text = text_cleaning(raw_text)
        inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True).to(cuda)
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        embedding = torch.squeeze(torch.mean(last_hidden_states, dim=1))
        feature[key] = embedding

    return feature


subredditList = ['news', 'iama', 'showerthoughts']

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
    print('-' * 50)

print('Done')
