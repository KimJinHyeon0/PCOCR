import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
import time
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
    #text = emoji_pattern.sub(r'', text)

    text = re.sub(r"[^a-zA-Z\d]", " ", text)  # Remove special Charecters
    text = re.sub(' +', ' ', text)  # Remove Extra Spaces
    text = text.strip()  # remove spaces at the beginning and at the end of string

    return text


def embeddingExtract(subreddit):
    sentence_df = pd.read_csv(SENTENCE_DF_PATH, index_col=0, na_filter=False)
    max_idx = sentence_df.index.max()

    print(f'Total Idx = {max_idx}')
    feature = torch.zeros((max_idx + 1, 768), device='cpu')

    print(f'Processing Text Cleaning...')
    cleaned_text = sentence_df['raw_text'].apply(lambda x: text_cleaning(x))

    num_instance = len(cleaned_text)
    print(f'NUM_INSTANCE = {num_instance}')
    start = time.time()
    for i, (index, text) in enumerate(cleaned_text.items()):
        if i % 1000 == 0:
            print(f'{subreddit} | {i}/{num_instance} ... {round(i/num_instance, 3)}% | avg_time = {round((time.time()-start)/1000, 3)}s')
            start = time.time()

        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(cuda)
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        embedding = torch.squeeze(torch.mean(last_hidden_states, dim=1))
        feature[index] = embedding

    return feature

for subreddit in ['iama', 'showerthoughts']:
    PRETRAINED_MODEL = 'BERT'
    print(f'PRETRAINED_MODEL = {PRETRAINED_MODEL}')
    print(f'\nExtracting Embedding | subreddit : {subreddit}\n')
    SENTENCE_DF_PATH = f'./data/{subreddit}_sentence.csv'
    OUT_EDGE_FEAT = f'./processed/{subreddit}_edge_feat_{PRETRAINED_MODEL}.npy'
    OUT_NODE_FEAT = f'./processed/{subreddit}_node_feat_{PRETRAINED_MODEL}.npy'
    output = embeddingExtract(subreddit).cpu()

    np.save(OUT_NODE_FEAT, output)
    print(f'\nSaved {subreddit}_node_feat_{PRETRAINED_MODEL}.npy')

    empty_feat = np.ones_like(output)
    np.save(OUT_EDGE_FEAT, empty_feat)
    print(f'Saved {subreddit}_edge_feat_{PRETRAINED_MODEL}.npy')

    del output
    torch.cuda.empty_cache()
    print('-' * 50)

print('Done')
