import torch
import numpy as np
import pandas as pd
import time

from torchtext.data import Field
from torchtext.vocab import GloVe, FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from transformers import BertTokenizer, BertModel
from transformers import RobertaModel, RobertaTokenizer
from transformers import DebertaModel, DebertaTokenizer

cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class get_word_embedding():
    def __init__(self, config):
        self.SUBREDDIT = config['subreddit']
        self.DEFAULT_TOKENIZER = False

        self.PRETRAINED_MODEL = config['pretrained_model']
        self.TEXT_CLEANING = config['text_cleaning']
        self.SAVE = config['save']

        self.CORPUS = pd.read_csv(f'./data/{self.SUBREDDIT}_sentence.csv', index_col=0, na_filter=False)

        if self.PRETRAINED_MODEL == 'bert-base-uncased':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased')

        elif self.PRETRAINED_MODEL == 'roberta-base':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.model = RobertaModel.from_pretrained('roberta-base')

        elif self.PRETRAINED_MODEL == 'deberta-base':
            self.tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
            self.model = DebertaModel.from_pretrained('microsoft/deberta-base')

        elif self.PRETRAINED_MODEL == 'fasttext':
            self.DEFAULT_TOKENIZER = True
            self.model = FastText(language='en')

        elif self.PRETRAINED_MODEL == 'glove':
            self.DEFAULT_TOKENIZER = True
            self.model = GloVe(name='840B', dim=300)

        elif self.PRETRAINED_MODEL == 'tf-idf':
            self.DEFAULT_TOKENIZER = True

        else:
            raise ValueError(
                "Could not find pretrained model \nAvailable : bert-base-uncased / fasttext.en.300d / glove.840B.300d / tf-idf / roberta_base / deberta_large")

        if self.DEFAULT_TOKENIZER:
            self.tokenizer = Field(tokenize='basic_english', lower=True)
            self.OUTPUT_DIMENSION = 300
        else:
            self.OUTPUT_DIMENSION = 768

        self.FEATURE = torch.zeros((self.CORPUS.index.max() + 1, self.OUTPUT_DIMENSION), device=cuda)


    def text_cleaning(self, text):
        from bs4 import BeautifulSoup
        import re

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
        # text = emoji_pattern.sub(r'', text)
        # text = re.sub(r"[^a-zA-Z\d]", " ", text)  # Remove special Charecters
        # text = re.sub(' +', ' ', text)  # Remove Extra Spaces
        text = text.strip()  # remove spaces at the beginning and at the end of string

        return text

    def run_text_cleaning(self):
        print(f'Processing Text Cleaning ...', end=' ')
        self.CORPUS = self.CORPUS['raw_text'].apply(lambda x: self.text_cleaning(x))
        print('Done')

    def save(self):
        OUT_NODE_FEAT = f'./processed/{self.SUBREDDIT}_node_feat_{self.PRETRAINED_MODEL}.npy'
        OUT_EDGE_FEAT = f'./processed/{self.SUBREDDIT}_edge_feat_{self.PRETRAINED_MODEL}.npy'

        self.FEATURE = self.FEATURE.cpu()
        np.save(OUT_NODE_FEAT, self.FEATURE)
        print(f'\nSaved {self.SUBREDDIT}_node_feat_{self.PRETRAINED_MODEL}.npy')

        np.save(OUT_EDGE_FEAT, np.ones_like(self.FEATURE))
        print(f'Saved {self.SUBREDDIT}_edge_feat_{self.PRETRAINED_MODEL}.npy')

    def run(self):
        num_instance = len(self.CORPUS)
        print(f'SUBREDDIT = {self.SUBREDDIT}')
        print(f'PRETRAINED_MODEL = {self.PRETRAINED_MODEL}')
        print(f'NUM_INSTANCE = {num_instance}')

        if self.TEXT_CLEANING:
            self.run_text_cleaning()

        if not self.DEFAULT_TOKENIZER:
            self.model.to(cuda)
        start = time.time()
        for i, (index, text) in enumerate(self.CORPUS.items()):
            if i % 1000 == 0:
                print(f'{self.PRETRAINED_MODEL} | {self.SUBREDDIT} | {i}/{num_instance} ... {round(i / num_instance, 3)}% | avg_time = {round((time.time() - start) / 1000, 3)}s')
                start = time.time()

            if not self.DEFAULT_TOKENIZER:
                tokenized = self.tokenizer(text, return_tensors="pt", truncation=True)
                input_ids = tokenized['input_ids'].cuda()
                attention_mask = tokenized['attention_mask'].cuda()
                if self.PRETRAINED_MODEL == 'roberta-base':
                    token_type_ids = torch.zeros_like(input_ids, device=cuda)
                else:
                    token_type_ids = tokenized['token_type_ids'].cuda()

                with torch.no_grad():
                    outputs = self.model(input_ids, attention_mask, token_type_ids)

                last_hidden_states = outputs.last_hidden_state
                embedding = torch.squeeze(torch.mean(last_hidden_states, dim=1))
                self.FEATURE[index] = embedding

            else:
                output = self.model.get_vecs_by_tokens(text)
                embedding = torch.mean(output, dim=0)
                self.FEATURE[index] = embedding

        if self.SAVE:
            self.save()

'''
=====CONFIGS=====

        subreddit : 'iama'
                    'showerthoughts'
        
        pretrained_model : 'bert-base-uncased'  
                           'roberta-base'
                           'deberta-base'
                           'fasttext'
                           'glove' 
                           'tf-idf' - not supported yet
        
        text_cleaning : bool
        
        save : bool
        
=====CONFIGS=====
'''

CONFIGS = {
    'subreddit' : 'iama',
    'pretrained_model' : 'glove',
    'text_cleaning' : True,
    'save' : True
}

model = get_word_embedding(CONFIGS)
model.run()
