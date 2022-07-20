import torch
import numpy as np
import pandas as pd
import time

from torchtext.data import Field
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
        self.OUTPUT_DIMENSION = config['output_dimension']
        self.TEXT_CLEANING = config['text_cleaning']
        self.SAVE = config['save']

        self.CORPUS = pd.read_csv(f'./data/{self.SUBREDDIT}_sentence.csv', index_col=0, na_filter=False)


        if self.PRETRAINED_MODEL == 'bert-base-uncased':
            self.tokenizer = BertTokenizer.from_pretrained(self.PRETRAINED_MODEL)
            self.model = BertModel.from_pretrained(self.PRETRAINED_MODEL)
            self.FEATURE = torch.zeros((self.CORPUS.index.max() + 1, 768), device=cuda)

        elif self.PRETRAINED_MODEL == 'roberta-base':
            self.tokenizer = RobertaTokenizer.from_pretrained(self.PRETRAINED_MODEL)
            self.model = RobertaModel.from_pretrained(self.PRETRAINED_MODEL)
            self.FEATURE = torch.zeros((self.CORPUS.index.max() + 1, 768), device=cuda)

        elif self.PRETRAINED_MODEL == 'microsoft/deberta-base':
            self.tokenizer = DebertaTokenizer.from_pretrained(self.PRETRAINED_MODEL)
            self.model = DebertaModel.from_pretrained(self.PRETRAINED_MODEL)
            self.FEATURE = torch.zeros((self.CORPUS.index.max() + 1, 768), device=cuda)

        elif self.PRETRAINED_MODEL == 'fasttext.en.300d' or\
             self.PRETRAINED_MODEL == 'glove.840B.300d' or\
             self.PRETRAINED_MODEL == 'tf-idf':
            self.DEFAULT_TOKENIZER = True
            self.tokenizer = Field(tokenize='basic_english', lower=True)
            self.FEATURE = torch.zeros((self.CORPUS.index.max() + 1, self.OUTPUT_DIMENSION), device=cuda)

        else:
            raise ValueError(
                "Could not find pretrained model \nAvailable : bert-base-uncased / fasttext.en.300d / glove.840B.300d / tf-idf / roberta_base / deberta_large")

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
                self.tokenizer.build_vocab([text], vectors=self.PRETRAINED_MODEL)
                vocab = self.tokenizer.vocab
                vectors = vocab.vectors.to(cuda)

                one_tensor = torch.ones((vectors.shape[0], 1), device=cuda)
                for token, cnt in vocab.freqs.items():
                    one_tensor[vocab.stoi[token]] = cnt
                output = vectors.mul(one_tensor)
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
                          'microsoft/deberta-base'
                          'fasttext.en.300d'
                          'glove.840B.300d' 
                          'tf-idf' - not supported
                           
        output_dimension : 'bert-base-uncased' -  768(fixed, no need to configure)
                           'roberta_base' -  768(fixed, no need to configure)
                           'deberta_base' -  768(fixed, no need to configure)
        
        text_cleaning : bool
        
        save : bool
        
=====CONFIGS=====
'''

CONFIGS = {
    'subreddit' : 'iama',
    'pretrained_model' : 'roberta-base',
    'output_dimension' : 300,
    'text_cleaning' : True,
    'save' : False
}

model = get_word_embedding(CONFIGS)
model.run()
