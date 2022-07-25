# -*- coding: utf-8 -*-

# Yu Qin
# 07232022

### This file contains utility function that generate sentence embeddings for SEC 10-K and 10-Q reports.

import sys

from importlib_metadata import metadata
sys.path.append(r'src/utils/')

import numpy as np
import pandas as pd
import os
import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import euclidean_distances

from Parse10K import Parse_10k
from Parse10Q import Parse_10q

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(filename='logs/SecEmbedding.log', level=logging.INFO, format='%(asctime)s %(message)s')

class Main_Dataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings.to('cuda')

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

def bert_embedding(model, tokenizer, data_dict):

    embedding_dict = {}

    for item in data_dict:
        embedding_dict[item] = []
        tokenized_dataset = Main_Dataset(tokenizer(data_dict[item], truncation=True, padding='max_length', max_length=512, return_tensors='pt'))
        tokenized_dataloader = DataLoader(tokenized_dataset, batch_size=16, shuffle=False)
        for batch in tokenized_dataloader:
            with torch.no_grad():
                embedding = model(**batch)
                embedding_dict[item].extend(embedding.pooler_output.clone().cpu().detach().tolist())

    return embedding_dict

if __name__ == '__main__':

    logging.info('Using Device: %s', torch.cuda.get_device_name(0))

    model_type = "roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModel.from_pretrained(model_type)
    model.cuda()

    metadata_10k = pd.read_csv('data/metadata/10K_list.csv')
    metadata_10k['embedding']  = 0
    metadata_10q = pd.read_csv('data/metadata/10Q_list.csv')
    metadata_10q['embedding']  = 0

    file_num = len(os.listdir('data/sec-edgar-filings/reports/'))
    logging.info('Total number of files to be processed: %s', file_num)

    countn = 0
    for comp in os.listdir('data/sec-edgar-filings/reports/'):

        logging.info('Processing company %s', comp)

        os.makedirs('data/sec-edgar-filings/embeddings/' + comp, exist_ok=True)
        os.makedirs('data/sec-edgar-filings/embeddings/' + comp + '/10-K', exist_ok=True)
        os.makedirs('data/sec-edgar-filings/embeddings/' + comp + '/10-Q', exist_ok=True)

        logging.info('Processing 10-K reports for company %s', comp)
        # 10-K
        for file in os.listdir(f'data/sec-edgar-filings/reports/{comp}/10-K/'):
            file_path = os.path.join(f'data/sec-edgar-filings/reports/{comp}/10-K/', file, 'full-submission.txt')
            report_index = metadata_10k[metadata_10k['accessNumber'] == file].index.values[0]

            try:
                data_dict = Parse_10k(file_path)
                embedding_dict = bert_embedding(model, tokenizer, data_dict)
                torch.cuda.empty_cache()

                with open(f'data/sec-edgar-filings/embeddings/{comp}/10-K/{file}.json', 'w') as f:
                    json.dump(embedding_dict, f)
                
                logging.info('Processed file: %s', file)
                metadata_10k['embedding'][report_index] = 1
            
            except:
                logging.info('Error in file: %s', file)
                metadata_10k['embedding'][report_index] = -1
        
        metadata_10k.to_csv('data/metadata/10K_list_embedding.csv', index=False)

        logging.info('Processing 10-Q reports for company %s', comp)
        # 10-Q
        for file in os.listdir(f'data/sec-edgar-filings/reports/{comp}/10-Q/'):
            file_path = os.path.join(f'data/sec-edgar-filings/reports/{comp}/10-Q/', file, 'full-submission.txt')
            report_index = metadata_10q[metadata_10q['accessNumber'] == file].index.values[0]
            
            try:
                data_dict = Parse_10q(file_path)
                embedding_dict = bert_embedding(model, tokenizer, data_dict)
                torch.cuda.empty_cache()

                with open(f'data/sec-edgar-filings/embeddings/{comp}/10-Q/{file}.json', 'w') as f:
                    json.dump(embedding_dict, f)
                
                logging.info('Processed file: %s', file)
                metadata_10q['embedding'][report_index] = 1
            
            except:
                logging.info('Error in file: %s', file)
                metadata_10q['embedding'][report_index] = -1
        
        metadata_10q.to_csv('data/metadata/10Q_list_embedding.csv', index=False)
        
        countn += 1
        logging.info('Processing progress: %s %s', countn / file_num * 100, '%')