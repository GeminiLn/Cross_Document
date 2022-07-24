# -*- coding: utf-8 -*-

# Yu Qin
# 07232022

### This file contains utility function that generate sentence embeddings for earnings call transcripts.

import pandas as pd
import numpy as np
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(filename='logs/EarningsEmbedding.log', level=logging.INFO, format='%(asctime)s %(message)s')

class Main_Dataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings.to('cuda')

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

def bert_embedding(model, tokenizer, data):

    embedding_matrix = []
    tokenized_dataset = Main_Dataset(tokenizer(data, truncation=True, padding='max_length', max_length=512, return_tensors='pt'))
    tokenized_dataloader = DataLoader(tokenized_dataset, batch_size=16, shuffle=False)
    for batch in tokenized_dataloader:
        with torch.no_grad():
            embedding = model(**batch)
            embedding_matrix.extend(embedding.pooler_output.clone().cpu().detach().tolist())

    return embedding_matrix

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    logging.info('Using Device: %s', torch.cuda.get_device_name(0))

    model_type = "roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModel.from_pretrained(model_type)
    model.to('cuda')

    metadata = pd.read_csv('data/metadata/earningscall_list.csv')
    metadata['embedding']  = 0

    file_num = len(os.listdir('data/earningscall/transcripts/'))
    logging.info('Total number of files to be processed: %s', file_num)

    countn = 0
    for file in os.listdir('data/earningscall/transcripts/'):
        data = pd.read_feather('data/earningscall/transcripts/' + file)
        transcriptid = int(file.split('.')[0])
        transcript_index = metadata[metadata['transcriptid'] == transcriptid].index.values[0]
        
        embedding_dict = {}
        try:
            for i in range(data.shape[0]):
                sentence_list = data['componenttext'][i].split('. ')
                embedding = bert_embedding(model, tokenizer, sentence_list)
                embedding_dict[int(data['transcriptcomponentid'][i])] = embedding
            
            with open('data/earningscall/embeddings/' + file.split('.')[0] + '_embedding.json', 'w') as f:
                json.dump(embedding_dict, f)
            
            logging.info('Processed file: %s', file)
            metadata['embedding'][transcript_index] = 1

        except:
            logging.error('Error in file: %s', file)
            metadata['embedding'][transcript_index] = -1
        
        countn += 1
        logging.info('Processing progress: %s %s', countn / file_num * 100, '%')

        if countn % 100 == 0:
            metadata.to_csv('data/metadata/earningscall_list_embedding.csv', index=False)