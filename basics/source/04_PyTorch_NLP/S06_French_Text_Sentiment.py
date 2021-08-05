import os
import json
import time
import torch
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from torch.utils.data import TensorDataset, random_split, \
    DataLoader, RandomSampler, SequentialSampler
from transformers import CamembertForSequenceClassification, CamembertTokenizer, \
    AdamW, get_linear_schedule_with_warmup

"""
4.6 Introduction
In this tutorial, we will analysis the sentiment of a user based on his comments. We will use the the CamemBERT
model as our embedding model.

4.6.1 What is CamemBERT?
CamemBERT is a “version” of pre-trained RoBERTa base on a set of french texts. RoBERTa is a sub version of BERT,
in which certain hyper-parameters of the pre-training has been modified. The objectif for the next sentence prediction
has been deleted. As a result, CamemBERT inherit the avantages of BERT.

In this tutorial, we will use the Transformer (huggingface) and PyTorch.

4.6.2 Architecture of our model

The architecture is very simple, we add a feed-forward network on top of the CamemBERT, and an output layer(softmax)

"""


def preprocess(raw_reviews, sentiments=None):
    """
    Cette fonction prends de la donnée brute en argument et retourne un 'dataloader' pytorch

    Args
        raw_reviews (array-like) : Une liste de reviews sous forme de 'str'

        sentiments : Une liste 'sentiments' (0 = negatif, 1 = positif) de la meme taille que
                     'raw_review'

    Returns
        inputs_ids, attention_masks, sentiments(optionel) : Objet  de PyTorch qui contient
                    les versions tokenisees et encodees des donnees brutes
    """
    # create a camembert tokenizer
    TOKENIZER = CamembertTokenizer.from_pretrained(
        'camembert-base',
        do_lower_case=True)
    encoded_batch = TOKENIZER.batch_encode_plus(raw_reviews,
                                                add_special_tokens=False,
                                                pad_to_max_length=True,
                                                return_attention_mask=True,
                                                return_tensors='pt')
    if sentiments:
        sentiments = torch.tensor(sentiments)
        return encoded_batch['input_ids'], encoded_batch['attention_mask'], sentiments
    return encoded_batch['input_ids'], encoded_batch['attention_mask']


def load_data(train_file_path, test_file_path, val_file_path):
    # the raw data has three column, but we only need two. So the required column list indicate the required col name
    col_list = ["popularity", "review"]
    # load train data
    # error_bad_lines=False skip rows that does not fit in the schema
    train = pd.read_csv(train_file_path, sep='|', error_bad_lines=False)
    train = train[col_list]
    print("training data:\n")
    print(train.head(5))
    print(f"training data has shape: {train.shape}")

    # load test data
    test = pd.read_csv(test_file_path, sep='|', error_bad_lines=False)
    test = test[col_list]
    print("test data:\n")
    print(test.head(5))
    print(f"test data has shape: {test.shape}")

    # load val data
    val = pd.read_csv(val_file_path, sep='|', error_bad_lines=False)
    val = val[col_list]
    print("val data:\n")
    print(val.head(5))
    print(f"val data has shape: {val.shape}")

    return train, test, val


def main():
    # step1: load data
    train_file_path = "/home/pliu/data_set/nlp/allo_cine/train.csv"
    test_file_path = "/home/pliu/data_set/nlp/allo_cine/test.csv"
    val_file_path = "/home/pliu/data_set/nlp/allo_cine/val.csv"
    train, test, val = load_data(train_file_path, test_file_path, val_file_path)


if __name__ == "__main__":
    main()
