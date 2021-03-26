#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 15:26:54 2021

@author: shweta
"""

import pandas as pd
import string

fpath  = 'data/doc_cluster_labels.tsv'

df = pd.read_csv(fpath, sep='\t')

df.columns = ['sentence1', 'sentence2', 'label']
df['idx'] = df.index
df_final = df[['idx', 'label', 'sentence1', 'sentence2']]

from sklearn.model_selection import train_test_split

train, test = train_test_split(df_final, test_size=0.1)



train.to_csv('data/orcas_dataset/train.csv', index=False)
test.to_csv('data/orcas_dataset/validation.csv', index=False)



fpath2  = 'data/quora_eval.tsv'
df2 = pd.read_csv(fpath2, sep='\t')
df2.columns = ['id', 'qid1', 'qid2', 'sentence1', 'sentence2', 'label']
df2 = df2.drop(columns=['id', 'qid1', 'qid2'])

df2['sentence1'] = df2['sentence1'].astype(str).apply(lambda x: x.lower())
df2['sentence2'] = df2['sentence2'].astype(str).apply(lambda x: x.lower())

df2['sentence1'] = df2['sentence1'].str.replace('[{}]'.format(string.punctuation), '')
df2['sentence2'] = df2['sentence2'].str.replace('[{}]'.format(string.punctuation), '')

df2['idx'] = df2.index
df2_final = df2[['idx', 'label', 'sentence1', 'sentence2']]
df2_final.to_csv('data/orcas_dataset/test.csv', index=False)




