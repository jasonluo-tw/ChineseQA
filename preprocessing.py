from gensim.models import KeyedVectors
from functions import prepare_data, cal_dis, ans_index, set_up_dict, encode_words
import json
import pickle
import fastText
import numpy as np
import jieba
import re
jieba.set_dictionary('../dict.txt.big')

## Open the dataset
with open('../train-QA.json', 'r', encoding='utf-8') as f:
    datas = json.load(f)
print('load dataset done !!!')

## Prepare data
train_context, ques, _, ans = prepare_data(datas['data']) # _ is id
print('prepare dataset done !!!')

## Calculate the distance
train_context, ques, ans = cal_dis(train_context, ques, ans)
print('calculate the passage and ques similarity done !!!')

## preprocess the ans
train_context, ques, ans, start_list, end_list, _ = ans_index(train_context, ques, ans)


## load the fasttext model
model = fastText.load_model('../pretrain/wiki.zh.bin')
print('re-load fasttext model')

## set up the dictionary
word2idx = {"<Pad>": 0}
idx2word = ['<Pad>']
embedding_matrix = [[0]*300]

for ii in train_context:
    word2idx, idx2word, embedding_matrix = set_up_dict(ii, word2idx, idx2word, embedding_matrix, model)

for ii in ques:
    word2idx, idx2word, embedding_matrix = set_up_dict(ii, word2idx, idx2word, embedding_matrix, model)

embedding_matrix = np.array(embedding_matrix)

## index the training data
max_context_length = max([len(i) for i in train_context])
max_ques_length = max([len(i) for i in ques])
encode_context = np.zeros((len(train_context), max_context_length))
encode_ques = np.zeros((len(train_context), max_ques_length))

for ii, (paragraph, quess) in enumerate(zip(train_context, ques)):
    encode_context[ii] = encode_words(paragraph, word2idx, max_context_length)
    encode_ques[ii] = encode_words(quess, word2idx, max_ques_length)

## Check the data quality
"""
for item in start_list:
    if item > max_context_length:
        print(item)
for item in end_list:
    if item > max_context_length:
        print(item)
"""

## save the data to pickle file
with open('./training_data_fasttext.pkl', 'wb') as f:
    pickle.dump([encode_context, encode_ques, start_list, end_list, embedding_matrix], f)
