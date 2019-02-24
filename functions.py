import numpy as np
import re
import jieba
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors

def prepare_data(Alist):
    train_context = []
    ques = []
    ids = []
    ans = []

    for article in Alist:
        articles = article['paragraphs']
        for context in articles:
            for qas in context['qas']:
                train_context.append(context['context'])
                ques.append(qas['question'])
                ids.append(qas['id'])
                ans.append(qas['answers'][0]['text']) ## answer_start, text

    return train_context, ques, ids, ans

def cal_dis(train_context, ques, ans):
    ## Load the pretrain model
    zh_model = KeyedVectors.load_word2vec_format('../pretrain/fasttext_pretrain_wiki.zh.vec')
    print('Load fasttext dataset done !!!')

    allcount = 0
    new_context = []
    new_ques = []
    new_ans = []

    for num in range(len(train_context)):
        corpus_origin = []
        corpus = re.split('，|。', train_context[num])
        corpus.insert(0, ques[num])

        for index, sen in enumerate(corpus):
            nomark_para = ''.join([i for i in sen if i.isalnum()])
            nomark_para = nomark_para.replace('\n', '')

            line = jieba.lcut(nomark_para)
            corpus[index] = line
            corpus_origin.append(nomark_para)
        
        corpus.pop()
        
        ## calculate the sentence similarity
        vectors = []
        for sen in corpus:
            sen_aver_vector = 0
            word_count = 0
            for word in sen:
                if word in zh_model.wv:
                    if word in ques[num]:
                        alpha = 3 / len(word)
                    else:
                        alpha = 1/ len(word)

                    sen_aver_vector = sen_aver_vector + alpha * zh_model.wv[word] / np.sqrt(np.sum(zh_model.wv[word] ** 2))
                    word_count += 1

            try:
                vectors.append(sen_aver_vector / word_count)
            except:
                vectors.append(np.zeros(300))

        relvent_num = np.argmax(cosine_similarity(vectors[0:1], vectors[1:])) + 1
        start = relvent_num - 4 if relvent_num - 4 >= 1 else 1
        end = start + 8 if start + 8 <= len(vectors) else len(vectors)

        check = ''.join(corpus_origin[i] for i in range(start, end))

        if ans[num] in check:
            new_context.append(jieba.lcut(check))
            new_ques.append(corpus[0])
            new_ans.append(jieba.lcut(ans[num]))
            allcount += 1

    print('Total successful data:', allcount)
    return new_context, new_ques, new_ans

def ans_index(train_context, ques, ans):
    start_list = []
    end_list = []
    train_clean = []
    ques_clean = []
    ans_clean = []

    bad_data = []
    for num in range(len(train_context)):
        count = len(ans[num])
        try:
            for index, word in enumerate(train_context[num]):
                break_ = False
                for ii in range(count):
                    if ans[num][ii] in train_context[num][index + ii]:
                        ans_start = index
                        ans_end = index + ii
                    else:
                        break
                    
                    if ii + 1 == count:
                        break_ = True
                        break
                if break_:
                    break

            start_list.append(ans_start)
            end_list.append(ans_end)
            train_clean.append(train_context[num])
            ques_clean.append(ques[num])
            ans_clean.append(ans[num])
        except:
            bad_data.append(num)

    ## check answers dimension
    print(len(train_clean), len(ques_clean), len(ans_clean), len(start_list), len(end_list))
    return train_clean, ques_clean, ans_clean, start_list, end_list, bad_data

### the function below is in FastText_1219.ipynb
def set_up_dict(paragraph, word2idx, idx2word, embedding_matrix, model):
    now = len(idx2word) - 1
    for word in paragraph:
        if word not in word2idx:
            now += 1
            word2idx[word] = now
            idx2word.append(word)
            embedding_matrix.append(model.get_sentence_vector(word))
    
    return word2idx, idx2word, embedding_matrix

def encode_words(document, word2idx, max_length):
    cc = np.zeros((max_length))
    for index, word in enumerate(document):
        if word in word2idx:
            cc[index] = word2idx[word]

    return cc

