import pickle
import numpy as np

class data_generator():
    def __init__(self, filename, batch_size, val_rate=None, shuffle=True):
        
        ## read file first
        with open(filename, 'rb') as f:
            datas = pickle.load(f)

        context, ques, start, end, embedding_matrix = datas
        
        ## define some parameter
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.val_rate = val_rate
        self.max_context_length = max([len(i) for i in context])
        self.max_ques_length = max([len(i) for i in ques])
        self.em_size = embedding_matrix.shape[1]
        self.embedding_matrix = embedding_matrix
        print('Batch_size: %d, Embedding_size: %d, max_length: %d, max_q_length: %d'%(self.batch_size, self.em_size, self.max_context_length, self.max_ques_length))

        ## shuffle data
        if self.shuffle:
            indices = np.arange(len(context))
            np.random.shuffle(indices)

            context = context[indices]
            ques = ques[indices]

            start = np.array(start)
            start = start[indices]

            end = np.array(end)
            end = end[indices]

        ## split data to train and val
        if val_rate is not None:
            self.trainP, self.valP = self.split_train_val(context)
            self.trainQ, self.valQ = self.split_train_val(ques)
            self.train_start, self.val_start = self.split_train_val(start)
            self.train_end, self.val_end = self.split_train_val(end)
            
            print('Training data: %d'%len(self.trainP))
            print('Validation data: %d'%len(self.valP))
        
        ## No validation data
        else:
            self.trainP = context
            self.trainQ = ques
            self.train_start = start
            self.train_end = end
            
            self.valP = None
            self.valQ = None
            self.val_start = None
            self.val_end = None
            
            print('Training data: %d'%len(self.trainP))
        
        self.train_steps = len(self.trainP) // self.batch_size

    def split_train_val(self, data):
        assert self.val_rate != None
        nb_validation_samples = int(self.val_rate * len(data))
        train = data[nb_validation_samples:]
        val = data[0:nb_validation_samples]

        return train, val

    def get_word_vector(self, sentence, max_length):
        cc = np.zeros((max_length, self.em_size))
        for index, idx in enumerate(sentence):
            cc[index] = self.embedding_matrix[int(idx)]
        
        return cc

    def train_generator(self):
        start = 0
        while True:
            stop = start + self.batch_size
            diff = stop - self.trainP.shape[0]
            if diff <= 0:
                P_batch = self.trainP[start:stop]
                Q_batch = self.trainQ[start:stop]
                
                start_batch = self.train_start[start:stop]
                end_batch = self.train_end[start:stop]
                
                start += self.batch_size
            else:
                P_batch = np.concatenate((self.trainP[start:], self.trainP[:diff]))
                Q_batch = np.concatenate((self.trainQ[start:], self.trainQ[:diff]))
                
                start_batch = np.concatenate((self.train_start[start:], self.train_start[:diff]))
                end_batch = np.concatenate((self.train_end[start:], self.train_end[:diff]))
                
                start = diff

            ## Get word vectors
            P_batchIN = np.zeros((len(P_batch), self.max_context_length, self.em_size))
            Q_batchIN = np.zeros((len(Q_batch), self.max_ques_length, self.em_size))

            for ii, (paragraph, quess) in enumerate(zip(P_batch, Q_batch)):
                P_batchIN[ii] = self.get_word_vector(paragraph, self.max_context_length)
                Q_batchIN[ii] = self.get_word_vector(quess, self.max_ques_length)

            yield [P_batchIN, Q_batchIN], [start_batch, end_batch]

    def val_generator(self):
        assert self.val_rate != None
        ## Get word vectors
        P_batchIN = np.zeros((len(self.valP), self.max_context_length, self.em_size))
        Q_batchIN = np.zeros((len(self.valQ), self.max_ques_length, self.em_size))
        
        for ii, (paragraph, quess) in enumerate(zip(self.valP, self.valQ)):
            P_batchIN[ii] = self.get_word_vector(paragraph, self.max_context_length)
            Q_batchIN[ii] = self.get_word_vector(quess, self.max_ques_length)

        return [P_batchIN, Q_batchIN], [self.val_start, self.val_end]
        #while True:
        #    yield [P_batchIN, Q_batchIN], [self.val_start, self.val_end]


if __name__ == '__main__':
    test = data_generator('../train_process_final_fasttext.pkl', 64, 0.1)
    [xx0, xx1], [yy0, yy1] = test.train_generator()
    print(xx0.shape, xx1.shape, yy0.shape, yy1.shape)
    #[xx0, xx1], [yy0, yy1] = test.val_generator()
    #print(xx0.shape, xx1.shape, yy0.shape, yy1.shape)
