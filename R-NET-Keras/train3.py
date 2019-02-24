import numpy as np
import argparse
from f1_score4sentence import f1_scores
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping

import pickle

from model3 import RNet
import sys
sys.setrecursionlimit(100000)

def split_train_val(data, rate=0.1):
    nb_validation_samples = int(rate * len(data))
    train = data[nb_validation_samples:]
    val = data[0:nb_validation_samples]

    return train, val

parser = argparse.ArgumentParser()
parser.add_argument('--hdim', default=100, help='Model to evaluate', type=int)
parser.add_argument('--batch_size', default=64, help='Batch size', type=int)
parser.add_argument('--nb_epochs', default=50, help='Number of Epochs', type=int)
#parser.add_argument('--optimizer', default='Adadelta', help='Optimizer', type=str)
parser.add_argument('--optimizer', default='Adam', help='Optimizer', type=str)
parser.add_argument('--lr', default=None, help='Learning rate', type=float)
parser.add_argument('--name', default='', help='Model dump name prefix', type=str)
parser.add_argument('--loss', default='sparse_categorical_crossentropy', help='Loss', type=str)

parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--char_level_embeddings', action='store_true')

# parser.add_argument('model', help='Model to evaluate', type=str)
args = parser.parse_args()


print('Loading datasets...', end='')
with open('../train_process_final_fasttext.pkl', 'rb') as f:
#with open('../train_process_final.pkl', 'rb') as f:
    datas = pickle.load(f)

encode_context, encode_ques, start_list, end_list, embeddings_matrix = datas
## shuffle data
np.random.seed(2)
indices = np.arange(len(encode_context))
np.random.shuffle(indices)

encode_context = encode_context[indices]
encode_ques = encode_ques[indices]

start_list = np.array(start_list)
start_list = start_list[indices]

end_list = np.array(end_list)
end_list = end_list[indices]

## Split data to train and val
trainP, valP = split_train_val(encode_context)
trainQ, valQ = split_train_val(encode_ques)
train_start, val_start = split_train_val(start_list)
train_end, val_end = split_train_val(end_list)
print('Done!')

print('Creating the model...', end='')
model = RNet(vocab_size=len(embeddings_matrix), vocab_init=embeddings_matrix,
        hdim=args.hdim, dropout=args.dropout, p_length=170, q_length=35,
        char_level_embeddings=args.char_level_embeddings)
print('Done!')

print('Compiling Keras model...', end='')
optimizer_config = {'class_name': args.optimizer,
                    'config': {'lr': args.lr} if args.lr else {}}

f1_ = f1_scores()
model.compile(optimizer=optimizer_config,
              loss=args.loss,
              metrics=['accuracy'])
print('Done!')

print('Training...', end='')

path = 'models_fasttext/' + args.name + '{epoch}-t{loss}-v{val_loss}.model'


call_backs = [ModelCheckpoint(path, verbose=1, save_best_only=True),
              EarlyStopping(monitor='val_loss', patience=5), f1_]
model.fit([trainP, trainQ], [train_start, train_end], validation_data=([valP, valQ], [val_start, val_end]), epochs=30, batch_size=args.batch_size, callbacks=call_backs)

model.save('./models_fasttext/Final_model.h5')

print('Training Done!')
