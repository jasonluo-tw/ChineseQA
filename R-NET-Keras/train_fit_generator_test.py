import numpy as np
import argparse
from f1_score4sentence import f1_scores
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping

import pickle

from model3_fit_generator_test import RNet
from generators import data_generator
import sys
sys.setrecursionlimit(100000)

parser = argparse.ArgumentParser()
parser.add_argument('--hdim', default=128, help='Model to evaluate', type=int)
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
datas = data_generator('../../train_process_final_fasttext.pkl', args.batch_size, val_rate=0.1)
print('Creating the model...', end='')
model = RNet(hdim=args.hdim, dropout=args.dropout, p_length=170, q_length=35,
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

path = 'models_fasttext/' + args.name + '{epoch}-t{loss}-v{val_loss}_fit_generator.model'


call_backs = [ModelCheckpoint(path, verbose=1, save_best_only=True),
              EarlyStopping(monitor='val_loss', patience=5), f1_]

## model fit_generator
valx, valy = datas.val_generator()
model.fit_generator(generator=datas.train_generator(), 
                    steps_per_epoch=datas.train_steps,
                    validation_data=(valx, valy),
                    validation_steps=1,
                    epochs=30,
                    callbacks=call_backs)

#model.save('./models_fasttext/Final_model_fit_generator.h5')

print('Training Done!')
