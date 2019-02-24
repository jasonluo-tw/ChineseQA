from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import pickle, argparse
from keras import backend as K
from keras.models import Model, load_model
from keras.layers.recurrent_test4 import *
from model3 import *

def split_train_val(data, rate=0.1):
    nb_validation_samples = int(rate * len(data))
    train = data[nb_validation_samples:]
    val = data[0:nb_validation_samples]

    return train, val

print("Loading dataset...", end='')
#with open('../train_process_final.pkl', 'rb') as f:
with open('../train_process_final_fasttext.pkl', 'rb') as f:
    datas = pickle.load(f)

_, encode_context, encode_ques, start_list, end_list, embeddings_matrix = datas


## Load model
custom_objects = {"SharedWeight:" : SharedWeight, "SharedWeightLayer": SharedWeightLayer, "GatedAttnGRUCell": GatedAttnGRUCell, "SelfMatchGRUCell": SelfMatchGRUCell, "PointerCell": PointerCell, "QuestionPooling": QuestionPooling, "Slice": Slice}

print('Preparing model...', end='')
model = RNet(vocab_size=len(embeddings_matrix), vocab_init=embeddings_matrix,
            hdim=75, dropout=0, p_length=300, q_length=20,
            char_level_embeddings=False)

#model.load_weights('./models_fasttext/24-t5.9688470938005525-v6.772979313249876.model')
model.load_weights('./models_fasttext/1-t10.019584606646278-v9.309939710510422.model')
model.compile(optimizer=keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print('Done!')


## shuffle data
print('Shuffle dataset...', end='')
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

path = 'models_fasttext/' + '{epoch}-t{loss}-v{val_loss}.model'


call_backs = [ModelCheckpoint(path, verbose=1, save_best_only=True),
              EarlyStopping(monitor='val_loss', patience=5)]

model.fit([trainP, trainQ], [train_start, train_end], validation_data=([valP, valQ], [val_start, val_end]), epochs=20, batch_size=64, callbacks=call_backs)

model.save('./models_fasttext/Final_model.h5')

print('Training Done!')

#print(model.evaluate([valP, valQ], [val_start, val_end]))
