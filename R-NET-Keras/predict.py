import numpy as np
import pickle
from keras import backend as K
from keras.models import Model, load_model
from keras.layers.recurrent_test4 import *
from model3 import *

def split_train_val(data, rate=0.1):
    nb_validation_samples = int(rate * len(data))
    train = data[nb_validation_samples:]
    val = data[0:nb_validation_samples]

    return train, val

custom_objects = {"SharedWeight:" : SharedWeight, "SharedWeightLayer": SharedWeightLayer, "GatedAttnGRUCell": GatedAttnGRUCell, "SelfMatchGRUCell": SelfMatchGRUCell, "PointerCell": PointerCell, "QuestionPooling": QuestionPooling, "Slice": Slice}

print('Preparing model...', end='')
model = load_model('./models_fasttext/24-t5.9688470938005525-v6.772979313249876.model', custom_objects)

inputs = model.inputs
outputs = [output for output in model.outputs]
predicting_model = Model(inputs, outputs)

print('Done!')

print("Loading dataset...", end='')
#with open('../train_process_final.pkl', 'rb') as f:
with open('../train_process_final_fasttext.pkl', 'rb') as f:
    datas = pickle.load(f)

_, encode_context, encode_ques, start_list, end_list, embeddings_matrix = datas

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

print(model.metrics_names)
#print(model.evaluate([trainP, trainQ], [train_start, train_end]))
print(model.evaluate([valP, valQ], [val_start, val_end]))
