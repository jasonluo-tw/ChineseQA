from keras.utils.generic_utils import CustomObjectScope
from keras.layers.recurrent import GRU
#from keras.layers.recurrent_custom import RNN, GatedAttnGRUCell, SelfMatchGRUCell
#from keras.layers.recurrent_custom import QuestionPooling, PointerCell, Slice
#from keras.layers.recurrent_custom import SharedWeight
from custom_layers.recurrent_custom import RNN, GatedAttnGRUCell, SelfMatchGRUCell
from custom_layers.recurrent_custom import QuestionPooling, PointerCell, Slice
from custom_layers.recurrent_custom import SharedWeight

from keras.layers.pooling import GlobalMaxPooling1D
from keras import regularizers

from keras.layers.core import Masking, RepeatVector, Dropout
from keras import backend as K
from keras.layers.wrappers import Bidirectional
from keras.layers import Input, Multiply, Dense, Embedding, Concatenate, Average
from keras.models import Model
import numpy as np

def RNet(vocab_size, vocab_init=None, hdim=150, dropout=0.1, 
        p_length=None, q_length=None, w2vec=300, char_level_embeddings=False):
    ## define variables
    H = hdim
    N = p_length
    M = q_length
    W = w2vec
    
    P_vecs = Input(shape=[N], name='Passage')
    Q_vecs = Input(shape=[M], name='Question')
    
    v = SharedWeight(size=(H, 1), name='v')
    WQ_u = SharedWeight(size=(2 * H, H), name='WQ_u')
    WP_u = SharedWeight(size=(2 * H, H), name='WP_u')
    WP_v = SharedWeight(size=(H, H), name='WP_v')
    W_g1 = SharedWeight(size=(2 * H, 2 * H), name='W_g1')
    W_g2 = SharedWeight(size=(H, H), name='W_g2')
    WP_h = SharedWeight(size=(2 * H, H), name='WP_h')
    Wa_h = SharedWeight(size=(2 * H, H), name='Wa_h')
    WQ_v = SharedWeight(size=(2 * H, H), name='WQ_v')
    WPP_v = SharedWeight(size=(H, H), name='WPP_v')
    VQ_r = SharedWeight(size=(H, H), name='VQ_r')
    
    vv = SharedWeight(size=(4 * H, 2 * H), name='vv')
    vv2 = SharedWeight(size=(2 * H, H), name='vv2')
    
    shared_weights = [v, WQ_u, WP_u, WP_v, W_g1, W_g2, WP_h, Wa_h, WQ_v, WPP_v, VQ_r, vv, vv2]
    if vocab_init is not None: 
        em = Embedding(vocab_size, W, weights=[vocab_init], trainable=False, mask_zero=True)
    else:
        em = Embedding(vocab_size, W, trainable=False, mask_zero=True)
    
    uP = em(P_vecs)
    uQ = em(Q_vecs)
    
    #uP = Masking()(P)
    for i in range(3):
        uP = Bidirectional(GRU(units=H, 
                               return_sequences=True, 
                               #kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
                               dropout=dropout))(uP)
    
    #uQ = Masking()(Q)
    for i in range(3):
        uQ = Bidirectional(GRU(units=H, 
                               return_sequences=True, 
                               #kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
                               dropout=dropout))(uQ)
    
    ## Gated Attn 
    cell = GatedAttnGRUCell(units=H)
    
    vP = RNN(cell, return_sequences=True)(uP, constants=[WP_u, uQ, WQ_u, WP_v, v, W_g1, vv])
    vP_back = RNN(cell, return_sequences=True, go_backwards=True)(uP, constants=[WP_u, uQ, WQ_u, WP_v, v, W_g1, vv])
    #vP = Average()([vP, vP_back])
    vP = Concatenate()([vP, vP_back])
    vP = GRU(units=H, return_sequences=True)(vP)

    ## Self Match
    cell2 = SelfMatchGRUCell(units=H)
    
    hP = RNN(cell2, return_sequences=True)(vP, constants=[vP, W_g2, WPP_v, WP_v, v, vv2])
    hP_back = RNN(cell2, return_sequences=True, go_backwards=True)(vP, constants=[vP, W_g2, WPP_v, WP_v, v, vv2])
    hP = Concatenate()([hP, hP_back])
    
    ## Question Pooling
    rQ = QuestionPooling()([uQ, WQ_u, WQ_v, v, VQ_r])
    rQ = Dropout(rate=dropout, name='rQ')(rQ)
    
    fake_em = Embedding(vocab_size, 2 * H, trainable=False)(P_vecs)
    fake_input = GlobalMaxPooling1D()(fake_em)
    fake_input = RepeatVector(n=2, name='fake_input')(fake_input)
    
    ## Pointer
    cell3 = PointerCell(units= 2 * H)
    Ptr = RNN(cell3, return_sequences=True)(fake_input, initial_state=[rQ], constants=[hP, WP_h, Wa_h, v])
    
    answer_start = Slice(0, name='answer_start')(Ptr)
    answer_end = Slice(1, name='answer_end')(Ptr)
    
    inputs = [P_vecs, Q_vecs] + shared_weights
    outputs = [answer_start, answer_end]
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.summary()

    return model

if __name__ == '__main__':
    aa = RNet(150)
