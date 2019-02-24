import numpy as np
from keras.callbacks import Callback
from collections import Counter
class f1_scores(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
            
    def on_epoch_end(self, epoch, logs={}):
        
        def f(x, y):
            cc = ''
            for i in range(x, y+1):
                cc += str(i)
            return cc

        def f1_score(pred, truth):
            common = Counter(pred) & Counter(truth)
            num_same = sum(common.values())
            if num_same == 0:
                f1 = 0
            else:
                precision = 1.0 * num_same / len(pred)
                recall = 1.0 * num_same / len(truth)
                f1 = (2 * precision * recall) / (precision + recall)

            return f1

        ## predict
        start_, end_ = self.model.predict([self.validation_data[0], self.validation_data[1]])
        start_number = list(map(lambda x: np.argmax(x), start_))
        end_number = list(map(lambda x: np.argmax(x), end_))
        pred = list(map(f, start_number, end_number))
        
        ## Truth
        truth_start, truth_end = np.squeeze(self.validation_data[2]), np.squeeze(self.validation_data[3])
        truth = list(map(f, truth_start, truth_end))

        _val_f1 = np.mean(list(map(f1_score, pred, truth)))
        self.val_f1s.append(_val_f1)
        print('#######################')
        print('— val_f1: %f'%(_val_f1))
        print('#######################')
        
    
if __name__ == '__main__':    
    metrics = f1_scores()

