from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Masking, Dense,  Bidirectional, Dropout, MaxPooling1D, Conv1D, Activation
from keras.optimizers import Adam

from load_data import load_csv, get_onehot
from ml_logging import Logger

num_classes = 30
num_amino_acids = 26
sequence_length = 100
logger = Logger('blstm_seq100')
save_path = '../models/blstm_seq100.h5'

model = Sequential()
#model.add(Masking(mask_value=0, input_shape=(1500, num_amino_acids)))
model.add(Conv1D(input_shape=(sequence_length, num_amino_acids), filters=320, kernel_size=26, padding="valid", activation="relu"))
model.add(MaxPooling1D(pool_length=13, stride=13))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(320, activation="tanh", return_sequences=True)))
model.add(Dropout(0.5))
#model.add(LSTM(num_classes, activation="softmax", name="AV"))
model.add(LSTM(50, activation="tanh"))
model.add(Dense(num_classes, activation=None, name="AV"))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
model.summary()

data_dir = '/mnt/data/computervision/train80_val10_test10'
train_data = load_csv(data_dir + '/train.csv')
print len(train_data)
val_data = load_csv(data_dir + '/validation.csv')
val_x, val_y = get_onehot(val_data, None, seq_len=sequence_length)
print len(val_data)

num_episodes = 20000
for i in range(num_episodes):
        x, y = get_onehot(train_data, 1000, seq_len=sequence_length)
        print i
        print model.train_on_batch(x, y)
        if (i % 1000 == 0) or i == num_episodes - 1:

                [loss, acc] = model.evaluate(val_x, val_y, batch_size=1000)
                print loss, acc
                logger.record_val_acc(i, acc)

                model.save(save_path)
                print 'saved to ' + save_path
del train_data

pred = model.predict(val_x, batch_size=1000).argmax(axis=-1)
logger.confusion_matrix(val_data, pred)
logger.length_plot(val_data, pred)
logger.save()

del val_data, val_x, val_y

test_data = load_csv(data_dir + '/test.csv')
test_x, test_y = get_onehot(test_data, None, seq_len=sequence_length)
print "test accuracy: ", model.evaluate(test_x, test_y, batch_size=1000)
