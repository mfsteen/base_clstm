# QUESTION: why are there two seeds?

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import numpy as np
#from keras.models import Sequential
#from keras.layers import LSTM, Masking, Dense,  Bidirectional, Dropout, MaxPooling1D, Conv1D, Activation
#from keras.optimizers import Adam

from load_data import load_csv, get_onehot
from ml_logging import Logger
from model_templates import dna_mask_blstm, aa_mask_blstm, dspace



# model can run with a DNA sequence or with a protein sequence.

# QUESTION: is it right that the alternative to DNA is proteins? I think I remember you saying that
# the alternative was amino acids, but below it says num_letters = 26 if the data is not DNA, so I
# think I may have misheard. Either way, the types of possible input should probably be clarified.

is_dna_data = False


# num_classes is number of different possible annotations.
# num_letters for a DNA sequence is 4 (AGCT), for a protein sequence is 26 (letters of the alphabet)

num_classes = 30
num_letters = 4 if is_dna_data else 26
sequence_length = 1500
embed_size = 64
model_name = 'blstm_mask_embed64_aa_30class_1500'
model_template = aa_mask_blstm
data_dir = '/mnt/data/computervision/train80_val10_test10'

mask = True
mask_len = 113


#logger = Logger(model_name)
save_path = '../models/'+model_name+'.h5'

model = model_template(num_classes, num_letters, sequence_length, embed_size=embed_size, mask_length=mask_len if mask else None)
model.summary()

train_data = load_csv(data_dir + '/train.csv')
print len(train_data)
#val_data = load_csv(data_dir + '/validation.csv', divide=2 if is_dna_data else 1)
#val_x, val_y = get_onehot(val_data, None, num_classes=num_classes, seq_len=sequence_length, is_dna_data=is_dna_data)
#print len(val_data)

num_episodes = 50000#200000
for i in range(num_episodes):
        x, y, m = get_onehot(train_data, 100, num_classes=num_classes, seq_len=sequence_length, is_dna_data=is_dna_data, mask_len=mask_len if mask else None)
        print i
        print model.train_on_batch([x,m] if mask else x, y)
        if (i % 10000 == 0) or i == num_episodes - 1:

                #[loss, acc] = model.evaluate(val_x, val_y, batch_size=100)
                #print loss, acc
                #logger.record_val_acc(i, acc)

                model.save(save_path)
                print 'saved to ' + save_path
del train_data

#pred = model.predict(val_x, batch_size=100).argmax(axis=-1)
#logger.confusion_matrix(val_data, pred)
#logger.length_plot(val_data, pred)
#logger.save()

#del val_data, val_x, val_y

test_data = load_csv(data_dir + '/test.csv', divide=2 if is_dna_data else 1)
test_x, test_y, test_m = get_onehot(test_data, None, num_classes=num_classes, seq_len=sequence_length, is_dna_data=is_dna_data, mask_len=mask_len if mask else None)
print "test accuracy: ", model.evaluate([test_x, test_m] if mask else test_x, test_y, batch_size=100)
