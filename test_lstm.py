from keras.models import Sequential, load_model
from keras.layers import LSTM, Masking, Dense,  Bidirectional, Dropout, MaxPooling1D, Conv1D, Activation
from keras.optimizers import Adam
from load_data import load_csv, get_onehot

model_file = '../models/blstm_seq100.h5'
data_dir = '/mnt/data/computervision/train80_val10_test10'
sequence_length = 100
random_crop = True

num_classes = 30
num_amino_acids = 26
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

model.load_weights(model_file)
model.summary()

test_data = load_csv(data_dir + '/test.csv')
crop_count = 0.0
for seq, y in test_data:
	if len(seq) > sequence_length:
		crop_count += 1
print "percent cropped: ", crop_count / len(test_data)	

test_x, test_y = get_onehot(test_data, None, seq_len=sequence_length, rand_start=random_crop)
print "test accuracy: ", model.evaluate(test_x, test_y, batch_size=1000)
