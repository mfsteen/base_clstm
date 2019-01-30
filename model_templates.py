from keras.models import Sequential, load_model
from keras.layers import LSTM, Masking, Dense,  Bidirectional, Dropout, MaxPooling1D, Conv1D, Activation
from keras.optimizers import Adam

def original_blstm(num_classes, num_letters, sequence_length):
	model = Sequential()
	model.add(Conv1D(input_shape=(sequence_length, num_letters), filters=320, kernel_size=26, padding="valid", activation="relu"))
	model.add(MaxPooling1D(pool_length=13, stride=13))
	model.add(Dropout(0.2))
	model.add(Bidirectional(LSTM(320, activation="tanh", return_sequences=True)))
	model.add(Dropout(0.5))
	#model.add(LSTM(num_classes, activation="softmax", name="AV"))
	model.add(LSTM(50, activation="tanh"))
	model.add(Dense(num_classes, activation=None, name="AV"))
	model.add(Activation("softmax"))
	model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
	return model

def dna_blstm(num_classes, num_letters, sequence_length):
	model = Sequential()
        model.add(Conv1D(input_shape=(sequence_length, num_letters), filters=26, kernel_size=3, strides=3, padding="valid", activation="relu"))
        model.add(Conv1D(filters=320, kernel_size=26, padding="valid", activation="relu"))
	model.add(MaxPooling1D(pool_length=13, stride=13))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(320, activation="tanh", return_sequences=True)))
        model.add(Dropout(0.5))
        #model.add(LSTM(num_classes, activation="softmax", name="AV"))
        model.add(LSTM(50, activation="tanh"))
        model.add(Dense(num_classes, activation=None, name="AV"))
        model.add(Activation("softmax"))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model
