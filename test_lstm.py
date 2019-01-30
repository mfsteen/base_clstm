
from load_data import load_csv, get_onehot
from model_templates import original_blstm, dna_blstm

is_dna_data = True

model_file = '../models/blstm_dna_conv3_4500.h5'
data_dir = '/mnt/data/computervision/dna_train80_val10_test10'
sequence_length = 300
random_crop = True

num_classes = 30
num_letters = 4 if is_dna_data else 26

model = (dna_blstm if is_dna_data else original_blstm)(num_classes, num_letters, sequence_length)

model.load_weights(model_file)
model.summary()

test_data = load_csv(data_dir + '/test.csv', divide=4 if is_dna_data else 1)
crop_count = 0.0
for seq, y in test_data:
	if len(seq) > sequence_length:
		crop_count += 1
print "percent cropped: ", crop_count / len(test_data)	

test_x, test_y = get_onehot(test_data, None, is_dna_data=is_dna_data, seq_len=sequence_length, rand_start=random_crop)
print "test accuracy: ", model.evaluate(test_x, test_y, batch_size=500)
