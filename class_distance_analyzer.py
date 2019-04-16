from keras.models import Model, load_model
from load_data import load_csv, get_onehot
import numpy as np

is_dna_data = True

num_classes = 1000 #test classes, not train classes
top_n = 10

model_name = 'blstm_dna_1000class_4500'
seq_len = 4500
#data_file = '/mnt/data/computervision/dna_train80_val10_test10/test.csv'
data_file = '../results/dna_unknown_1000class_pairs.csv'

model_file = '../models/'+model_name+'.h5'
model = load_model(model_file)
embed_model = Model(inputs=model.input, outputs=model.get_layer("lstm_2").output)
print embed_model.summary()

single_dict = dict()
pair_dict = dict()
data = load_csv(data_file)
for (x, y) in data:
	if y in pair_dict:
		continue
	if y in single_dict:
		assert x != single_dict[y]
		pair_dict[y] = [single_dict[y], x]
	else:
		single_dict[y] = x
	if len(pair_dict) == num_classes:
		break

chosen_data = []
for i in range(2):
	for y in pair_dict:
		x = pair_dict[y][i]
#		print len(x)
		chosen_data.append((x, y))

x, y = get_onehot(chosen_data, None, is_dna_data=is_dna_data, seq_len=seq_len)
embed = embed_model.predict(x)
"""
correct_count = 0
for i in range(num_classes):
	best_dist = None
	best_index = 0
	ex = embed[i + num_classes]
	for j in range(num_classes):
		dist = np.linalg.norm(ex - embed[j])
		if best_dist == None or dist < best_dist:
			best_dist = dist
			best_index = j
	if i == best_index:
		correct_count += 1
	print i, ":", best_index
print correct_count
"""
pos_counts = []
for _ in range(top_n):
	pos_counts.append(0)
correct_count = 0.0
for i in range(num_classes):
	distances = dict()
	ex = embed[i + num_classes]
	for j in range(num_classes):
		dist = np.linalg.norm(ex - embed[j])
		distances[j] = dist
	best = sorted(distances, key=distances.get)[0:top_n]
	#print i, ":", best
	for pos in range(top_n):
		if best[pos] == i:
			pos_counts[pos] += 1
			correct_count += 1
print pos_counts
print correct_count/num_classes
