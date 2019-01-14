from keras.models import Model, load_model
from load_data import load_csv, get_onehot
import numpy as np
import csv

num_classes = 30

model_file = '../models/blstm_openset.h5'
data_dir = '/mnt/data/computervision/train80_val10_test10'

model = load_model(model_file)
av_model = Model(inputs=model.input, outputs=model.get_layer("AV").output)
print av_model.summary()

train_data = load_csv(data_dir + '/train.csv')

batch_size = 10000
avs = []
actual = []
lower = 0
while lower < len(train_data):
	print lower
	upper = min(lower + batch_size, len(train_data))
	x, y = get_onehot(train_data[lower:upper], None)
	pred = av_model.predict(x, batch_size=1000)
	avs.append(pred)
	actual.append(y)
	lower += batch_size

del train_data

sums = np.zeros((num_classes, num_classes), np.float32)
counts = np.zeros((num_classes), np.float32)
class_avs = []
for i in range(num_classes):
	class_avs.append([])

for i in range(len(avs)):
	for j in range(avs[i].shape[0]):
		pred_class = np.argmax(avs[i][j])
		actual_class = np.argmax(actual[i][j])
		if pred_class == actual_class:
			sums[actual_class] += avs[i][j]
			counts[actual_class] += 1.0
			class_avs[actual_class].append(avs[i][j])
print counts

means = []
distances = []
top_dist_count = 20

for i in range(num_classes):
	means.append(sums[i] / max(counts[i], 1.0))
	d = []
	for ex in class_avs[i]:
		d.append(np.linalg.norm(ex - means[i]))
	distances.append(sorted(d)[-top_dist_count:])


mean_filename = '../results/mean_activations.csv'
with open(mean_filename, 'w') as outfile:
	w = csv.writer(outfile)
	for i in range(num_classes):
		w.writerow(means[i].tolist())
print 'wrote ' + mean_filename

dist_filename = '../results/mav_distances.csv'
with open(dist_filename, 'w') as outfile:
	w = csv.writer(outfile)
	for i in range(num_classes):
		w.writerow(distances[i])
print 'wrote ' + dist_filename
