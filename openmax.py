from keras.models import Model, load_model
from load_data import load_csv, get_onehot
import numpy as np
import math
import csv
import libmr

num_classes = 30

model_file = '../models/blstm_openset.h5'
#data_file = '/mnt/data/computervision/train80_val10_test10/test.csv'
data_file = '/mnt/data/computervision/train80_val10_test10/unknowns.csv'

model = load_model(model_file)
av_model = Model(inputs=model.input, outputs=model.get_layer("AV").output)
print av_model.summary()

data = load_csv(data_file)
print len(data)
x, y = get_onehot(data, None)
avs = av_model.predict(x, batch_size=1000)

print 'done getting avs'
del data, x, y

means = []
with open('../results/mean_activations.csv', 'r') as infile:
	r = csv.reader(infile)
	for row in r:
		means.append(np.array(row, dtype=np.float32))

dists = []
with open('../results/mav_distances.csv', 'r') as infile:
	r = csv.reader(infile)
	for row in r:
		dists.append(np.array(row, dtype=np.float32).tolist())

models = []
for row in dists:
	model = libmr.MR()
	model.fit_high(row, len(row))
	models.append(model)

print 'done fitting models'

alpha = 5
alpha_weights = [((alpha+1) - i)/float(alpha) for i in range(1, alpha+1)]

softmaxes = []
scores = []
distances = []
distance_sum = 0.0

for i in range(avs.shape[0]):
	x = avs[i]

	e_sum = 0.0
	for j in range(num_classes):
		e_sum += math.exp(x[j])
	softmax = np.zeros((num_classes), dtype=np.float32)
	for j in range(num_classes):
		softmax[j] = math.exp(x[j]) / e_sum
	softmaxes.append(np.max(softmax))

	top_classes = x.argsort()[::-1]
	unknown_act = 0.0
	for rank in range(alpha):
		j = top_classes[rank]
		dist = np.linalg.norm(x - means[j])

		if rank == 0:
			distances.append(dist)
			distance_sum += dist

		score = models[j].w_score(dist)
		weight = 1.0 - score * alpha_weights[rank]
		unknown_act += x[j] * (1 - weight)
		x[j] = x[j] * weight

	e_sum = math.exp(unknown_act)
	for j in range(num_classes):
		e_sum += math.exp(x[j])
	openmax = np.zeros((num_classes+1), dtype=np.float32)
	openmax[0] = math.exp(unknown_act) / e_sum
	for j in range(num_classes):
		openmax[j+1] = math.exp(x[j]) / e_sum
	y = np.argmax(openmax)
	score = 0.0 if y == 0 else openmax[y]
	scores.append(score)

print "average distance: " + str(distance_sum / len(scores))

thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995]

for threshold in thresholds:
	for arr in [softmaxes, scores]:
		in_count = 0.0
		out_count = 0.0
		for score in arr:
			if score < threshold:
				out_count += 1
			else:
				in_count += 1
		in_percent = in_count / (in_count + out_count)
		out_percent = out_count / (in_count + out_count)
		print 'threshold: ' + str(threshold) + ' known: ' + str(in_percent) + ' unknown: ' + str(out_percent) 

for threshold in range(20):
	in_count = 0.0
	out_count = 0.0
	for dist in distances:
		if dist < threshold:
			in_count += 1
		else:
			out_count += 1
	in_percent = in_count / (in_count + out_count)
        out_percent = out_count / (in_count + out_count)
        print 'threshold: ' + str(threshold) + ' known: ' + str(in_percent) + ' unknown: ' + str(out_percent)

	
