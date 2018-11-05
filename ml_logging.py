import numpy as np
import csv

num_classes = 30

class Logger:
	def __init__(self, name):
		self.acc_plot = []
		self.name = name

	def record_val_acc(self, time, acc):
		self.acc_plot.append([time, acc])
	
	#data: (x, y) pairs before onehot encoding, pred: predicted class integers
	def confusion_matrix(self, data, pred):
		self.conf_mat = np.zeros((num_classes, num_classes), dtype=np.int32)
		for i in range(len(data)):
			self.conf_mat[data[i][1], pred[i]] += 1

	def length_plot(self, data, pred):
		lengths = []
		for (x, y) in data:
			lengths.append(len(x))
		self.len_plot = []
		for i in range(len(data)):
			correct = 1 if data[i][1] == pred[i] else 0
			self.len_plot.append([lengths[i], correct])

	def save(self):
		path = '../results/' + self.name
		with open(path + '_acc_plot.csv', 'w') as outfile:
			w = csv.writer(outfile)
			for row in self.acc_plot:
				w.writerow(row)

		with open(path + '_conf_matrix.csv', 'w') as outfile:
			w = csv.writer(outfile)
			w.writerow(['class'] + range(num_classes) + ['total', 'acc'])
			cm = self.conf_mat.tolist()
			for i in range(num_classes):
				total = 0
				for count in cm[i]:
					total += count
				acc = float(cm[i][i]) / max(total, 1)
				w.writerow([i] + cm[i] + [total, acc])

		with open(path + '_length_plot.csv', 'w') as outfile:
			w = csv.writer(outfile)
			for row in self.len_plot:
				w.writerow(row)
