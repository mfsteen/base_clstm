import csv
import random
import numpy as np

random.seed(3)

#use on the output of split_data.py

def load_csv(input_path):
	result = []
	with open(input_path) as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			result.append((row[1], int(row[0])))
	return result #(x, y) pairs

#set batch_size = None to onehot encode the entire dataset without changing the order
def get_onehot(pairs, batch_size, seq_len=1500, rand_start=False):
	aminoAcids=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
	aa_dict = dict()
	for i in range(len(aminoAcids)):
		aa_dict[aminoAcids[i]] = i

	num_classes = 30

	sample = random.sample(pairs, batch_size) if batch_size is not None else pairs
	size = len(sample)	

	xData=np.zeros((size,seq_len,len(aminoAcids)), dtype=np.int8)
	yData=np.zeros((size,num_classes), dtype=np.int8)
	for i in range(size):
	    y=sample[i][1]
	    if y < num_classes:
	    	yData[i,y] = 1
	    seq = sample[i][0]
	    counter=0
	    start=0
	    if rand_start and len(seq) > seq_len:
		start = random.randint(0, len(seq)-seq_len)
	    for c in seq[start:]:
	        xData[i,counter,aa_dict[c]] = 1
	        counter=counter+1
		if counter == seq_len:
		    break
	    if counter == 0:
		print "empty"
	
	return xData, yData

