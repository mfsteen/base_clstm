import csv
import random
import numpy as np

random.seed(3)

#use on the output of split_data.py

def load_csv(input_path, divide=1):
	result = []
	i = 0
	with open(input_path) as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			if i % divide == 0:
				result.append((row[1], int(row[0])))
			i += 1
	return result #(x, y) pairs

#set batch_size = None to onehot encode the entire dataset without changing the order
def get_onehot(pairs, batch_size, num_classes=30, seq_len=1500, is_dna_data=False, rand_start=False):
	letters=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
	if is_dna_data:
		letters = ['A','C','G','T']
	aa_dict = dict()
	for i in range(len(letters)):
		aa_dict[letters[i]] = i

	sample = random.sample(pairs, batch_size) if batch_size is not None else pairs
	size = len(sample)	

	xData=np.zeros((size,seq_len,len(letters)), dtype=np.int8)
	yData=np.zeros((size,num_classes), dtype=np.int8)
	
	total_chars = 0
	unknown_chars = 0

	for i in range(size):
	    y=sample[i][1]
	    if y < num_classes:
	    	yData[i,y] = 1
	    seq = sample[i][0]
	    total_chars += len(seq)
	    counter=0
	    start=0
	    if rand_start and len(seq) > seq_len:
		start = random.randint(0, len(seq)-seq_len)
	    for c in seq[start:]:
		if c in aa_dict:
	        	xData[i,counter,aa_dict[c]] = 1
		else:
			unknown_chars += 1
	        counter=counter+1
		if counter == seq_len:
		    break
	    if counter == 0:
		print "empty"
	print total_chars, unknown_chars
	
	return xData, yData

