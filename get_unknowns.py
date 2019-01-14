import csv

input_filename = '/mnt/data/sharing/prokaryote_refseq.tsv'
class_filename = '../results/class_names.csv'
output_filename = '/mnt/data/computervision/train80_val10_test10/unknowns.csv'
class_output_filename = '../results/unknown_class_names.csv'

data_size = 800000

class_dict = dict()
with open(class_filename, 'r') as infile:
	r = csv.reader(infile)
	for row in r:
		class_dict[row[1]] = True
print class_dict

unknowns = []
unknown_class_dict = dict()

with open(input_filename, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        i = 0
        for row in reader:
                if i > 0: #ignore the first line
                        x = row[3]
                        label = row[2]
			
                        if not label in class_dict:
                		unknowns.append(x)

				if not label in unknown_class_dict:
					unknown_class_dict[label] = 0
				unknown_class_dict[label] += 1

				if len(unknowns) == data_size:
					break
				
		i += 1
	print i

with open(output_filename, 'w') as outfile:
	w = csv.writer(outfile)
	for x in unknowns:
		#y is a single "unknown" class, equal to max class + 1
		w.writerow([len(class_dict), x])

with open(class_output_filename, 'w') as outfile:
	w = csv.writer(outfile)
	for key in unknown_class_dict:
		w.writerow([key, unknown_class_dict[key]])
print 'wrote ' + output_filename + ', ' + class_output_filename

