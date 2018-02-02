import os, sys


base_data_dir = '/Users/rick/src/ml_data'


def preprocess(input_file, output_file, flag=1, split_day=None):
	print('input_file:', input_file)
	print('output_file:', output_file)

	output = open(base_data_dir + '/' + output_file, 'w')
	test_output = None
	if flag == 1 and split_day:
		test_output = open(base_data_dir + '/test.csv', 'w')

	with open(base_data_dir + '/' + input_file, 'r') as f:
		next(f)
		for line in f:
			fields = line.split(',')

			for i in range(5, 14):
				fields[i] = str(hash(fields[i]) % 100000)
			new_line = ','.join(fields)

			print(new_line)
			if fields[2].startswith(split_day):
				if test_output:
					test_output.write(new_line)
			else:
				output.write(new_line)
	output.close()
	if test_output:
		test_output.close()




if '__main__' == __name__:
	preprocess(input_file='raw/kaggle_click_through_rate/test.csv',
	           output_file='data/kaggle_click_through_rate/test.csv',
	           flag=1,
	           split_day='141030')