import csv

def max_index(list_):
	max_val = -1
	max_idx = -1
	for i, val in enumerate(list_):
		if float(val) > max_val:
			max_val = float(val)
			max_idx = i
	return max_idx

accuracy = {
	'pop': 0,
	'metal': 0,
	'disco': 0,
	'blues': 0,
	'reggae': 0,
	'classical': 0,
	'rock': 0,
	'hiphop': 0,
	'country': 0,
	'jazz': 0,
}

with open('result.csv', 'r', newline='') as csvfile:
	reader = csv.reader(csvfile)
	header = next(reader)
	label_map = header[2:]
	for line in reader:
		label = line[0]
		prediction = line[2:]
		predicted_genre = label_map[max_index(prediction)]

		if label == predicted_genre:
			accuracy[label] += 1;

print('[Single-label Accuracy]')
print()
print('Genre       Accuracy')
print('--------------------')
for key in accuracy:
	print(f'{key:<11} {accuracy[key] / 100}')
print()
