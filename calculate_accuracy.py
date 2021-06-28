import os
import librosa
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras import utils
from tensorflow.keras.utils import to_categorical

from myutils import extract_mel_spectrogram, my_train_test_split_5fold
import pickle
import csv


label_map = {
	'jazz': 0,
	'reggae': 1,
	'rock': 2,
	'blues': 3,
	'hiphop': 4,
	'country': 5,
	'metal': 6,
	'classical': 7,
	'disco': 8,
	'pop': 9,
}

labels = ['jazz', 'reggae', 'rock', 'blues', 'hiphop', 'country', 'metal', 'classical', 'disco', 'pop']

def ground_truth(y_row):
	for (i, num) in enumerate(y_row):
		if num == 1:
			return labels[i]
	return ''

def interpret(pred, thres=0.1):
	ret = {}
	for i in range(10):
		if pred[i] >= thres:
			ret[labels[i]] = pred[i]
	return ret
	

def get_model(fold=0):
	model = Sequential(name=f'CNN_{fold}')

	model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(128,660,1)))
	model.add(MaxPooling2D(pool_size=(2,4)))

	model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,4)))

	model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,4)))

	model.add(Flatten())

	model.add(Dense(64, activation='relu'))

	model.add(Dropout(0.25))

	model.add(Dense(10, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	return model


def get_feats():
	#X, y = extract_mel_spectrogram('./datasets/GTZAN')
	X_file = open('./feats/X', 'rb')
	X = pickle.load(X_file)
	X_file.close()
	
	y_file = open('./feats/y', 'rb')
	y = pickle.load(y_file)
	y_file.close()
	
	return X, y

X, y = get_feats()
scale = X.min()
X /= scale

X = X.reshape(X.shape[0], 128, 660, 1)
y = to_categorical(y, 10)



# ensemble
models = []
for fold in range(5):
	model = load_model(f'./models/model{fold}.20epochs')
	models.append(model)


preds = []
for fold in range(5):
	preds.append(models[fold].predict(X[:]))
	
pred = preds[0] + preds[1] + preds[2] + preds[3] + preds[4]
pred /= 5
print(type(pred))

def myparse(ndarr):
	ret = []
	for num in ndarr:
		ret.append(f'{num:.4f}')
	return ret

with open('result.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(['Genre', 'Clip ID'] + labels)
	for i in range(10):
		gt = ground_truth(y[i * 100])
		acc = 0
		for _, row in enumerate(pred[i*100: i*100+100]):
			itp = interpret(row)
			max_conf_genre = ''
			max_val = 0
			for key in itp:
				if itp[key] > max_val:
					max_val = itp[key]
					max_conf_genre = key
			if max_conf_genre == gt:
				acc += 1
			#print(_ % 100, gt, max_conf_genre, itp)
			new_row = [gt, str(_ % 100)] + myparse(row)
			writer.writerow(new_row)
		acc /= 100;
		print(gt, acc)

