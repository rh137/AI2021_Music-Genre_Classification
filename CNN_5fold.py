import os
import librosa
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras import utils
from tensorflow.keras.utils import to_categorical

from myutils import extract_mel_spectrogram, my_train_test_split_5fold
import pickle


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


for fold in range(5):
	#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=.2)
	X_train, X_test, y_train, y_test = my_train_test_split_5fold(X, y, fold)

	scale = X_train.min()
	X_train /= scale
	X_test /= scale

	X_train = X_train.reshape(X_train.shape[0], 128, 660, 1)
	X_test = X_test.reshape(X_test.shape[0], 128, 660, 1)

	y_train = to_categorical(y_train, 10)
	y_test = to_categorical(y_test, 10)


	# CNN
	np.random.seed(23456)
	tf.random.set_seed(123)

	model = get_model(fold)
	model.summary()

	# 20 epo
	history = model.fit(X_train, y_train, batch_size=16, validation_data=(X_test, y_test), epochs=20)
	model.save(f'./models/model{fold}.20epochs')
	
	'''
	# 30 epo
	model.fit(X_train, y_train, batch_size=16, validation_data=(X_test, y_test), epochs=10)
	model.save(f'./models/model{fold}.30epochs')
	'''
	'''
	# 40 epo
	model.fit(X_train, y_train, batch_size=16, validation_data=(X_test, y_test), epochs=10)
	model.save(f'./models/model{fold}.40epochs')

	# 50 epo
	model.fit(X_train, y_train, batch_size=16, validation_data=(X_test, y_test), epochs=10)
	model.save(f'./models/model{fold}.50epochs')

	'''
	'''
	pred = model.predict(X_test)
	for _ in range(len(pred)):
	#	print(pred[0])
		print(ground_truth(y_test[_]), interpret(pred[_]))
	'''

