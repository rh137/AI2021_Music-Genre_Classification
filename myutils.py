import os
import librosa
import pandas as pd
import numpy as np

def my_train_test_split_5fold(X, y, fold_id=0):
	X_train, X_test = [], []
	y_train, y_test = [], []
	
	n = len(X)
	assert n == len(y)
	n_genre = n // 10
	for i in range(10):
		start = i * n_genre
		for j in range(n_genre):
			if j // (n_genre // 5) == fold_id:
				X_test.append(X[start + j])
				y_test.append(y[start + j])
			else:
				X_train.append(X[start + j])
				y_train.append(y[start + j])
	
	X_train, X_test = np.array(X_train), np.array(X_test)
	y_train, y_test = np.array(y_train), np.array(y_test)

	return X_train, X_test, y_train, y_test


def extract_mel_spectrogram(GTZAN_path='./datasets/GTZAN', max_size=660):
	print('Start extracting Mel-Spectrograms from audio files.')

	labels = []
	mel_specs = []

	for dir_name in os.listdir(GTZAN_path):
		if dir_name == ".DS_Store": continue
		for audio in os.scandir(GTZAN_path + '/' + dir_name):
			if os.path.basename(audio) == ".DS_Store": continue
			y_, sr_ = librosa.core.load(audio)
		
			# audio file name: {genre}.{number}.wav
			# in python: os.path.basename(audio)
			label = os.path.basename(audio).split('.')[0]
			labels.append(label)
			
			spect = librosa.feature.melspectrogram(y=y_, sr=sr_, n_fft=2048, hop_length=1024)
			spect = librosa.power_to_db(spect, ref=np.max)

			if spect.shape[1] != max_size:
				spect.resize(128, max_size, refcheck=False)

			mel_specs.append(spect)
		print(f'{dir_name} done.')
	
	X = np.array(mel_specs)
	labels = pd.Series(labels)
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
	y = labels.map(label_map).values

	return X, y

