from myutils import extract_mel_spectrogram

X, y = extract_mel_spectrogram()



import pickle

X_file = open('./feats/X', 'wb')
pickle.dump(X, X_file)
X_file.close()

y_file = open('./feats/y', 'wb')
pickle.dump(y, y_file)
y_file.close()
