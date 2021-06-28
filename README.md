# AI2021_Music-Genre-Classification

1. report
2. Steps to reproduce:
	0. download the GTZAN dataset, the dataset is expected to be stored in `./datasets/GTZAN` and have 10 subdirectories ONLY:
		```
		.
		├── datasets
		│   └── GTZAN
		│       ├── blues
		│       ├── classical
		│       ├── country
		│       ├── disco
		│       ├── hiphop
		│       ├── jazz
		│       ├── metal
		│       ├── pop
		│       ├── reggae
		│       └── rock
		...
		```
	1. `python3 feature_extraction.py`
		- Extracts mel-spectrograms as feature from `.wav` files in GTZAN.
		- The extracted features and labels will be under `./feats`.
		- *You may be asked to install `librosa` or `tensorflow` at this step. Just install what is needed.*
	2. `python3 CNN_5fold.py`
		- Splits the features into 5 folds
		- Trains 5 CNN models, each is trained on 80% of the whole dataset.
		- Stores the models under `./models` (provided)
	3. `python3 ensemble.py`
		- Ensembles the 5 models obtained from last step.
		- Uses the ensembled model to predict all clips in GTZAN.
		- Saves the result to result.csv
	4. (optional) `python3 calculate_accuracy.py`
		- Calculates the "single-label" accuracy.
