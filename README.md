# AI2021_Music-Genre-Classification

1. Report [link](https://drive.google.com/file/d/1IB67xkPpYMxm6RwDL9oGb5PX_TLUm9ek/view?usp=sharing)
2. Steps to reproduce:
	1. Download the [GTZAN dataset](http://marsyas.info/downloads/datasets.html), the dataset is expected to be stored in `./datasets/GTZAN` and have 10 subdirectories ONLY. We expect the file structures to be like:
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
	2. `python3 feature_extraction.py`
		- Extracts mel-spectrograms as features from `.wav` files in GTZAN.
		- The extracted features and labels will be stored under `./feats`.
		- *You may be asked to install* `librosa` *or* `tensorflow` *at this step. Just install what is needed.*
	3. `python3 CNN_5fold.py`
		- Splits the features into 5 folds
		- Trains 5 CNN models, each is trained on 80% of the whole dataset.
		- Stores the models under `./models` (the models which were trained 20 epoches are provided)
	4. `python3 ensemble.py`
		- Ensembles the 5 models obtained from last step.
		- Uses the ensembled model to predict all clips in GTZAN.
		- Saves the result to result.csv
	5. (optional) `python3 calculate_accuracy.py`
		- Calculates the "single-label" accuracy.
