Speech-Detection
Description
This project focuses on detecting hate and offensive speech using machine learning and deep learning models. The dataset combines HASOC (Hate Speech and Offensive Content Identification) and Kaggle hate speech datasets to classify tweets into binary (hate vs. not hate) and multi-class (hate, offensive, profane, none) categories. The primary aim is to create an efficient classifier using models like SVM, LSTM, and DistilBERT.


Clone the repository
git clone https://github.com/your-repo-link.git
cd your-repo-folder
Install the required Python libraries:
pip install -r requirements.txt
Run the preprocessing script:
python preprocess.py
This step cleans the dataset by:

Lowercasing
Removing symbols, punctuation, and stop words
Tokenizing and lemmatizing the text
Train the model:

For Binary Classification:
python train_binary.py

For Multi-Class Classification
python train_multiclass.py
Evaluate the model:
python evaluate.py
rerequisites
Software Requirements:

Google Colab or Jupyter Notebook
Python 3.x
TensorFlow and Keras for model training
Required libraries in requirements.txt
Hardware Requirements:

Google Colab's Compute Engine (e.g., Nvidia K80 GPU, 24GB RAM)
Expected Results
Binary Classification:

Accuracy: ~87%
Precision, Recall, and F1-Score: Metrics provided after model evaluation.
Multi-Class Classification:

Macro Precision, Recall, and F1-Score for labels (Hate, Offensive, Profane, None): Provided in the results section.

Dataset Description
HASOC Dataset: 4500 tweets

Hate Speech: 18%
Offensive: 16%
Profane: 31%
Kaggle Dataset: 32,000 tweets

Labeled as Hate or Not
Combined Dataset:

40% HASOC data, 60% Kaggle data for binary classification.
Only HASOC data used for multi-class classification.
Model Information
SVM:

Vectorization: TF-IDF with unigram and character n-grams.
Best kernel: Linear.
Validation Split: 80:20.
LSTM:

Preprocessing: Word embeddings.
Input Padding: Maximum tweet sequence length normalized.
DistilBERT:

Transfer learning using pre-trained DistilBERT.
Fine-tuned on HASOC dataset with sequence padding.

Folder Structure:
/project-folder
│
├── preprocess.py         # Script for preprocessing the dataset
├── train_binary.py       # Script for training binary classification models
├── train_multiclass.py   # Script for training multi-class classification models
├── evaluate.py           # Script for model evaluation
├── requirements.txt      # List of Python dependencies
├── data/                 # Folder containing datasets
│   ├── train.csv         # Training data
│   ├── test.csv          # Test data
├── models/               # Folder to save trained models
└── README.md             # Project description and instructions
Future Work
Future work could focus on:

Integrating multilingual datasets to detect hate speech across languages.
Employing ensemble methods to combine the strengths of multiple classifiers.
Using explainability tools like SHAP or LIME to better understand model predictions.
Expanding the dataset with more diverse and real-world examples of hate speech to improve generalization.
