# Generative AI

## 📌 Project Title
Word Embedding Generation using CBOW and Skip-Gram Models

This project implements Continuous Bag of Words (CBOW) and Skip-gram word embedding models using Python and Keras (TensorFlow backend).
A single cybersecurity paragraph is treated as the corpus. From this corpus, datasets are generated, models are trained with different learning rates, and the learned embeddings are evaluated using cosine similarity.

The goal is to learn 10-dimensional word embeddings and analyze how learning rate affects convergence and embedding quality.

Corpus Used

A single cybersecurity-related paragraph describing:

Login monitoring

Malware detection

Authentication

Encryption

Access control

Auditing and compliance

This paragraph is treated as one corpus, which is sufficient to demonstrate CBOW and Skip-gram learning.

Project Structure
cyber_corpus/
│
├── dataset_builder.py        # Preprocessing + CBOW & Skip-gram dataset generation
├── train_cbow.py             # CBOW model training with learning-rate experiments
├── train_skipgram.py         # Skip-gram model training with learning-rate experiments
├── vocab.txt                 # Vocabulary (word, id, frequency)
├── cbow_dataset.csv          # CBOW training dataset
├── skipgram_dataset.csv      # Skip-gram training dataset
├── embeddings_cbow.csv       # Learned CBOW embeddings
├── embeddings_skipgram.csv   # Learned Skip-gram embeddings
├── loss_cbow_lr_*.txt        # Epoch-wise CBOW loss logs
├── loss_skipgram_lr_*.txt    # Epoch-wise Skip-gram loss logs
├── similarity_results.txt    # Cosine similarity nearest-neighbour results
└── README.md                 # Project documentation

Steps Performed
1. Text Preprocessing

Converted text to lowercase

Removed punctuation

Tokenized text into words

Built vocabulary with unique word IDs

Saved to:

vocab.txt

2. Dataset Generation

Context window size: W = 4

CBOW Dataset

Input: Context words

Output: Target word

One training sample per target word

Saved to:

cbow_dataset.csv

Skip-gram Dataset

Input: Target word

Output: Context words

Multiple training samples per word

Saved to:

skipgram_dataset.csv

3. Model Architecture (CBOW & Skip-gram)

Both models use the same Keras architecture:

Input (vocab size)
↓
Dense(10)        ← Word embedding layer
↓
Dense(vocab size)
↓
Softmax


Embedding dimension: 10

Loss function: Categorical Cross-Entropy

Optimizer: Adam

4. Learning-Rate Experiments

Models were trained using three learning rates:

0.01

0.001

0.0001

Epoch-wise training loss was saved for analysis.

Files generated:

loss_cbow_lr_0.01.txt
loss_cbow_lr_0.001.txt
loss_cbow_lr_0.0001.txt
loss_skipgram_lr_0.01.txt
loss_skipgram_lr_0.001.txt
loss_skipgram_lr_0.0001.txt

5. Embedding Export

After training, embeddings were extracted from the hidden Dense layer.

Saved as:

embeddings_cbow.csv
embeddings_skipgram.csv


Each row represents a 10-dimensional embedding vector for one vocabulary word.

6. Embedding Evaluation (Cosine Similarity)

Five query words from different cybersecurity domains were selected:

login

malware

encryption

access

audit

For each query word, Top-5 nearest neighbours were computed using cosine similarity.

Results saved to:

similarity_results.txt

Tools and Technologies

Python 3.11

TensorFlow / Keras

NumPy

How to Run
Option 1: Google Colab (Recommended)

Upload all files to Colab

Run:

!python dataset_builder.py
!python train_cbow.py
!python train_skipgram.py

Option 2: Local System

Make sure TensorFlow is installed:

python -m pip install tensorflow numpy


Then run:

python dataset_builder.py
python train_cbow.py
python train_skipgram.py

Conclusion

This project demonstrates how CBOW and Skip-gram models can be implemented using Keras to learn meaningful word embeddings from a cybersecurity corpus. Learning-rate experiments highlight convergence behavior, and cosine similarity evaluation confirms that semantically related words are embedded close to each other.

