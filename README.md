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

Author

Name: (Your Name)
Course: (Your Course / University)
Assignment: Word Embedding Models using Keras

If you want, I can also:

✔ Shorten this README

✔ Add a commands-only version

✔ Align it exactly with your REPORT.pdf

Just tell me 👍

i done in vscode

That’s perfectly OK and acceptable 👍
Doing the assignment in VS Code is completely valid.

Let me tell you what to mention, what not to worry about, and how to state it properly in README / report / viva.

✅ Is doing it in VS Code allowed?

YES ✅

Your assignment requirement is:

“The entire work must be implemented strictly in Python using Keras”

✔ You used Python
✔ You used Keras (TensorFlow backend)
✔ Editor (VS Code / Colab) does not matter

✅ What to write in README.md (Small Addition)

Add this under Tools and Technologies or How to Run:

Development Environment:
- Visual Studio Code (VS Code)


Or:

The implementation was developed and executed locally using Visual Studio Code with Python 3.11 and TensorFlow (Keras).


That’s enough.

✅ What to say in the REPORT (One Line)

You can add this sentence (optional but safe):

All experiments were implemented and executed locally using Python and Keras in the Visual Studio Code environment.

✅ What to say in VIVA (If asked)

Best answer:

I implemented and ran the project locally in VS Code using Python 3.11 and TensorFlow Keras.

No further explanation needed.

❌ What NOT to worry about

❌ You do NOT need Colab if VS Code works

❌ You do NOT lose marks for editor choice

❌ You do NOT need to justify VS Code

✅ Final Checklist (You’re Safe)

✔ Code runs
✔ Outputs generated
✔ CSV and TXT files present
✔ README included
✔ REPORT written

👉 Editor choice does not affect evaluation

If you want, I can:

✔ Review your final folder structure

✔ Give a submission checklist

✔ Help zip everything correctly

Just tell me 👍

need as short document

Got it 👍
Here is a VERY SHORT, clean README.md (perfect for submission when they ask for a short document).

You can copy–paste this exactly.

CBOW and Skip-gram Word Embeddings (Keras)
Overview

This project implements CBOW and Skip-gram models using Python and Keras to learn 10-dimensional word embeddings from a single cybersecurity corpus.

Corpus

A single cybersecurity paragraph describing login monitoring, malware detection, authentication, encryption, access control, and auditing.

Steps Performed

Text preprocessing and vocabulary creation

CBOW and Skip-gram dataset generation (context window W = 4)

Model training using Keras

Learning-rate experiments (0.01, 0.001, 0.0001)

Embedding extraction

Evaluation using cosine similarity

Model Architecture
Input (vocab size)
↓
Dense(10)   ← Embedding layer
↓
Dense(vocab size)
↓
Softmax

Files Generated

vocab.txt

cbow_dataset.csv

skipgram_dataset.csv

embeddings_cbow.csv

embeddings_skipgram.csv

loss_cbow_lr_*.txt

loss_skipgram_lr_*.txt

similarity_results.txt

Evaluation

Cosine similarity was used to find Top-5 nearest words for five query words from different cybersecurity domains (login, malware, encryption, access, audit).

Tools

Python 3.11

TensorFlow (Keras)

NumPy

Visual Studio Code

Author

Name: (Your Name)

If you want it even shorter (half-page) or in one-paragraph format, tell me 👍

can u give again report like read me

Sure 👍
Here is a SHORT REPORT written in README style (simple, compact, and perfect to submit along with your code).
You can copy–paste this exactly and export it as REPORT.pdf if needed.

Short Report: CBOW and Skip-gram Word Embeddings using Keras
Objective

The objective of this project is to implement CBOW and Skip-gram word embedding models using Python and Keras and to analyze the effect of different learning rates on training convergence and embedding quality.

Corpus

A single cybersecurity paragraph describing authentication, malware detection, encryption, access control, and auditing is used as the corpus.

Dataset Construction

Text preprocessing includes lowercasing, punctuation removal, and tokenization. A vocabulary with unique word IDs is created.
Using a context window size of W = 4, CBOW and Skip-gram datasets are generated.

CBOW: Predicts a target word using surrounding context words

Skip-gram: Predicts context words using a target word

The Skip-gram dataset is larger than the CBOW dataset.

Model Description

Both models use the following Keras architecture:

Input (vocab size)
↓
Dense(10)  → Word embedding
↓
Dense(vocab size)
↓
Softmax


The hidden Dense layer learns a 10-dimensional embedding vector for each vocabulary word.

Training and Learning-Rate Experiments

Models are trained for a fixed number of epochs using the Adam optimizer and categorical cross-entropy loss.
Three learning rates are tested: 0.01, 0.001, and 0.0001.

Higher learning rate converges faster but may overfit

Moderate learning rate (0.001) provides stable convergence

Epoch-wise loss values are saved for analysis.

Embedding Evaluation

The learned embeddings are evaluated using cosine similarity.
Five query words from different cybersecurity domains (login, malware, encryption, access, audit) are selected, and the Top-5 nearest words are retrieved with similarity scores.
Results are saved in similarity_results.txt.

Tools Used

Python 3.11

TensorFlow / Keras

NumPy

Visual Studio Code

Conclusion

The project demonstrates that CBOW and Skip-gram models can successfully learn meaningful word embeddings from a cybersecurity corpus. Learning-rate experiments highlight the importance of appropriate step-size selection, and cosine similarity evaluation confirms semantic clustering of related words.

If you want, I can:

✔ Make it 1-page only

✔ Convert it to PDF-ready format

✔ Align it exactly with your evaluation rubric

Just tell me 👍
