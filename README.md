# Sentence-auto-complete

# Next-Word Prediction with Keras LSTM

##  Project Overview

This project implements a character-level Recurrent Neural Network (RNN) using **TensorFlow/Keras** to predict the next word in a sentence. This is the foundational concept behind modern "sentence autocompletion" systems used in mobile keyboards, search engines, and code editors.

The model is trained on a small corpus of text to learn sequential patterns in the language. Given a "seed text" (e.g., "the cat"), the model can predict the most probable word to follow (e.g., "sat").

##  Methodology & Pipeline

The model is built using a classic sequence generation pipeline:

1.  **Corpus Preparation:** A small, simple text corpus is defined in the notebook.

2.  **Text Vectorization:**
    * The `tensorflow.keras.preprocessing.text.Tokenizer` is fit on the corpus to create a word-to-index vocabulary mapping (e.g., 'the': 1, 'cat': 2).

3.  **N-Gram Sequence Generation:**
    * To create supervised training data, the corpus is converted into "n-gram" sequences. For a sentence like "The dog chased the cat":
        * `[The, dog]`
        * `[The, dog, chased]`
        * `[The, dog, chased, the]`
        * `[The, dog, chased, the, cat]`
    * This creates input/output pairs where the model learns to predict the last word from the preceding sequence.

4.  **Padding:**
    * All sequences are "pre-padded" using `pad_sequences` to ensure they have a uniform length (`max_sequence_len`) for the LSTM input layer.

5.  **Creating Inputs (X) and Labels (y):**
    * The padded sequences are split:
        * **X (Input):** All tokens *except* the last one (`input_sequences[:,:-1]`).
        * **y (Label):** The *last* token (`input_sequences[:,-1]`).
    * The labels (y) are one-hot encoded using `tf.keras.utils.to_categorical`, turning the problem into a multi-class classification.

##  Model Architecture (Keras LSTM)

A `Sequential` Keras model was built with the following layers:

1.  **Embedding Layer:**
    * `Embedding(total_words, 100, ...)`
    * This layer learns a 100-dimensional dense vector representation for each word in the vocabulary, capturing semantic relationships.

2.  **LSTM Layer:**
    * `LSTM(150)`
    * The core RNN layer with 150 units, responsible for learning the sequential context and patterns in the word vectors.

3.  **Output Layer:**
    * `Dense(total_words, activation='softmax')`
    * A fully connected layer where each neuron represents one word in the vocabulary. The `softmax` activation outputs a probability distribution for the most likely next word.

4.  **Compilation:**
    * **Loss:** `categorical_crossentropy` (standard for multi-class text classification).
    * **Optimizer:** `adam`.

##  Prediction

A `generate_text` function uses the trained model to generate new text. It takes a seed string, tokenizes and pads it, predicts the next word (`np.argmax` on the output probabilities), appends it, and repeats the process.

## Technologies & Libraries Used

* **Python**
* **Deep Learning:** TensorFlow 2.x, Keras
* **Keras Layers:** `Sequential`, `Embedding`, `LSTM`, `Dense`
* **Preprocessing:** `Tokenizer`, `pad_sequences`
* **Utilities:** NumPy
* **Environment:** Jupyter Notebook
