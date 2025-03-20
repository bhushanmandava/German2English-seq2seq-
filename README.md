# German to English Translation with Seq2Seq Model

## Project Overview

This project implements a sequence-to-sequence (seq2seq) model using PyTorch to translate German sentences to English.  The model is trained and evaluated on the Multi30k dataset, utilizing libraries like `torch`, `spacy`, `datasets`, `torchtext`, and `evaluate`. The notebook covers data loading, preprocessing, model definition, training, and evaluation.

## Requirements

To run this project, you'll need Python 3.6+ and the following libraries. It's highly recommended to use a virtual environment.

- torch
- torch.nn
- torch.optim
- random
- numpy
- spacy
- datasets
- torchtext
- tqdm
- evaluate

You can install these packages using pip:

```
pip install torch spacy datasets torchtext tqdm evaluate numpy pandas
```

Specific versions used in development:
- Python: 3.10
- spacy: 3.8.4

Additionally, you'll need to download the spaCy models for English and German:

```
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

## Setup

1.  Clone this repository.
2.  Create a virtual environment (recommended):
    ```
    python -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    .venv\Scripts\activate  # On Windows
    ```
3.  Install the required packages using pip as shown above.
4.  Download the required spaCy models.

## Project Structure

The core of this project is the Jupyter Notebook `main.ipynb`.  It contains the complete workflow:

1.  **Import Libraries**: Imports all necessary libraries and sets the random seed for reproducibility.
2.  **Data Loading**: Loads the Multi30k dataset using the `datasets` library.
3.  **Tokenization**: Defines a tokenization function using `spacy` to process the English and German sentences.
4.  **Vocabulary Creation**: Creates vocabulary objects using `torchtext` to map tokens to numerical indices. Special tokens such as ``, ``, ``, and `` are included.
5.  **Numericalization**: Converts the tokenized sentences into numerical IDs using the created vocabularies.
6.  **Data Loaders**: Creates data loaders using `torch.utils.data.DataLoader` to efficiently load batches of data during training.  Padding is applied to ensure all sequences in a batch have the same length.
7.  **Model Definition**: Defines the Encoder, Decoder, and Seq2Seq models.
    *   **Encoder**: An LSTM-based encoder that processes the source (German) sentence and produces the context vectors (hidden and cell states).
    *   **Decoder**: An LSTM-based decoder that generates the target (English) sentence, conditioned on the context vectors from the encoder.
    *   **Seq2Seq**: Combines the encoder and decoder into a complete sequence-to-sequence model. Includes teacher forcing during training.
8.  **Model Initialization**: Initializes the encoder, decoder, and seq2seq models.  Weights are initialized from a uniform distribution.
9. **Training**: Trains the seq2seq model.
10. **Evaluation**: Evaluates the trained model.

## Key Components

### 1. Data Preprocessing

*   **Tokenization:**  The `tokenize_example` function uses `spacy` to tokenize the German and English sentences.  It also lowercases the tokens (if specified) and adds `` (start of sentence) and `` (end of sentence) tokens.
*   **Vocabulary Creation:** The `torchtext.vocab.build_vocab_from_iterator` function creates vocabulary objects for both German and English.  It filters out tokens that appear less than `min_freq` times and adds special tokens (``, ``, ``, ``).
*   **Numericalization:** The `numericalize_example` function converts the tokenized sentences into numerical IDs using the vocabularies.
*   **Padding:** The `get_collate_fn` function pads the sequences within a batch to have the same length using `nn.utils.rnn.pad_sequence`. The `` token is used for padding.

### 2. Model Architecture

*   **Encoder:**  Consists of an embedding layer, an LSTM layer, and a dropout layer.  It takes the source sentence as input and returns the hidden and cell states.
*   **Decoder:**  Consists of an embedding layer, an LSTM layer, a linear layer, and a dropout layer. It takes the current input token, the previous hidden state, and the previous cell state as input and returns the predicted output and the updated hidden and cell states.
*   **Seq2Seq:** Takes the source sentence and the target sentence as input.  It uses the encoder to get the context vectors and then uses the decoder to generate the translated sentence.  Teacher forcing is used during training to improve the model's convergence.

### 3. Training

*   **Teacher Forcing:** During training, teacher forcing is used. With a certain probability (`teaching_force_ratio`), the true target token is used as the next input to the decoder; otherwise, the decoder's prediction is used. This helps the model converge faster.
*   **Loss Function:** This will depend on the completed code, but typical is `CrossEntropyLoss`.
*   **Optimizer:** Also depend on the rest of the code, but typical is `Adam`.

### 4. Evaluation

*   The trained model is evaluated on the validation and test datasets.  Metrics such as BLEU score (using the `evaluate` library) are commonly used to assess the translation quality.

## Usage

1.  Open the `main.ipynb` notebook in Jupyter Notebook or JupyterLab.
2.  Run all cells sequentially to execute the entire pipeline.
3.  Examine the training and validation loss curves to monitor the model's performance.
4.  Review the BLEU score on the test set to evaluate the final translation quality.

## Device

The script automatically detects and utilizes the best available device (`mps`, `cuda`, or `cpu`).

## Notes

*   Ensure that you have enough memory to load and process the dataset, especially when working with larger batch sizes or vocabulary sizes.
*   Experiment with different hyperparameters (e.g., embedding dimension, hidden dimension, number of layers, dropout rate, learning rate, batch size) to optimize the model's performance.
*   Consider implementing attention mechanisms to improve the translation quality, particularly for longer sentences.

## Future Work

*   Implement attention mechanisms to improve translation quality.
*   Experiment with different encoder and decoder architectures (e.g., Transformers).
*   Add beam search to the decoding process.
*   Implement more advanced evaluation metrics.
*   Create a simple web interface for users to input German sentences and get English translations.

