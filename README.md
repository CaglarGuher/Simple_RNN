# Text Generation using LSTM

This script trains a text generation model using an LSTM (Long Short-Term Memory) network. The model reads an input text file, tokenizes it, and learns to generate text based on the input. The generated text is saved to an output file.

## Key Features:

1. Tokenizes input text using NLTK's `word_tokenize`.
2. Creates a custom dictionary to map words to indices and vice versa.
3. Defines an LSTM-based neural network model with configurable hyperparameters.
4. Trains the model using input text and CrossEntropyLoss as the loss function.
5. Generates text using the trained model and saves it to an output file.

## Usage:

To train the model, provide the necessary arguments such as input file, output file, and hyperparameters. For example:


## Command-line Arguments:

- `--input_file`: Path to the input text file (default: "book1.txt").
- `--output_file`: Path to the output text file (default: "results.txt").
- `--embed_size`: Size of the word embeddings (default: 512).
- `--hidden_size`: Size of the LSTM hidden state (default: 4096).
- `--num_layers`: Number of LSTM layers (default: 1).
- `--num_epochs`: Number of training epochs (default: 20).
- `--batch_size`: Batch size for training (default: 12).
- `--timesteps`: Timesteps for truncated backpropagation through time (default: 50).
- `--learning_rate`: Learning rate for the optimizer (default: 0.001).

After training, the model generates 1000 words and saves them to the output file specified.
