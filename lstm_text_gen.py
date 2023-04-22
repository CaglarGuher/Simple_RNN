import torch
import os
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
import argparse
import logging
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a text generation model using LSTM")
    parser.add_argument('--input_file', type=str, default="book1.txt", help="Path to the input text file")
    parser.add_argument('--output_file', type=str, default="results.txt", help="Path to the output text file")
    parser.add_argument('--embed_size', type=int, default=512, help="Size of the word embeddings")
    parser.add_argument('--hidden_size', type=int, default=4096, help="Size of the LSTM hidden state")
    parser.add_argument('--num_layers', type=int, default=1, help="Number of LSTM layers")
    parser.add_argument('--num_epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=12, help="Batch size for training")
    parser.add_argument('--timesteps', type=int, default=50, help="Timesteps for truncated backpropagation through time")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument('--generated_word',type = int,default=0.001, help="How many word you want to generate")
    args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()])

embed_size = args.embed_size
hidden_size = args.hidden_size
num_layers = args.num_layers
num_epochs = args.num_epochs
batch_size = args.batch_size
timesteps = args.timesteps
learning_rate = args.learning_rate

class Dic(object):
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.index = 0

    def add(self,word):
        if word not in self.word_to_index:
            self.word_to_index[word] = self.index
            self.index_to_word[self.index] = word
            self.index += 1

    def __len__(self):
        return len(self.word_to_index)

class Text(object):
    def __init__(self):
        self.dictionary = Dic()

    def get_data(self, path, batch_size):
        with open(path, "r", encoding='utf-8') as f:
            token = 0
            for line in f:
                tokens = word_tokenize(line)
                token += len(tokens)
                for word in tokens:
                    self.dictionary.add(word)
        rep_tensor = torch.LongTensor(token)
        index = 0
        with open(path, "r", encoding='utf-8') as f:
            for line in f:
                tokens = word_tokenize(line)
                for word in tokens:
                    if word in self.dictionary.word_to_index:
                        rep_tensor[index] = self.dictionary.word_to_index[word]
                        index += 1
        num_batches = rep_tensor.shape[0] // batch_size
        rep_tensor = rep_tensor[:num_batches * batch_size]
        rep_tensor = rep_tensor.view(batch_size, -1)
        return rep_tensor

class TextGen(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size,num_layers) :
        super(TextGen,self).__init__()

        self.embed = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers,batch_first = True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        x = self.embed(x)
        out, (h, c) = self.lstm(x, h)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))
        out = self.linear(out)
        return out, (h, c)

corpus = Text()
rep_tensor = corpus.get_data(args.input_file, batch_size)
vocab_size = len(corpus.dictionary)
num_batches = rep_tensor.shape[1] // timesteps

model = TextGen(vocab_size, embed_size, hidden_size, num_layers)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    states = (torch.zeros(num_layers, batch_size, hidden_size),
              torch.zeros(num_layers, batch_size, hidden_size))

    for i in range(0, rep_tensor.size(1) - timesteps, timesteps):
        input = rep_tensor[:, i:i + timesteps]
        target = rep_tensor[:, (i + 1):(i + 1) + timesteps]
        outputs, _ = model(input, states)
        loss = loss_fn(outputs, target.reshape(-1))
        model.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        step = (i + 1) // timesteps

    logging.info("Epoch[{}/{}],Loss : {:.4f}".format(epoch + 1, num_epochs, loss.item()))

with torch.no_grad():
    with open(args.output_file, "w") as f:
        state = (torch.zeros(num_layers, 1, hidden_size),
                 torch.zeros(num_layers, 1, hidden_size))
        input = torch.randint(0, vocab_size, (1,)).long().unsqueeze(1)

        for i in range(args.generated_word):
            output, _ = model(input, state)
            prob = output.exp()
            word_id = torch.multinomial(prob, num_samples=1).item()

            input.fill_(word_id)

            word = corpus.dictionary.index_to_word[word_id]
            word = "\n" if word == "<eos>" else word + " "
            f.write(word)
            if (i + 1) % 10 == 0:
                logging.info("Sampled [{}/{}] words and saved to {}".format(i + 1, args.generated_word, args.output_file))