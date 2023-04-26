import torch


class Word2Vec(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(Word2Vec, self).__init__()
        self.embedding_layer = torch.nn.Embedding(vocab_size, embedding_size)
        self.linear_1 = torch.nn.Linear(2 * embedding_size, hidden_size)
        self.linear_2 = torch.nn.Linear(hidden_size, 1)

        self.activation_fn = torch.nn.ReLU()

    def forward(self, pairs):
        embeddings = self.embedding_layer(pairs)
        concat = embeddings.flatten(start_dim=1)
        activations = self.activation_fn(self.linear_1(concat))

        return torch.nn.functional.sigmoid(self.linear_2(activations))
