from typing import Dict

import torch
from torch.nn import Embedding

import torch 
import torch.nn as nn
import torch.nn.functional as F

DROPOUT = 0.2
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)

        self.lstm = torch.nn.LSTM(input_size=self.embed.embedding_dim, 
                                hidden_size=hidden_size, 
                                num_layers=num_layers, 
                                bidirectional=bidirectional, 
                                dropout=dropout,
                                batch_first=True)
        
        self.fc = torch.nn.Linear(hidden_size * 2, num_class)
        self.act = torch.nn.Sigmoid()

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch, length):
        embedded = self.embed(batch)
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, length, batch_first=True)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)

        dense_outputs=self.fc(hidden)
        outputs=self.act(dense_outputs)
        
        return outputs

class TaggingClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        max_len: int,
    ) -> None:
        super(TaggingClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)

        self.lstm = torch.nn.LSTM(input_size=self.embed.embedding_dim, 
                                hidden_size=hidden_size, 
                                num_layers=num_layers, 
                                bidirectional=bidirectional, 
                                dropout=dropout,
                                batch_first=True)
        
        self.fc = torch.nn.Linear(hidden_size*2, num_class)
        self.act = torch.nn.Sigmoid()
        self.num_class = num_class
        self.max_len = max_len

    def forward(self, batch):
        embedded = self.embed(batch)
        max_length = [self.max_len for _ in range(len(batch))]
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, max_length, batch_first=True)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        dense_outputs=self.fc(packed_output.data)
        outputs=self.act(dense_outputs)
        # outputs=outputs.view((batch.shape[0], batch.shape[1], outputs.shape[1]))

        return outputs


class SlotRNN(torch.nn.Module):
    def __init__(self, embeddings, hidden_size, n_classes, bidirectional=False):
        super(SlotRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.bidirectional = bidirectional
        self.embedding = Embedding.from_pretrained(embeddings, freeze=False)
        self.dropout = torch.nn.Dropout(p=0.01)

        self.rnn = torch.nn.GRU(self.embedding.embedding_dim, hidden_size=self.hidden_size,
                          bidirectional=self.bidirectional,
                          num_layers=2, batch_first=True)

        if bidirectional == True:
            self.linear = torch.nn.Linear(hidden_size*2, n_classes)
        elif bidirectional == False:
            self.linear = torch.nn.Linear(hidden_size, n_classes)

    def forward(self, input):
        input_embedding = self.embedding(input)
        x = F.dropout(input_embedding, DROPOUT) 
        rnn_out, _ = self.rnn(x, None)
        rnn_out = F.dropout(rnn_out, DROPOUT) 
        affine_out = self.linear(torch.squeeze(rnn_out, 0))

        return torch.nn.functional.log_softmax(affine_out, dim=1)


