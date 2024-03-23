import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from .rnn_nn import Embedding, RNN, LSTM


class RNNClassifier(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, use_lstm=True, **additional_kwargs):
        """
        Inputs:
            num_embeddings: size of the vocabulary
            embedding_dim: size of an embedding vector
            hidden_size: hidden_size of the rnn layer
            use_lstm: use LSTM if True, vanilla RNN if false, default=True
        """
        super().__init__()

        # Change this if you edit arguments
        hparams = {
            'num_embeddings': num_embeddings,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'use_lstm': use_lstm,
            **additional_kwargs
        }
        # if you do not inherit from lightning module use the following line
        self.hparams = hparams
        
        # if you inherit from lightning module, comment out the previous line and use the following line
        #self.hparams.update(hparams)
        
        ########################################################################
        # TODO: Initialize an RNN network for sentiment classification         #
        # hint: A basic architecture can have an embedding, an rnn             #
        # and an output layer                                                  #
        ########################################################################
        

        pass
        self.embedding = Embedding(num_embeddings, embedding_dim, padding_idx=0)
        
        if use_lstm:
            self.rnn = LSTM(input_size=embedding_dim, hidden_size=hidden_size)
        else:
            self.rnn = RNN(input_size=embedding_dim, hidden_size=hidden_size)

        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, sequence, lengths=None):
        """
        Inputs
            sequence: A long tensor of size (seq_len, batch_size)
            lengths: A long tensor of size batch_size, represents the actual
                sequence length of each element in the batch. If None, sequence
                lengths are identical.
        Outputs:
            output: A 1-D tensor of size (batch_size,) represents the probabilities of being
                positive, i.e. in range (0, 1)
        """
        output = None

        ########################################################################
        # TODO: Apply the forward pass of your network                         #
        # hint: Don't forget to use pack_padded_sequence if lenghts is not None#
        # pack_padded_sequence should be applied to the embedding outputs      #
        ########################################################################

        pass

        embeddings = self.embedding(sequence)
        #print("Embeddings size:", embeddings.size())


        if lengths is not None:
            packed_embeddings = pack_padded_sequence(embeddings, lengths.int().tolist(), batch_first=False)
            packed_outputs,output2 = self.rnn(packed_embeddings)
        else:
            output1,output2 = self.rnn(embeddings)

        last_output = output2[-1]

        output = self.fc(last_output)
        #print("FC output size:", output.size())
        
        output = self.sigmoid(output).squeeze()
        #print("Final output size:", output.size())


       
        

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return output
