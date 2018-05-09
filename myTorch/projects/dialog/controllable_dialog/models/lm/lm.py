import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class LanguageModel(nn.Module):
    def __init__(
        self, src_emb_dim, src_vocab_size,
        src_hidden_dim,
        pad_token_src, bidirectional=False,
        nlayers_src=1, nlayers_trg=1
    ):
        """Initialize Language Model."""
        super(LanguageModel, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.src_emb_dim = src_emb_dim
        self.src_hidden_dim = src_hidden_dim
        self.bidirectional = bidirectional
        self.nlayers_src = nlayers_src
        self.pad_token_src = pad_token_src
        
        # Word Embedding look-up table for the soruce language
        self.src_embedding = nn.Embedding(
            self.src_vocab_size,
            self.src_emb_dim,
            self.pad_token_src,
        )

        # Encoder GRU
        self.encoder = nn.GRU(
            self.src_emb_dim // 2 if self.bidirectional else self.src_emb_dim,
            self.src_hidden_dim,
            self.nlayers_src,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Projection layer from decoder hidden states to target language vocabulary
        self.decoder2vocab = nn.Linear(src_hidden_dim, src_vocab_size)

    def forward(self, input_src, src_lengths):
        # Lookup word embeddings in source and target minibatch
        src_emb = self.src_embedding(input_src)

        # Pack padded sequence for length masking in encoder RNN (This requires sorting input sequence by length)
        src_emb = pack_padded_sequence(src_emb, src_lengths, batch_first=True)
        
        # Run sequence of embeddings through the encoder GRU
        _, src_h_t = self.encoder(src_emb)

    
        


