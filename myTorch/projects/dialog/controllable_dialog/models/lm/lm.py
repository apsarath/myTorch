import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class LanguageModel(nn.Module):
    def __init__(
        self, src_emb_dim, src_vocab_size,
        src_hidden_dim,
        pad_token_src, bidirectional=False,
        nlayers_src=1):
        """Initialize Language Model."""
        super(LanguageModel, self).__init__()

        self._src_vocab_size = src_vocab_size
        self._src_emb_dim = src_emb_dim
        self._src_hidden_dim = src_hidden_dim
        self._bidirectional = bidirectional
        self._nlayers_src = nlayers_src
        self._pad_token_src = pad_token_src
        
        # Word Embedding look-up table for the soruce language
        self._src_embedding = nn.Embedding(
            self._src_vocab_size,
            self._src_emb_dim,
            self._pad_token_src,
        )

        # Encoder GRU
        self._encoder = nn.GRU(
            self._src_emb_dim // 2 if self._bidirectional else self._src_emb_dim,
            self._src_hidden_dim,
            self._nlayers_src,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Projection layer from decoder hidden states to target language vocabulary
        self._decoder2vocab = nn.Linear(src_hidden_dim, src_vocab_size)

    def forward(self, input_src, src_lengths):
        # Lookup word embeddings in source and target minibatch
        src_emb = self._src_embedding(input_src)

        # Pack padded sequence for length masking in encoder RNN (This requires sorting input sequence by length)
        #src_emb = pack_padded_sequence(src_emb, src_lengths, batch_first=True)
        
        # Run sequence of embeddings through the encoder GRU
        all_h, src_h_t = self._encoder(src_emb)

        output_logits = F.relu(self._decoder2vocab(all_h.contiguous().view(-1,all_h.size(2))))

        output_logits = output_logits.view(all_h.size(0), all_h.size(1), output_logits.size(1))
        return output_logits

    def register_optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def optimizer(self):
        return self._optimizer

    def save(self, save_dir):
        """Saves the model and the optimizer.
        Args:
            save_dir: absolute path to saving dir.
        """

        file_name = os.path.join(save_dir, "model.p")
        torch.save(self.state_dict(), file_name)

        file_name = os.path.join(save_dir, "optim.p")
        torch.save(self.optimizer.state_dict(), file_name)

    def load(self, save_dir):
        """Loads the model and the optimizer.
        Args:
            save_dir: absolute path to loading dir.
        """

        file_name = os.path.join(save_dir, "model.p")
        self.load_state_dict(torch.load(file_name))

        file_name = os.path.join(save_dir, "optim.p")
        self.optimizer.load_state_dict(torch.load(file_name))

    @property
    def num_parameters(self):
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params
