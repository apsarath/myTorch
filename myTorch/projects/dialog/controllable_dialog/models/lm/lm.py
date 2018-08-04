import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class LM(nn.Module):
    def __init__(
        self, src_emb_dim, src_vocab_size,
        src_hidden_dim,
        pad_token_src, bidirectional,
        nlayers_src, dropout_rate,
        device, pretrained_embeddings=None):
        """Initialize Language Model."""
        super(LM, self).__init__()

        self._src_vocab_size = src_vocab_size
        self._src_emb_dim = src_emb_dim
        self._src_hidden_dim = src_hidden_dim
        self._bidirectional = bidirectional
        self._nlayers_src = nlayers_src
        self._pad_token_src = pad_token_src
        self._dropout_rate = dropout_rate
        self._device = device
        self._pretrained_embeddings = pretrained_embeddings
        
        # Word Embedding look-up table for the soruce
        if self._pretrained_embeddings is None:
            self._src_embedding = nn.Embedding(
                self._src_vocab_size,
                self._src_emb_dim,
                self._pad_token_src,
            )
            #self._src_emb_proj_layer = nn.Linear(self._src_emb_dim, self._src_emb_dim)
        else:
            print("Loading pretrainined embeddings into the model...")
            self._pretrained_emb_size = self._pretrained_embeddings.shape[1]
            #self._src_emb_proj_layer = nn.Linear(self._pretrained_emb_size, self._src_emb_dim)
            self._src_embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(self._pretrained_embeddings), freeze=False)
            
        # Encoder GRU
        self._encoder = nn.GRU(
            self._src_emb_dim,
            self._src_hidden_dim //2 if self._bidirectional else self._src_hidden_dim,
            self._nlayers_src,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Projection layer from decoder hidden states to target language vocabulary
        self._encoder2vocab = nn.Linear(self._src_hidden_dim, self._src_vocab_size)

    def encode(self, input_src, src_lengths, is_training):
        # Lookup word embeddings in source and target minibatch
        
        src_emb = self._src_embedding(input_src)
        src_emb = F.dropout(src_emb, self._dropout_rate, is_training)

        # Pack padded sequence for length masking in encoder RNN (This requires sorting input sequence by length)
        #src_emb = pack_padded_sequence(src_emb, src_lengths, batch_first=True)

        # Run sequence of embeddings through the encoder GRU
        all_t, src_h_t = self._encoder(src_emb)
        return all_t

    def forward(self, input_src, src_lengths, is_training):
        h_t = self.encode(input_src, src_lengths, is_training)

        output_logits = self._encoder2vocab(h_t.contiguous().view(-1,h_t.size(2)))
        return output_logits

    def decode_step(self, inputs_list, k, decoder_states=None):
        if decoder_states:
            decoder_state = torch.stack([state for state in decoder_states], dim=1)
        else:
            decoder_state = None

        input_src = torch.stack([torch.LongTensor([ipt[-1]]).to(self._device) for ipt in inputs_list])
        src_input = F.dropout(self._src_embedding(input_src), self._dropout_rate, False)

        all_t, src_h = self._encoder(src_input, decoder_state)
        output_logits = self._encoder2vocab(all_t.contiguous().view(-1,src_h.size(2)))

        logprobs = F.log_softmax(output_logits, dim=1)
        logprobs, words = logprobs.topk(k, 1)

        new_decoder_states = [src_h[:,i,:] for i in range(len(inputs_list))]

        return words.detach().cpu().numpy(), logprobs.detach().cpu().numpy(), new_decoder_states
        

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
