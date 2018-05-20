import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class Seq2Act2Seq(nn.Module):
    def __init__(
        self, src_emb_dim, src_vocab_size, num_attributes,
        num_acts, act_emb_dim, act_layer_dim,
        src_hidden_dim, tgt_hidden_dim,
        pad_token_src, bidirectional,
        nlayers_src, nlayers_tgt, dropout_rate):
        """Initialize Language Model."""
        super(Seq2Act2Seq, self).__init__()

        self._src_vocab_size = src_vocab_size
        self._num_attributes = num_attributes
        self._num_acts = num_acts
        self._src_emb_dim = src_emb_dim
        self._act_emb_dim = act_emb_dim
        self._act_layer_dim = act_layer_dim
        self._src_hidden_dim = src_hidden_dim
        self._tgt_hidden_dim = tgt_hidden_dim
        self._bidirectional = bidirectional
        self._nlayers_src = nlayers_src
        self._nlayers_tgt = nlayers_tgt
        self._pad_token_src = pad_token_src
        self._dropout_rate = dropout_rate
        
        # Word Embedding look-up table for the soruce
        self._src_embedding = nn.Embedding(
            self._src_vocab_size,
            self._src_emb_dim,
            self._pad_token_src,
        )

        self._act_embedding = []
        for i in range(self._num_attributes):
            self._act_embedding.append(nn.Embedding(
                                            self._num_acts,
                                            self._act_emb_dim))

        self._list_of_modules = nn.ModuleList(self._act_embedding)
        
        # Word Embedding look-up table for the target
        #self._tgt_embedding = nn.Embedding(
        #    self._src_vocab_size,
        #    self._src_emb_dim,
        #    self._pad_token_src,
        #)

        # Encoder GRU
        self._encoder = nn.GRU(
            self._src_emb_dim,
            self._src_hidden_dim //2 if self._bidirectional else self._src_hidden_dim,
            self._nlayers_src,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Decoder RNN
        dec_inp_dim = self._src_emb_dim + self._src_hidden_dim + self._act_emb_dim*self._num_attributes
        #dec_inp_dim = self._src_emb_dim
        self._decoder = nn.GRU(
            dec_inp_dim,
            self._tgt_hidden_dim,
            self._nlayers_tgt,
            batch_first=True,
        )

        # Projection layer from decoder hidden states to target language vocabulary
        self._decoder2vocab = nn.Linear(self._tgt_hidden_dim, self._src_vocab_size)

        # Curr act prediction.
        self._l1_curr = nn.Linear(self._src_hidden_dim, self._act_layer_dim)
        self._l2curr_act = nn.Linear(self._act_layer_dim, self._num_acts*self._num_attributes)

        #Next act prediction
        self._l1_next = nn.Linear(self._src_hidden_dim + self._act_emb_dim*self._num_attributes, self._act_layer_dim)
        self._l2next_act = nn.Linear(self._act_layer_dim, self._num_acts*self._num_attributes)

    def forward(self, input_src, src_lengths, input_tgt, input_acts, is_training):
        # Lookup word embeddings in source and target minibatch
        src_emb = F.dropout(self._src_embedding(input_src), self._dropout_rate, is_training)
        tgt_emb = F.dropout(self._src_embedding(input_tgt), self._dropout_rate, is_training)
        curr_act_emb = []
        for i, input_act in enumerate(input_acts):
            curr_act_emb.append(F.dropout(self._act_embedding[i](input_act), self._dropout_rate, is_training))
        curr_act_emb = torch.cat(curr_act_emb, dim=1)

        # Pack padded sequence for length masking in encoder RNN (This requires sorting input sequence by length)
        src_emb = pack_padded_sequence(src_emb, src_lengths, batch_first=True)
        
        # Run sequence of embeddings through the encoder GRU
        _, src_h_t = self._encoder(src_emb)

        # extract the last hidden of encoder
        h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1) if self._bidirectional else src_h_t[-1]
        h_t = F.dropout(h_t, self._dropout_rate, is_training)

        #curr_act prediction
        curr_act_logits = torch.chunk(
            self._l2curr_act(F.relu(self._l1_curr(h_t))),
            self._num_attributes,
            dim=1)
                                

        #next act prediction
        next_act_logits = torch.chunk(
            self._l2next_act(F.relu(self._l1_next(torch.cat((curr_act_emb, h_t),dim=1)))),
            self._num_attributes,
            dim=1)

        #tgt_input = tgt_emb
        tgt_input = torch.cat((
                        tgt_emb, 
                        curr_act_emb.unsqueeze(1).expand(curr_act_emb.size(0), tgt_emb.size(1), curr_act_emb.size(1)),
                        h_t.unsqueeze(1).expand(h_t.size(0), tgt_emb.size(1), h_t.size(1))), dim=2)
        h_t = h_t.unsqueeze(0).expand(self._nlayers_tgt, h_t.size(0), h_t.size(1)).contiguous()

        tgt_h, _ = self._decoder(tgt_input, h_t)
        output_logits = self._decoder2vocab(tgt_h.contiguous().view(-1,tgt_h.size(2)))
        output_logits = output_logits.view(tgt_h.size(0), tgt_h.size(1), output_logits.size(1))
        return output_logits, curr_act_logits, next_act_logits


    def decode_step(self, decoder_input, h_t, k):
        tgt_emb = F.dropout(self._src_embedding(decoder_input), self._dropout_rate, False)
        tgt_input = torch.cat((
                        tgt_emb,
                        h_t.unsqueeze(1).expand(h_t.size(0), tgt_emb.size(1), h_t.size(1))), dim=2)
        tgt_h, _ = self._decoder(tgt_input)
        output_logits = self._decoder2vocab(tgt_h.contiguous().view(-1,tgt_h.size(2)))
        output_logits = output_logits.view(tgt_h.size(0), tgt_h.size(1), output_logits.size(1))
        logprobs = log_softmax(logits, dim=1)
        logprobs, word_ids = logprobs.topk(k, 1)
        return tgt_h, logprobs, word_ids
        

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
