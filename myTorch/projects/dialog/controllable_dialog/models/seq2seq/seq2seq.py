import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class Seq2Seq(nn.Module):
    def __init__(
        self, src_emb_dim, src_vocab_size,
        src_hidden_dim, tgt_hidden_dim,
        pad_token_src, bidirectional,
        nlayers_src, nlayers_tgt, dropout_rate,
        device, pretrained_embeddings=None):
        """Initialize Language Model."""
        super(Seq2Seq, self).__init__()

        self._src_vocab_size = src_vocab_size
        self._src_emb_dim = src_emb_dim
        self._src_hidden_dim = src_hidden_dim
        self._tgt_hidden_dim = tgt_hidden_dim
        self._bidirectional = bidirectional
        self._nlayers_src = nlayers_src
        self._nlayers_tgt = nlayers_tgt
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
            self._tgt_embedding = self._src_embedding
            #self._src_emb_proj_layer = nn.Linear(self._src_emb_dim, self._src_emb_dim)
        else:
            print("Loading pretrainined embeddings into the model...")
            self._pretrained_emb_size = self._pretrained_embeddings.shape[1]
            #self._src_emb_proj_layer = nn.Linear(self._pretrained_emb_size, self._src_emb_dim)
            self._src_embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(self._pretrained_embeddings), freeze=False)
            #self._tgt_embedding = self._src_
            self._tgt_embedding = nn.Embedding(
                self._src_vocab_size,
                self._src_emb_dim,
                self._pad_token_src,
            )
            
        # Encoder GRU
        self._encoder = nn.GRU(
            self._src_emb_dim,
            self._src_hidden_dim //2 if self._bidirectional else self._src_hidden_dim,
            self._nlayers_src,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Decoder RNN
        dec_inp_dim = self._src_emb_dim + self._src_hidden_dim
        #dec_inp_dim = self._src_emb_dim
        self._decoder = nn.GRU(
            dec_inp_dim,
            self._tgt_hidden_dim,
            self._nlayers_tgt,
            batch_first=True,
        )

        # Projection layer from decoder hidden states to target language vocabulary
        self._decoder2vocab = nn.Linear(self._tgt_hidden_dim, self._src_vocab_size)

    def encode(self, input_src, src_lengths, is_training):
        # Lookup word embeddings in source and target minibatch
        
        src_emb = self._src_embedding(input_src)
        src_emb = F.dropout(src_emb, self._dropout_rate, is_training)

        # Pack padded sequence for length masking in encoder RNN (This requires sorting input sequence by length)
        src_emb = pack_padded_sequence(src_emb, src_lengths, batch_first=True)

        # Run sequence of embeddings through the encoder GRU
        _, src_h_t = self._encoder(src_emb)

        # extract the last hidden of encoder
        h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1) if self._bidirectional else src_h_t[-1]
        h_t = F.dropout(h_t, self._dropout_rate, is_training)

        return h_t

    def forward(self, input_src, src_lengths, input_tgt, is_training):
        h_t = self.encode(input_src, src_lengths, is_training)
        tgt_emb = F.dropout(self._tgt_embedding(input_tgt), self._dropout_rate, is_training)

        tgt_input = torch.cat((tgt_emb, h_t.unsqueeze(1).expand(h_t.size(0), tgt_emb.size(1), h_t.size(1))), dim=2)
        #tgt_input = F.dropout(tgt_input, self._dropout_rate, is_training)

        h_t = h_t.unsqueeze(0).expand(self._nlayers_tgt, h_t.size(0), h_t.size(1)).contiguous()

        tgt_h, _ = self._decoder(tgt_input, h_t)
        output_logits = self._decoder2vocab(tgt_h.contiguous().view(-1,tgt_h.size(2)))
        output_logits = output_logits.view(tgt_h.size(0), tgt_h.size(1), output_logits.size(1))
        return output_logits

    def decode_step(self, inputs_list, encoder_states, k, decoder_states=None):
        h_t = torch.stack(encoder_states)
        if decoder_states:
            decoder_state = torch.stack([state for state in decoder_states], dim=1)

        input_tgt = torch.stack([torch.LongTensor([ipt[-1]]).to(self._device) for ipt in inputs_list]).squeeze(1)
        tgt_emb = F.dropout(self._tgt_embedding(input_tgt), self._dropout_rate, False)

        tgt_input = torch.cat((tgt_emb, h_t), dim=1).unsqueeze(1)
        h_t = h_t.unsqueeze(0).expand(self._nlayers_tgt, h_t.size(0), h_t.size(1)).contiguous()
        
        all_t, tgt_h = self._decoder(tgt_input, decoder_state if decoder_states else h_t)
        output_logits = self._decoder2vocab(all_t.contiguous().view(-1,tgt_h.size(2)))

        logprobs = F.log_softmax(output_logits, dim=1)
        logprobs, words = logprobs.topk(k, 1)

        new_decoder_states = [tgt_h[:,i,:] for i in range(len(inputs_list))]

        return words.detach().cpu().numpy(), logprobs.detach().cpu().numpy(), encoder_states, new_decoder_states
        

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
