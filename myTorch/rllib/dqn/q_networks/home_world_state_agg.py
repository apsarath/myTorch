import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from myTorch.memory import RNNCell, GRUCell, LSTMCell


class HomeWorldStateAgg(nn.Module):

    def __init__(self, device, obs_dim, action_dim, rnn_type="LSTM"):
        super(self.__class__, self).__init__()

        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._device = device
        self._rnn_hidden_size = 256
        self._embedding_dim = 20
        self._rnn_input_size = self._embedding_dim
        self._max_vocab_size = 100
        #self._embedding_dropout = 0.6
        self._max_num_steps = 30
        self._pad = 0

        self._rnn_type = rnn_type
        self._rnn_cell = LSTMCell if self._rnn_type== "LSTM" else GRUCell
        self._quest_cell = self._rnn_cell(self._device, input_size=self._rnn_input_size, hidden_size=self._rnn_hidden_size)
        self._rnn = { "quest": self._quest_cell }

        self._input_embedding = nn.Embedding(self._max_vocab_size, self._embedding_dim, sparse=False, padding_idx=self._pad)
        self._obs_fc1 = nn.Linear(self._max_num_steps, self._embedding_dim)
        #self._embedding_dropout = nn.Dropout(self._embedding_dropout)
        self._fc1 = nn.Linear(self._embedding_dim + self._rnn_hidden_size, 100)
        self._fc2 = nn.Linear(100, self._action_dim)
        self._hidden = None

    def reset_hidden(self, batch_size):
        self._hidden = {}
        for k in self._rnn: 
            self._hidden[k] = {}
            self._hidden[k]["h"] = torch.zeros(batch_size, self._rnn_hidden_size).to(self._device)
            if self._rnn_type == "LSTM":
                self._hidden[k]["c"] = torch.zeros(batch_size, self._rnn_hidden_size).to(self._device)

    def _encode(self, input, data_type):
        hidden_next = self._hidden[data_type]
        encoded_emb = hidden_next['h']
        input_mask = input.eq(self._pad).type(torch.cuda.FloatTensor)
        input_emb = self._input_embedding(input.type(torch.cuda.LongTensor).detach())
        #input_emb = self._embedding_dropout(input_emb)
        for i in range(self._max_num_steps):
            hidden_next = self._rnn[data_type](input_emb[:,i,:], hidden_next)
            encoded_emb = (1 - input_mask[:,i]).unsqueeze(1)*hidden_next['h'] + input_mask[:, i].unsqueeze(1)*encoded_emb
        return encoded_emb

    def forward(self, input):
        if len(input.shape) < 3:
            input = input.unsqueeze(0)
        self.reset_hidden(input.shape[0])
        self._quest_emb = self._encode(input[:,0,:self._max_num_steps], "quest")
        try:
            self._obs_emb = self._obs_fc1(input[:,1,:self._max_num_steps])
        except:
            import pdb; pdb.set_trace()
        x = F.relu(self._fc1(torch.cat((self._quest_emb, self._obs_emb), dim=1)))
        qvals = self._fc2(x)
        return qvals

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def obs_dim(self):
        return self._obs_dim

    @property
    def device(self):
        return self._device

    def get_attributes(self):
        return (self._device, self._obs_dim, self._action_dim, self._rnn_type)

    def get_params(self):
        return self.state_dict()

    def set_params(self, state_dict):
        self.load_state_dict(state_dict)

    def make_target_net(self, qnet):
        target_net = self.__class__(*qnet.get_attributes()).to(self._device)
        return target_net

if __name__=="__main__":
    x = FeedForward(50,10)
    x1 = FeedForward.make_target_net(x)
    import pdb; pdb.set_trace()

