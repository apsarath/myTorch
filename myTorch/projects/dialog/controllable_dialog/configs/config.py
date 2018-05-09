import os

class config_container(object):

    def __iter__(self):
        for attr, value in self.__dict__.items():
            if value:
                yield value

    def __getattr__(self, item):
        return None

    def _to_dict(self):
        d = {}
        for attr, value in self.__dict__.items():
            if isinstance(value, config_container):
                d[attr] = value._to_dict()
            else:
                d[attr] = value
        return d

    def __repr__(self):
        import json
        return json.dumps(self._to_dict(), indent=2)


def default_config():
    config = config_container()
    config.patience = 100
    config.gpu_percent = 100
    config.init_scale = 0.1
    config.max_grad_norm = 5
    config.keep_prob = 1.0
    config.batch_size = 32
    config.base_data_path = "/data/lisatmp4/chinna/data/OpenSubData/"#"/Users/chinna/Downloads/OpenSubData/"
    config.temperature = 1.0
    config.toy_mode = True
    config.eou = "<eou>"
    config.go = "<go>"
    config.unk = "<unk>"
    config.num = "<num>"
    config.pad = "<pad>"
    config.extra_vocab = [config.eou, config.go, config.unk, config.num]
    config.sentence_len_cut_off = 15
    config.min_sent_len = 6
    config.train_valid_split = 0.85
    config.beam_width = 10
    config.time_major = False
    config.length_penalty_weight = 0.0
    return config

def lm_config():
    config = default_config()
    config.model = "language_model.lm"
    config.rnn_hidden_size = 1500
    config.input_hidden_size = 300
    return config

def seq2seq_config():
    config = default_config()
    config.model = "seq2seq.Seq2Seq"
    config.rnn_hidden_size = 1500
    config.input_hidden_size = 300
    return config

def load_config(flags):
    if flags.model_type == "language_model":
        config = lm_config()
    elif flags.model_type == "seq2seq":
        config = seq2seq_config()
    else:
        assert(0)
    if flags.dataset == "OPUS":
        config.data_set = "opus.OPUS"
        config.vocab_size = 25000 + len(config.extra_vocab)
    else:
        assert(0)
    return config

if __name__ == '__main__':
    config = config()
    print(config)
