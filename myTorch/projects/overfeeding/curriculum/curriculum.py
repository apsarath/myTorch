from copy import deepcopy


def curriculum_generator(config):
    """
        Method to generate a sequence of configs in increasing order of difficulty.
        For now, we generate the curriculum only using the `min_len` and `max_len` attributes.
    """
    min_seq_len = config.min_seq_len
    max_seq_len = config.max_seq_len
    step_seq_len = config.step_seq_len
    for seq_len in range(min_seq_len, max_seq_len + 1, step_seq_len):
        curriculum_config = deepcopy(config)
        curriculum_config.seq_len = seq_len
        yield curriculum_config
