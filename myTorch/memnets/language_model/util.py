from myTorch.memnets.language_model.lm import LanguageModel
from myTorch.utils import get_optimizer
from copy import deepcopy

def make_model_wider(model, config, device, model_idx, step, logger, logging, vocab):
    previous_layer_size = model.layer_size
    model.make_net_wider(expanded_layer_size=config.expanded_layer_size,
                         can_make_optimizer_wider=config.make_optimizer_wider,
                         use_noise=config.use_noise,
                         use_random_noise=config.use_random_noise)
    previous_optimizer_state_dict = deepcopy(model.optimizer.state_dict())
    new_layer_size = model.layer_size

    wider_model = LanguageModel(device, len(vocab), config.input_emb_size,
                              num_layers=config.num_layers, layer_size=config.expanded_layer_size,
                              cell_name=config.model, activation=config.activation,
                              output_activation="linear", layer_norm=config.layer_norm,
                              identity_init=config.identity_init, chrono_init=config.chrono_init,
                              t_max=config.bptt / 3, memory_size=config.memory_size, k=config.k,
                              use_relu=config.use_relu).to(device)

    if (config.expand_model_weights):
        # load the expanded weights
        wider_model.load_state_dict(model.state_dict())

    new_config = deepcopy(config)
    new_config.lr = new_config.new_lr
    optimizer = get_optimizer(wider_model.parameters(), config)

    log_message_value = "Model index: {}. " \
                        "Previous learning rate {}. " \
                        "New learning rate {}," \
                        "step: {}".format(
        model_idx,
        config.lr,
        new_config.lr,
        step)

    log_message_tag = "New learning rate for the optimizer"

    if config.use_tflogger:
        logger.log_text(tag=log_message_tag,
                        value=log_message_value
                        )
    logging.info(log_message_tag + ": " + log_message_value)

    if (config.make_optimizer_wider):
        # get new optimizer and do the standard bookkeeping tasks
        prev_param_state_values = list(previous_optimizer_state_dict['state'].values())
        param_names_in_new_optimizer = optimizer.state_dict()['param_groups'][0]['params']
        for index, param in enumerate(optimizer.param_groups[0]['params']):
            new_value = prev_param_state_values[index]
            new_value['exp_avg'] = new_value['exp_avg'].to(device)
            new_value['exp_avg_sq'] = new_value['exp_avg_sq'].to(device)
            optimizer.state[0][param_names_in_new_optimizer[index]] = new_value
        log_message_value = "Model index: {}. " \
                            "Previous size {}. " \
                            "New size {}," \
                            "step: {}".format(
            model_idx,
            previous_layer_size,
            new_layer_size,
            step)

        log_message_tag = "Widening Optimizer"

        if config.use_tflogger:
            logger.log_text(tag=log_message_tag,
                            value=log_message_value
                            )
        logging.info(log_message_tag + ": " + log_message_value)

    model = wider_model.to(device)

    log_message_tag = "Widening model (expand_model_weights = {})" \
        .format(config.expand_model_weights)
    log_message_value = "Model index: {}, " \
                        "Previous size {}, " \
                        "New size {}," \
                        "step: {}".format(
        model_idx,
        previous_layer_size,
        new_layer_size,
        step)

    if config.use_tflogger:
        logger.log_text(tag=log_message_tag,
                        value=log_message_value
                        )
    logging.info(log_message_tag + ": " + log_message_value)

    return model, optimizer

