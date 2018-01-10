from opus import *

def make_dataset(dataset_name, config):
    if dataset_name == "opus":
        return OPUS(config)
    else:
        assert("Unsupported dataset : {}".format(dataset_name))
