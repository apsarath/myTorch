#!/usr/bin/env python
import tensorflow as tf
import sys
import os
#from models import lm
#from data_readers import reddit
import imp
import pdb
import time
import numpy as np
from config import main_config
from models import lm
from data_readers import reddit

flags = tf.flags
logging = tf.logging
FLAGS = flags.FLAGS


def main(_):
    config = main_config.load_config()
    start_time = time.time()

    config.gen_extra_data = False

    print("Config :")
    print(config)
    print("\n")

    reader = eval(config.data_reader)(config)
    data = reader.load_data()["raw_data"]
    # compute n-gram freq

    def get_hist(n_gram_len):
        hist = {}
        for d in data:
            if len(d) >= n_gram_len:
                for s in range(len(d) - n_gram_len):
                    n_gram = tuple(d[s:s + n_gram_len])
                    if n_gram in hist:
                        hist[n_gram] += 1
                    else:
                        hist[n_gram] = 1
        return hist
    len_vals = np.array([len(d) for d in data])
 
    
    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    tf.app.run()
