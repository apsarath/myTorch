import math
import numpy as np
import cPickle as pickle

class CopyDataGen(object):

    def __init__(self, num_bits = 8, min_len = 1, max_len = 20):
        
        self.num_bits = num_bits
        self.tot_bits = num_bits + 1
        self.min_len = min_len
        self.max_len = max_len
        self.ex_seen = 0
        self.rng = np.random.RandomState(5)

    def next(self, seqlen=None):
        
        if seqlen == None:
            seqlen = self.rng.random_integers(self.min_len, self.max_len)
        x = np.zeros((2*seqlen+1,self.tot_bits), dtype="float32")
        y = np.zeros((2*seqlen+1,self.tot_bits), dtype ="float32")
        mask = np.zeros(2*seqlen+1, dtype="float32")
        data = self.rng.binomial(1,0.5,size=(seqlen,self.num_bits))
        x[0:seqlen,0:-1] = data
        x[seqlen,-1] = 1
        y[seqlen+1:,0:-1] = data
        mask[seqlen+1:] = 1
        output = {}
        output['x'] = x
        output['y'] = y
        output['mask'] = mask
        output['seqlen'] = seqlen
        output['datalen'] = 2*seqlen+1
        self.ex_seen += 1
        return output

    def save(self, fname):

        state = {}
        state["num_bits"] = self.num_bits
        state["tot_bits"] = self.tot_bits
        state["min_len"] = self.min_len
        state["max_len"] = self.max_len
        state["rng_state"] = self.rng.get_state()
        state["ex_seen"] = self.ex_seen
        pickle.dump(state, open(fname,"wb"))

    def load(self, fname):

        state = pickle.load(open(fname,"rb"))
        self.num_bits = state["num_bits"]
        self.tot_bits = state["tot_bits"]
        self.min_len = state["min_len"]
        self.max_len = state["max_len"]
        self.rng.set_state(state["rng_state"])
        self.ex_seen = state["ex_seen"]



if __name__=="__main__":

    gen = CopyDataGen()
    data = gen.next(seqlen=2)
    print data['seqlen']
    print data['x']
    print data['y']
    print data['mask']


