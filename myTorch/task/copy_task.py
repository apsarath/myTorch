import math
import numpy as np

class CopyDataGen(object):

    def __init__(self, num_bits = 8, min_len = 1, max_len = 20):
        
        self.num_bits = num_bits
        self.tot_bits = num_bits + 1
        self.min_len = min_len
        self.max_len = max_len

    def next(self, seqlen=None):
        
        if seqlen == None:
            seqlen = np.random.random_integers(self.min_len, self.max_len)
        x = np.zeros((2*seqlen+1,self.tot_bits), dtype="float32")
        y = np.zeros((2*seqlen+1,self.tot_bits), dtype ="float32")
        mask = np.zeros(2*seqlen+1, dtype="float32")
        data = np.random.binomial(1,0.5,size=(seqlen,self.num_bits))
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
        return output


if __name__=="__main__":

    gen = CopyDataGen()
    data = gen.next(seqlen=2)
    print data['seqlen']
    print data['x']
    print data['y']
    print data['mask']


