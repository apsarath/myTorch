import os
import re
import numpy as np


class lmrd_data(object):
    """
    Binary (positive/negative) sentiment analysis of IMDB movie reviews.
    
    source: directory containing `neg` and `pos` data directories
    source_dict: path to `imdb.vocab`
    batch_size: mini-batch size
    maxlen: the maximum length of sequences to sample
        (sequences up to length maxlen are sampled from longer passages)
    n_words_source: unless -1, limit the number of words to this amount
        (selected in order of most used) when iterating at word level
    char_level: whether to iterate at character level (else, word level)
    rng: numpy random number generator
    """
    def __init__(self, source, source_dict, batch_size, maxlen,
                 n_words_source=-1, char_level=False, random=False, rng=None):
        self.source = source
        self.source_dict = source_dict
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.n_words_source = n_words_source
        self.char_level = char_level
        self.random = random
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.RandomState()
    
        if char_level:
            # Make a dictionary mapping known characters to integers
            #   0 is 'unk'
            # (58 entries)
            chars = ['<unk>', ' ', '\n', '"', '(', ')',
                     '#', '$', '%', '&', "'", '*', '+', ',', '-', '.', '/',
                     ':', ';', '=', '?', '@',
                     'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                     'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                     'w', 'x', 'y', 'z',
                     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            substitutes = {'\t': ' ', "'": '"', '[': '(', ']': ')'}
            self.char_dict = dict(zip(chars, np.arange(len(chars))))
        
        else:
            # Load the word dictionary
            try:
                f = open(source_dict, 'rt')
                words = ['<unk>'] + re.split('\n', f.read().strip())
            except:
                print("Failed to load dictionary from {}".format(source_dict))
                raise
            if n_words_source > 0:
                words = words[:n_words_source]
            self.word_dict = dict(zip(words, np.arange(len(words))))
            
        # Load files
        self.pos_content = self._add_dir(os.path.join(source, "pos"))
        self.neg_content = self._add_dir(os.path.join(source, "neg"))
        
        self.num_entries = len(self.pos_content)+len(self.neg_content)
        self.num_batches = self.num_entries//self.batch_size
        if self.num_entries%self.batch_size:
            self.num_batches += 1
    
    # Read all text files and encode characters in integer format
    def _add_dir(self, path):
        contents_list = []
        for fn in os.listdir(path):
            try:
                f = open(os.path.join(path, fn), 'rt')
                file_text = f.read()
            except:
                print("Failed to load file "
                        "{}".format(os.path.join(path, fn)))
                raise
            
            # char level
            if self.char_level:
                i = 0
                encoded_contents = []
                while i < len(file_text):
                    ch = str.lower(file_text[i])
                    try:
                        encoded_contents.append(self.char_dict[ch])
                    except KeyError:
                        # Check for substitutes
                        if ch in substitutes:
                            encoded_contents.append(char_dict[substitutes[ch]])
                        
                        # Check if it's '<br />'
                        elif file_text[i:i+6]=='<br />':
                            encoded_contents.append(char_dict['\n'])
                            i = i+5 # +1 added later
                            
                        # Unknown characters are 0
                        else:
                            encoded_contents.append(0)
                        
                    i += 1
                
            # word level
            else:
                words = re.split('\W', file_text.strip())
                encoded_contents = [self.word_dict[w] if w in self.word_dict
                                                      else 0 for w in words]
                    
            # Strings should be sampled more frequently from longer entries.
            contents_list.append(encoded_contents)
        
        return contents_list
    
    def __call__(self):
        """
        Sample positive and negative sources without replacement.
        From each source, randomly sample one string up to maxlen length.
        """
        pos_indices = [(0, i) for i in range(len(self.pos_content))]
        neg_indices = [(1, i) for i in range(len(self.neg_content))]
        all_indices = np.array(pos_indices+neg_indices)
        if self.random:
            self.rng.shuffle(all_indices)
            
        for i in range(0, len(all_indices), self.batch_size):
            bs = self.batch_size
            batch_indices = all_indices[i*bs:(i+1)*bs]
            this_batch_size = len(batch_indices)
            
            x = np.zeros((self.maxlen, this_batch_size), dtype=np.int64)
            m = np.zeros((self.maxlen, this_batch_size), dtype=np.int64)
            y = np.zeros((this_batch_size,), dtype=np.int64)
            
            content = (self.pos_content, self.neg_content)
            for j, (label, idx) in enumerate(batch_indices):
                full_string = content[label][idx]
                if self.random:
                    idx_start = self.rng.randint(0,
                                        max(len(full_string)-self.maxlen, 1))
                else:
                    idx_start = 0
                string = full_string[idx_start:idx_start+self.maxlen]
                x[:len(string), j] = string
                m[:len(string), j] = 1
                y[j] = label
                
            yield x, m, y
            
    def __len__(self):
        return self.num_batches
            
    
