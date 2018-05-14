import numpy as np


def lmrd_iterator(source, source_dict, batch_size, maxlen,
                   n_words_source=-1, char_level=False, rng=None):
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
    
    if char_level:
        # Make a dictionary mapping known characters to integers
        #   0 is 'unk'
        # (58 entries)
        chars = ['<unk>', ' ', '\n', '"', '(', ')',
                '#', '$', '%', '&', "'", '*', '+', ',', '-', '.', '/', ':',
                ';', '=', '?', '@',
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                'y', 'z',
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        substitutes = {'\t': ' ', "'": '"', '[': '(', ']': ')'}
        char_dict = dict(zip(chars, np.arange(len(chars))))
    
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
        word_dict = dict(zip(words, np.arange(len(words))))
    
    # Read all text files and encode characters in integer format
    def add_dir(path):
        contents_list = []
        lengths_list = []
        cum_sum = 0
        for fn in os.listdir(path):
            try:
                f = open(os.path.join(path, fn), 'rt')
                file_text = f.read()
            except:
                print("Failed to load file "
                        "{}".format(os.path.join(path, fn)))
                raise
            
            # char level
            if char_level:
                i = 0
                encoded_contents = []
                while i < len(file_text):
                    ch = str.lower(file_text[i])
                    try:
                        encoded_contents.append(char_dict[ch])
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
                encoded_contents = [word_dict[w] if w in word_dict \
                                                         else 0 for w in words]
                    
            # Strings should be sampled more frequently from longer entries.
            contents_list.append(encoded_contents)
            samples_per_entry = max(len(encoded_contents)//maxlen, 1)
            lengths_list.append(samples_per_entry)
            cum_sum += samples_per_entry
        
        weights_list = [float(l)/cum_sum for l in lengths_list]
        return contents_list, weights_list
    
    # Load files
    pos_content, pos_weights = add_dir(os.path.join(source, "pos"))
    neg_content, neg_weights = add_dir(os.path.join(source, "neg"))
            
    if rng is None:
        rng = np.random.RandomState()
    
    num_entries = len(pos_content)+len(neg_content)
    num_batches = num_entries//batch_size
    if num_entries%batch_size:
        num_batches += 1
    
    # Randomly sample strings up to maxlen length, with equal probability from
    # the set of positive sentiments and the set of negative sentiments.
    # Strings are sampled more frequently from longer entries, as long as these
    # can support unique strings of length maxlen.
    def gen():
        for i in range(num_batches):
            x = np.zeros((batch_size, maxlen), dtype=np.int32)
            m = np.zeros((batch_size, maxlen), dtype=np.uint8)
            y = np.zeros((batch_size,), dtype=np.int32)
            
            for j in range(batch_size):
                def sample_string(content, weights):
                    idx = rng.choice(len(content), p=weights)
                    char_idx = rng.randint(0, max(len(content[idx])-maxlen, 1))
                    string = content[idx][char_idx:char_idx+maxlen]
                    return string
                
                label = rng.randint(2)
                if label==0:
                    string = sample_string(pos_content, pos_weights)
                else:
                    string = sample_string(neg_content, neg_weights)
                    
                x[j, :len(string)] = string
                m[j, :len(string)] = 1
                y[j] = label
                
            yield x, m, y
            
    return gen 
