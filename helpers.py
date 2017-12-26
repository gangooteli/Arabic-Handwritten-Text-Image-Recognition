import pickle
import glob, os
import cv2
import numpy as np

def prepare_arab_char_set(dict_path, num_char = '0123456789V'):
    #Manual arab_char_set
    arr = [u'\u0621', u'\u0622', u'\u0623', u'\u0624', u'\u0625' ,u'\u0626', 
            u'\u0627', u'\u0628', u'\u0629', u'\u062A', u'\u062B', u'\u062C', 
            u'\u062D', u'\u062F', u'\u0630', u'\u0631', u'\u0632', u'\u0633', 
            u'\u0634', u'\u0635', u'\u0636', u'\u0637', u'\u0638' ,u'\u0639', 
            u'\u063A', u'\u0640', u'\u0641', u'\u0642', u'\u0643', u'\u0644', 
            u'\u0645', u'\u0646', u'\u0647', u'\u0648', u'\u0649', u'\u064A',
            u'\u0650', u'\u0651', u'\u0652']
    
    arr = set(arr)
    
    # preparation of char set from dictionary of words
    with open(dict_path+'arabic_word_dict.pickle', 'rb') as handle:
        b = pickle.load(handle, encoding='utf-8')
        #b = pickle.load(handle)
    
    arabic_list  = list(b.values())
    arabic_set = set(''.join (arabic_list))
    
    #number set to remove from arabic char set
    num_set = set(num_char)
    arab_char_set = (arabic_set.union(arr)) - num_set
    
    return arab_char_set


def prepare_arab_char_dict(arab_char_set):
    arab_char2num = dict(zip(arab_char_set, range(len(arab_char_set))))
    arab_char2num['<GO>'] = len(arab_char2num)
    arab_char2num['<PAD>'] = len(arab_char2num)

    arab_num2char = dict(zip(arab_char2num.values(), arab_char2num.keys()))

    return arab_char2num, arab_num2char

def load_train(train_data_path, dict_path, image_size):
    
    path = train_data_path
    files = sorted(glob.glob(path+'/*'))
    images = []
    arabic_word = []
    
    # preparation of char set from dictionary of words
    with open(dict_path+'arabic_word_dict.pickle', 'rb') as handle:
        b = pickle.load(handle, encoding='utf-8')
        #b = pickle.load(handle)
    
    print("Reading test images")
    for fl in files:
        #print fl
        flbase = os.path.basename(fl).split('.')[0]+'.upx'
        print (flbase)
        img = cv2.imread(fl)
        img = cv2.resize(img, (image_size, image_size), cv2.INTER_LINEAR)
        images.append(img)
        print(b[flbase])
        arabic_word.append(b[flbase])
    
    ### because we're not creating a DataSet object for the test images,normalization happens here
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    print (images.shape)
    arabic_word_max_length = max([len(word) for word in arabic_word])
    
    return images, arabic_word, arabic_word_max_length


def pad_batch(inputs, arab_char2num, max_sequence_length=None):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used
    
    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix 
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active 
            time steps in each input sequence
    """
    
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    
    inputs_batch_major = np.empty(shape=[batch_size, max_sequence_length], dtype=object)
    inputs_batch_major.fill(arab_char2num['<PAD>'])
    #inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD
    
    #print (inputs_batch_major[5])
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            #print ("ele ", element)
            #inputs_batch_major[i, j] = arab_char2numY[element]
            inputs_batch_major[i, j] = element
            
    #print (inputs_batch_major[5])

    # [batch_size, max_time] -> [max_time, batch_size]
    #inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    #return inputs_time_major, sequence_lengths
    #inputs_batch_major = [arab_char2numY[i] for i in inputs_batch_major]
    
    #print (inputs_batch_major[5])
    
    return inputs_batch_major, sequence_lengths



def next_feed(train_arabic_word_inputs, train_arabic_word_targets, arab_char2num):
    #batch = next(batches)
    #encoder_inputs_, _ = pad_batch(batch)
    decoder_inputs_, _ = pad_batch(train_arabic_word_inputs, arab_char2num)
    decoder_targets_, _ = pad_batch(train_arabic_word_targets, arab_char2num)
    #print(len(decoder_targets_))
    #print(len(decoder_inputs_))
    """
    return{
        #encoder_inputs: encoder_inputs_,
        images : train_images,
        input_seqs: decoder_inputs_,
        targets_seqs: decoder_targets_,
    }
    """

    return decoder_inputs_, decoder_targets_

