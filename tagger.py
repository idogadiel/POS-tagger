"""
intro2nlp, assignment 4, 2020

In this assignment you will implement a Hidden Markov model and an LSTM model
to predict the part of speech sequence for a given sentence.
(Adapted from Nathan Schneider)

"""
import csv
import math
import random
from collections import Counter
from math import log, isfinite

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# With this line you don't need to worry about the HW  -- GPU or CPU
# GPU cuda cores will be used if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# You can call use_seed with other seeds or None (for complete randomization)
# but DO NOT change the default value.
def use_seed(seed=1512021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_deterministic(True)
    # torch.backends.cudnn.deterministic = True


# torch.backends.cudnn.deterministic = True

# utility functions to read the corpus
def who_am_i():  # this is not a class method
    """Returns a ductionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name1': 'Ido Gadiel', 'id1': '200736494', 'email1': 'gadiele@post.bgu.ac.il',
            'name2': 'Niv Dudovitch', 'id2': '307955492', 'email2': 'nivdu@post.bgu.ac.il'}


def read_annotated_sentence(f):
    line = f.readline()
    if not line:
        return None
    sentence = []
    while line and (line != "\n"):
        line = line.strip()
        word, tag = line.split("\t", 2)
        sentence.append((word, tag))
        line = f.readline()
    return sentence


def load_annotated_corpus(filename):
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        sentence = read_annotated_sentence(f)
        while sentence:
            sentences.append(sentence)
            sentence = read_annotated_sentence(f)
    return sentences


def get_tuples(words):
    tuples = []
    for idx, val in enumerate(words):
        if idx < len(words) - 1:
            tuples.append((words[idx], words[idx + 1]))
    return tuples


START = "<DUMMY_START_TAG>"
END = "<DUMMY_END_TAG>"
UNK = "<UNKNOWN>"

allTagCounts = Counter()
# use Counters inside these
perWordTagCounts = {}
transitionCounts = {}
emissionCounts = {}
# log probability distributions: do NOT use Counters inside these because
# missing Counter entries default to 0, not log(0)
A = {}  # transisions probabilities
B = {}  # emmissions probabilities


def learn_params(tagged_sentences):
    """Populates and returns the allTagCounts, perWordTagCounts, transitionCounts,
    and emissionCounts data-structures.
    allTagCounts and perWordTagCounts should be used for baseline tagging and
    should not include pseudocounts, dummy tags and unknowns.
    The transisionCounts and emmisionCounts
    should be computed with pseudo tags and shoud be smoothed.
    A and B should be the log-probability of the normalized counts, based on
    transisionCounts and  emmisionCounts

    Args:
    tagged_sentences: a list of tagged sentences, each tagged sentence is a
    list of pairs (w,t), as retunred by load_annotated_corpus().

    Return:
    [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B] (a list)
    """

    global allTagCounts
    # use Counters inside these
    global perWordTagCounts
    global transitionCounts
    global emissionCounts
    # log probability distributions: do NOT use Counters inside these because
    # missing Counter entries default to 0, not log(0)
    global A  # transisions probabilities
    global B  # emmissions probabilities
    global START

    for tagged_sentence in tagged_sentences:

        # update transitionCounts
        sentence_tagged_tuples = get_tuples(tagged_sentence)
        for tagged_tuples in sentence_tagged_tuples:
            tag1 = tagged_tuples[0][1]
            tag2 = tagged_tuples[1][1]
            if tag1 not in transitionCounts:
                transitionCounts[tag1] = Counter()
            transitionCounts[tag1].update([tag2])

        for tagged_word in tagged_sentence:
            word = tagged_word[0]
            tag = tagged_word[1]

            # update allTagCounts
            allTagCounts.update([tag])

            # update perWordTagCounts
            if word not in perWordTagCounts:
                perWordTagCounts[word] = Counter()
            perWordTagCounts[word].update([tag])

            # update emissionCounts
            if tag not in emissionCounts:
                emissionCounts[tag] = Counter()
            emissionCounts[tag].update([word])

    # create A
    for tag, counter in transitionCounts.items():
        transition_entry = {}
        total = sum(counter.values())
        for inner_tag, count in counter.items():
            transition_entry[inner_tag] = log(count / total)
        A[tag] = transition_entry

    transition_entry = {}
    total = sum(allTagCounts.values())
    for inner_tag, count in allTagCounts.items():
        transition_entry[inner_tag] = log(count / total)
    A[START] = transition_entry

    # create B
    for tag, counter in emissionCounts.items():
        emission_entry = {}
        total = sum(counter.values())
        for word, count in counter.items():
            emission_entry[word] = log(count / total)
        B[tag] = emission_entry

    return [allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B]


def baseline_tag_sentence(sentence, perWordTagCounts, allTagCounts):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Each word is tagged by the tag most
    frequently associated with it. OOV words are tagged by sampling from the
    distribution of all tags.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        Return:
        list: list of pairs
    """
    tagged_sentence = []
    for word in sentence:
        if word in perWordTagCounts.keys():
            tag = perWordTagCounts[word].most_common(1)[0][0]
            tagged_sentence.append((word, tag))
        else:
            sampled_tag = random.choice(list(allTagCounts.elements()))
            tagged_sentence.append((word, sampled_tag))

    return tagged_sentence


# ===========================================
#       POS tagging with HMM
# ===========================================


def hmm_tag_sentence(sentence, A, B):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Tagging is done with the Viterby
    algorithm.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

    Return:
        list: list of pairs
    """
    global START
    global END
    dummy_tags = [START, END]
    tags = []
    current_node = viterbi(sentence, A, B)

    while current_node is not None:
        tags.append(current_node[0])
        current_node = current_node[1]

    tags = list(reversed(list(filter(lambda x: x not in dummy_tags, tags))))
    tagged_sentence = list(zip(sentence, tags))

    return tagged_sentence


def viterbi(sentence, A, B):
    """Creates the Viterbi matrix, column by column. Each column is a list of
    tuples representing cells. Each cell ("item") is a tupple (t,r,p), were
    t is the tag being scored at the current position,
    r is a reference to the corresponding best item from the previous position,
    and p is a log probability of the sequence so far).

    The function returns the END item, from which it is possible to
    trace back to the beginning of the sentence.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): the HMM emmission probabilities.

    Return:
        obj: the last item, tagged with END. should allow backtraking.

        """
    global START, END
    all_words, all_tags = set(), set()
    [all_words.update(tag[1].keys()) for tag in B.items()]
    all_tags.update(A.keys())
    all_tags.remove(START)

    # set START
    init_column = [(START, None, 0)]
    latest_column = init_column

    for word in sentence:
        new_column = []
        for current_tag in all_tags:
            max_node = None
            for previous_node in latest_column:
                prev_tag = previous_node[0]
                prev_prob = previous_node[2]
                transition_prob = A[prev_tag].get(current_tag, None)
                emission_prob = 0
                if word in all_words:
                    emission_prob = B[current_tag].get(word, None)
                if transition_prob is not None and emission_prob is not None:
                    current_prob = prev_prob + transition_prob + emission_prob
                    if max_node is None or current_prob > max_node[2]:
                        max_node = (current_tag, previous_node, current_prob)

            if max_node is not None:
                new_column.append(max_node)

        latest_column = new_column

    # set END
    max_node = max(latest_column, key=lambda x: x[2])
    v_last = END, max_node, max_node[2]
    return v_last


# a suggestion for a helper function. Not an API requirement
def retrace(end_item):
    """Returns a list of tags (retracing the sequence with the highest probability,
        reversing it and returning the list). The list should correspond to the
        list of words in the sentence (same indices).
    """


# a suggestion for a helper function. Not an API requirement
def predict_next_best(word, tag, predecessor_list):
    """Returns a new item (tupple)
    """


def joint_prob(sentence, A, B):
    """Returns the joint probability of the given sequence of words and tags under
     the HMM model.

     Args:
         sentence (pair): a sequence of pairs (w,t) to compute.
         A (dict): The HMM Transition probabilities
         B (dict): tthe HMM emmission probabilities.
     """
    missing_key_value = -100
    p = 0  # joint log prob. of words and tags
    for tuple in get_tuples(sentence):
        previous_tag = tuple[0][1]
        current_tag = tuple[1][1]
        current_word = tuple[1][0]
        transition_prob = A[previous_tag].get(current_tag, missing_key_value)
        emission_prob = B[current_tag].get(current_word, missing_key_value)
        p += transition_prob + emission_prob

    assert isfinite(p) and p < 0  # Should be negative. Think why!
    return p


# ===========================================
#       POS tagging with BiLSTM
# ===========================================

""" You are required to support two types of bi-LSTM:
    1. a vanila biLSTM in which the input layer is based on simple word embeddings
    2. a case-based BiLSTM in which input vectors combine a 3-dim binary vector
        encoding case information, see
        https://arxiv.org/pdf/1510.06168.pdf
"""


# Suggestions and tips, not part of the required API
#
#  1. You can use PyTorch torch.nn module to define your LSTM, see:
#     https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
#  2. You can have the BLSTM tagger model(s) implemented in a dedicated class
#     (this could be a subclass of torch.nn.Module)
#  3. Think about padding.
#  4. Consider using dropout layers
#  5. Think about the way you implement the input representation
#  6. Consider using different unit types (LSTM, GRU,LeRU)

def initialize_rnn_model(params_d):
    """Returns a dictionary with the objects and parameters needed to run/train_rnn
       the lstm model. The LSTM is initialized based on the specified parameters.
       thr returned dict is may have other or additional fields.

    Args:
        params_d (dict): a dictionary of parameters specifying the model. The dict
                        should include (at least) the following keys:
                        {'max_vocab_size': max vocabulary size (int),
                        'min_frequency': the occurence threshold to consider (int),
                        'input_rep': 0 for the vanilla and 1 for the case-base (int),
                        'embedding_dimension': embedding vectors size (int),
                        'num_of_layers': number of layers (int),
                        'output_dimension': number of tags in tagset (int),
                        'pretrained_embeddings_fn': str,
                        'data_fn': str
                        }
                        max_vocab_size sets a constraints on the vocab dimention.
                            If the its value is smaller than the number of unique
                            tokens in data_fn, the words to consider are the most
                            frequent words. If max_vocab_size = -1, all words
                            occuring more that min_frequency are considered.
                        min_frequency privides a threshold under which words are
                            not considered at all. (If min_frequency=1 all words
                            up to max_vocab_size are considered;
                            If min_frequency=3, we only consider words that appear
                            at least three times.)
                        input_rep (int): sets the input representation. Values:
                            0 (vanilla), 1 (case-base);
                            <other int>: other models, if you are playful
                        The dictionary can include other keys, if you use them,
                             BUT you shouldn't assume they will be specified by
                             the user, so you should spacify default values.
    Return:
        a dictionary with the at least the following key-value pairs:
                                       {'lstm': torch.nn.Module object,
                                       input_rep: [0|1]}
        #Hint: you may consider adding the embeddings and the vocabulary
        #to the returned dict

    """
    max_vocab_size = params_d['max_vocab_size']
    min_frequency = params_d['min_frequency']
    pretrained_embeddings_fn = params_d['pretrained_embeddings_fn']
    input_rep = params_d['input_rep']
    embedding_dimension = params_d['embedding_dimension']
    num_of_layers = params_d['num_of_layers']
    output_dimension = params_d['output_dimension']
    data_fn = params_d['data_fn']

    # hard coded params
    hidden_dim = 64

    all_unique_tags, all_unique_words = get_all_unique_tags_and_words(data_fn, input_rep, min_frequency, max_vocab_size)
    vocab_size = len(all_unique_words)
    tagset_size = len(all_unique_tags)

    if input_rep == 0:  # vanilla lstm
        model = LSTMTagger(embedding_dimension, hidden_dim, vocab_size, output_dimension, num_of_layers)
    else:  # case_base lstm
        model = LSTMTagger_case_base(embedding_dimension, hidden_dim, vocab_size, output_dimension, num_of_layers)

    # create word to index dictionary
    word_to_ix = {}
    for word in all_unique_words:
        if word not in word_to_ix:  # in case word has not been assigned an index yet
            word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index

    # create tags to index dictionary
    tag_to_ix = {}
    for ix, tag in enumerate(all_unique_tags):
        tag_to_ix[tag] = ix

    # load pretrained embedding vectors from path
    vectors = load_pretrained_embeddings(pretrained_embeddings_fn)

    # load pretrained embeddings to the model embedding layer:
    # create embeddings_matrix_tensor as the pretrained embedding weights:
    embeddings_matrix = np.zeros((vocab_size + 1, embedding_dimension))
    for word, i in word_to_ix.items():
        embedding_vector = vectors.get(word);
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector
    # numpy embeddings to pytorch tensor (type float)
    embeddings_matrix_tensor = torch.from_numpy(embeddings_matrix).float()

    # Assign pre-trained embedding to the word embedding layer of the model.
    model.word_embeddings_layer.weight = nn.Parameter(embeddings_matrix_tensor, requires_grad=True)
    res_dict = {'lstm': model,
                'input_rep': input_rep,
                'embeddings_matrix': embeddings_matrix,
                'words_vocabulary': word_to_ix,
                'tags_vocabulary': tag_to_ix,
                'vocab_size': vocab_size,
                'tagset_size': tagset_size}
    return res_dict


def load_pretrained_embeddings(path, vocab=None):
    """ Returns an object with the the pretrained vectors, loaded from the
        file at the specified path. The file format is the same as
        https://www.kaggle.com/danielwillgeorge/glove6b100dtxt
        You can also access the vectors at:
         https://www.dropbox.com/s/qxak38ybjom696y/glove.6B.100d.txt?dl=0
         (for efficiency (time and memory) - load only the vectors you need)
        The format of the vectors object is not specified as it will be used
        internaly in your code, so you can use the datastructure of your choice.

    Args:
        path (str): full path to the embeddings file
        vocab (list): a list of words to have embeddings for. Defaults to None.
    """

    embeddings_index = {};
    with open(path) as f:
        for line in f:
            values = line.split();
            word = values[0];
            coefs = np.asarray(values[1:], dtype='float32');
            embeddings_index[word] = coefs;
    return embeddings_index


def train_rnn(model_dict, train_data, val_data=None):
    """Trains the BiLSTM model on the specified data.

    Args:
        model (dict): the model dict as returned by initialize_rnn_model()
        train_data (list): a list of annotated sentences in the format returned
                            by load_annotated_corpus()
        val_data (list): a list of annotated sentences in the format returned
                            by load_annotated_corpus() to be used for validation.
                            Defaults to None
    """
    # hard coded params:
    max_epochs = 400
    train_percentage = 0.9  # the ratio for train-validation is (0.9,0.1) from the train original size. NOTE: used if val_data is None

    # params from dict:
    model = model_dict['lstm']
    input_rep = model_dict['input_rep']
    word_to_ix = model_dict['words_vocabulary']
    tag_to_ix = model_dict['tags_vocabulary']
    vocab_size = model_dict['vocab_size']
    tagset_size = model_dict['tagset_size']
    if input_rep == 0:
        data, val_data, max_len_sent = process_data(train_data, val_data, train_percentage)
    else:
        data, val_data, max_len_sent = process_data_case_base(train_data, val_data, train_percentage)

    criterion = nn.NLLLoss(reduction='sum', ignore_index=tagset_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # to device:
    criterion = criterion.to(device)
    model = model.to(device)
    if input_rep == 0:
        loss, acc = train_model(model, optimizer, criterion, max_epochs, data, word_to_ix, tag_to_ix, val_data,
                                max_len_sent=max_len_sent)
    else:
        loss, acc = train_model_case_base(model, optimizer, criterion, max_epochs, data, word_to_ix, tag_to_ix,
                                          val_data, max_len_sent=max_len_sent)


def rnn_tag_sentence(sentence, model_dict):
    """ Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence and t is the predicted tag.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict):  a dictionary with the trained BiLSTM model and all that is needed
                        to tag a sentence.

    Return:
        list: list of pairs
    """
    # params from dict:
    model = model_dict['lstm']
    input_rep = model_dict['input_rep']
    word_to_ix = model_dict['words_vocabulary']
    tag_to_ix = model_dict['tags_vocabulary']
    vocab_size = model_dict['vocab_size']
    tagset_size = model_dict['tagset_size']

    if input_rep == 0:
        model.eval()
        with torch.no_grad():
            # Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix, len(sentence), pad=False)
            sentence_in = torch.as_tensor(sentence_in, dtype=torch.long)
            sentence_in = sentence_in.to(device)
            sentence_in = sentence_in.unsqueeze(0)

            tag_scores = model(sentence_in)
            # Compute the loss, gradients
            tag_scores = tag_scores.view(tag_scores.shape[0] * tag_scores.shape[1], tag_scores.shape[2])

            pred_tags = torch.argmax(tag_scores, dim=1).tolist()
            tagged_sentence = list(zip(sentence, pred_tags))
    else:
        # generate capital letters feature
        cb_ohv = sentence_to_one_hot(tuple(sentence))
        # convert the sentence to lower case
        sentence = tuple([w.lower() for w in sentence])

        model.eval()
        with torch.no_grad():
            # Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix, len(sentence), pad=False)
            sentence_in = torch.as_tensor(sentence_in, dtype=torch.long)
            cb_ohv_in = prepare_sequence_case_base(cb_ohv, tag_to_ix, len(sentence))
            cb_ohv_in = torch.as_tensor([cb_ohv_in]).to(device)
            sentence_in = sentence_in.to(device)
            cb_ohv_in = cb_ohv_in.to(device)
            sentence_in = sentence_in.unsqueeze(0)
            tag_scores = model(sentence_in, cb_ohv_in)
            # Compute the loss, gradients
            tag_scores = tag_scores.view(tag_scores.shape[0] * tag_scores.shape[1], tag_scores.shape[2])

            pred_tags = torch.argmax(tag_scores, dim=1).tolist()
            tagged_sentence = list(zip(sentence, pred_tags))
    return tagged_sentence


def get_best_performing_model_params():
    """Returns a disctionary specifying the parameters of your best performing
        BiLSTM model.
        IMPORTANT: this is a *hard coded* dictionary that will be used to create
        a model and train a model by calling
               initialize_rnn_model() and train_lstm()
    """
    model_params = {'max_vocab_size': -1,
                    'min_frequency': 10,
                    'pretrained_embeddings_fn': 'load our locally stored files',
                    'input_rep': 1,
                    'num_of_layers': 5,
                    'data_fn': 'load our locally stored files',
                    'output_dimension': 17,
                    'embedding_dimension': 100
                    }

    return model_params


# ===========================================================
#       Wrapper function (tagging with a specified model)
# ===========================================================

def tag_sentence(sentence, model):
    """Returns a list of pairs (w,t) where pair corresponds to a word (same index) in
    the input sentence. Tagging is done with the specified model.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict): a dictionary where key is the model name and the value is
        an ordered list of the parameters of the trained model (baseline, HMM)
        or the model itself (LSTMs).

        Models that must be supported (you can add more):
        1. baseline: {'baseline': [perWordTagCounts, allTagCounts]}
        2. HMM: {'hmm': [A,B]}
        3. Vanilla BiLSTM: {'blstm':[Torch.nn.Module]}
        4. BiLSTM+case: {'cblstm': [Torch.nn.Module]}
        5. (NOT REQUIRED: you can add other variations, agumenting the input
            with further subword information, with character-level word embedding etc.)

        The parameters for the baseline model are:
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        The parameters for the LSTM are:
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.


    Return:
        list: list of pairs
    """
    if model == 'baseline':
        return baseline_tag_sentence(sentence, model.values()[0], model.values()[1])
    if model == 'hmm':
        return hmm_tag_sentence(sentence, model.values()[0], model.values()[1])
    if model == 'blstm':
        return rnn_tag_sentence(sentence, model.values()[0])
    if model == 'cblstm':
        return rnn_tag_sentence(sentence, model.values()[0])


def count_correct(gold_sentence, pred_sentence):
    """Return the total number of correctly predicted tags,the total number of
    correcttly predicted tags for oov words and the number of oov words in the
    given sentence.

    Args:
        gold_sentence (list): list of pairs, assume to be gold labels
        pred_sentence (list): list of pairs, tags are predicted by tagger

    """
    assert len(gold_sentence) == len(pred_sentence)
    # TODO complete the code
    correct = None
    correctOOV = None
    OOV = None

    return correct, correctOOV, OOV


"""## Auxiliary functions:"""


def prepare_sequence(seq, to_ix, max_len_sent, pad=True):
    idxs = [to_ix[w] if w in to_ix.keys() else len(to_ix.keys()) for w in seq]
    if (len(idxs) < max_len_sent) and pad:
        idxs = pad_with_zeros(idxs, max_len_sent, len(to_ix))
    return idxs  # torch.tensor(idxs, dtype=torch.long)


def pad_with_zeros(state_to_append, max_len_sent, plus_len):
    pad_vec = np.zeros(max_len_sent - len(state_to_append)) + plus_len
    return list(np.append(state_to_append, pad_vec))


def eval_model(model, loss_function, validation_data, word_to_ix, tag_to_ix, max_len_sent, batch_size):
    targets_batch_list = []
    sent_batch_list = []
    counter = 0
    number_of_batches = 0
    epoch_loss = 0
    epoch_correct_classifications = 0
    epoch_total_classifications = 0
    model.eval()
    with torch.no_grad():
        for sentence, tags in zip(*validation_data):
            # sentence = sentence.lower()
            # split to batches by batch_size
            sentence_in = prepare_sequence(sentence, word_to_ix, max_len_sent)
            sent_batch_list.append(sentence_in)
            targets = prepare_sequence(tags, tag_to_ix, max_len_sent)
            targets_batch_list.append(targets)
            if (len(sent_batch_list) + counter) != (len(validation_data[0])):
                if len(sent_batch_list) < batch_size:
                    continue
            counter += len(sent_batch_list)
            sent_batch_list = torch.FloatTensor(sent_batch_list).to(device)
            targets_batch_list = torch.FloatTensor(targets_batch_list).to(device)
            number_of_batches += 1
            tag_scores = model(sent_batch_list)
            tag_scores = tag_scores.view(tag_scores.shape[0] * tag_scores.shape[1], tag_scores.shape[2])
            targets_batch_list = torch.as_tensor(
                targets_batch_list.view(targets_batch_list.shape[0] * targets_batch_list.shape[1]), dtype=torch.long)
            loss = loss_function(tag_scores, targets_batch_list)
            # loss and acc calc:
            epoch_loss += loss
            tag_scores = torch.argmax(tag_scores, dim=1)  # , keepdim = True)
            non_pad_elements = (targets_batch_list != len(tag_to_ix))  # .nonzero()
            corrects = (tag_scores[non_pad_elements] == targets_batch_list[non_pad_elements])

            epoch_correct_classifications += sum(corrects)
            epoch_total_classifications += len(tag_scores[non_pad_elements])
            targets_batch_list = []
            sent_batch_list = []

            '''
            # Tensors of word indices.
          sentence_in = prepare_sequence(sentence, global_word_to_ix, max_len_sent)
          sentence_in = torch.as_tensor(sentence_in, dtype=torch.long)
          # Step 3. Run our forward pass.
          sentence_in = sentence_in.to(device)
          sentence_in = sentence_in.unsqueeze(-1)
  
          tag_scores = model(sentence_in)
          # Step 4. Compute the loss, gradients
          pred_tags = torch.argmax(tag_scores, dim=1).tolist()
          tagged_sentence = list(zip(sentence, pred_tags)) # {sentence[i]: pred_tags[i].item() for i in range(len(sentence))} 
            '''
    epoch_acc = epoch_correct_classifications / epoch_total_classifications
    epoch_loss = epoch_loss / epoch_total_classifications  # len(validation_data[0]) # number_of_batches
    return epoch_loss, epoch_acc


def check_stop_condition(curr_val_acc, rounds_without_improvement, best_acc, patience):
    if curr_val_acc > best_acc:
        best_acc = curr_val_acc
        rounds_without_improvement = 0
    else:
        rounds_without_improvement += 1
    if rounds_without_improvement >= patience:
        return True, best_acc, rounds_without_improvement
    return False, best_acc, rounds_without_improvement


def get_model_layers(model):
    layers = [module for module in model.modules() if type(module) != nn.Sequential]
    layers = layers[1:]
    return layers


def train_model(model, optimizer, loss_function, max_epochs, training_data, word_to_ix, tag_to_ix, validation_data,
                max_len_sent, batch_size=128):
    print('train model : \n')
    # early stopping params
    rounds_without_improvement = 0
    patience = 5
    best_acc = -1
    for epoch in range(max_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
        # return to train mode after val mode.
        model.train()
        print('epoch number : ' + str(epoch) + '\n')
        epoch_loss = 0
        epoch_correct_classifications = 0
        epoch_total_classifications = 0
        targets_batch_list = []
        sent_batch_list = []
        counter = 0
        number_of_batches = 0
        for sentence, tags in zip(*training_data):
            # split to batches by batch_size
            sentence_in = prepare_sequence(sentence, word_to_ix, max_len_sent)
            sent_batch_list.append(sentence_in)
            targets = prepare_sequence(tags, tag_to_ix, max_len_sent)
            targets_batch_list.append(targets)
            if (len(sent_batch_list) + counter) != (len(training_data[0])):
                if len(sent_batch_list) < batch_size:
                    continue
            counter += len(sent_batch_list)
            sent_batch_list = torch.FloatTensor(sent_batch_list).to(device)

            targets_batch_list = torch.FloatTensor(targets_batch_list).to(device)
            # print('targets_batch_list6')
            number_of_batches += 1
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
            tag_scores = model(sent_batch_list)
            # Step 4. Compute the loss, gradients, and update the parameters by
            tag_scores = tag_scores.view(tag_scores.shape[0] * tag_scores.shape[1], tag_scores.shape[2])
            targets_batch_list = torch.as_tensor(
                targets_batch_list.view(targets_batch_list.shape[0] * targets_batch_list.shape[1]), dtype=torch.long)
            loss = loss_function(tag_scores, targets_batch_list)
            loss.backward()
            optimizer.step()

            # loss and acc calc:
            epoch_loss += loss
            tag_scores = torch.argmax(tag_scores, dim=1)  # , keepdim = True)
            non_pad_elements = (targets_batch_list != len(tag_to_ix))  # .nonzero()
            corrects = (tag_scores[non_pad_elements] == targets_batch_list[non_pad_elements])
            epoch_correct_classifications += sum(corrects)
            epoch_total_classifications += len(tag_scores[non_pad_elements])
            targets_batch_list = []
            sent_batch_list = []

        epoch_acc = epoch_correct_classifications / epoch_total_classifications
        epoch_loss = epoch_loss / epoch_total_classifications
        print('train loss : ' + str(epoch_loss.item()) + ', train acc : ' + str(epoch_acc.item()) + '\n')
        if validation_data:
            val_loss, val_acc = eval_model(model, loss_function, validation_data, word_to_ix, tag_to_ix, max_len_sent,
                                           batch_size)
            print('validation loss : ' + str(val_loss.item()) + ', validation acc : ' + str(val_acc.item()) + '\n')
            stop_train, best_acc, rounds_without_improvement = check_stop_condition(val_acc.item(),
                                                                                    rounds_without_improvement,
                                                                                    best_acc, patience)
            if stop_train:
                print('\n Stop training contidion was met \n')
                break
    if validation_data:
        return val_loss, val_acc
    # if not specifying validation data return train performances
    return epoch_loss, epoch_acc


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, num_of_layers=2):
        super(LSTMTagger, self).__init__()
        self.num_of_layers = num_of_layers
        # self.hidden_dim = hidden_dim
        self.embedding_dimension = embedding_dim
        self.output_dimension = tagset_size
        self.input_dimension = vocab_size  # ?
        self.word_embeddings_layer = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=vocab_size)  # ?

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, num_layers=num_of_layers, batch_first=True,
                            dropout=0.2)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(2 * hidden_dim, tagset_size)

    def forward(self, sentence):
        sentence = torch.as_tensor(sentence, dtype=torch.long).to(device)
        embeds = self.word_embeddings_layer(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)  # .view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=2)
        return tag_scores


# load train set
def get_all_unique_tags_and_words(path, input_rep, min_frequency, max_vocab_size):
    train_set = pd.read_csv(path, sep='\t', header=None, engine='python', quoting=csv.QUOTE_NONE)
    train_set.columns = ['word', 'tag']
    # unique tags:
    all_unique_tags = train_set.tag.unique().astype(str)
    # process unique words:
    if input_rep == 1:
        all_unique_words = train_set.word.str.lower().astype(str)
    else:
        all_unique_words = train_set.word.astype(str)
    all_unique_words = all_unique_words.value_counts()  # unique().astype(str)
    # remove words occure less the min_frequency
    all_unique_words = all_unique_words[all_unique_words >= min_frequency]  # [:max_vocab_size]
    # keep only the max_vocab_size of most common words
    if max_vocab_size != -1:
        all_unique_words = all_unique_words[:max_vocab_size]
    all_unique_words = np.array(all_unique_words.keys())
    return all_unique_tags, all_unique_words


# case base functions:

class LSTMTagger_case_base(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, num_of_layers=2):
        super(LSTMTagger_case_base, self).__init__()
        self.num_of_layers = num_of_layers
        self.embedding_dimension = embedding_dim
        self.output_dimension = tagset_size
        self.vocab_size = vocab_size  # ?
        self.word_embeddings_layer = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=vocab_size)  # ?
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.)
        # +3 fot the casebase vector
        self.lstm = nn.LSTM(embedding_dim + 3, hidden_dim, bidirectional=True, num_layers=num_of_layers,
                            batch_first=True, dropout=0.2)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(2 * hidden_dim, tagset_size)

    def forward(self, sentence, cb):
        sentence = torch.as_tensor(sentence, dtype=torch.long).to(device)
        cb = torch.as_tensor(cb, dtype=torch.long).to(device)
        embeds = self.word_embeddings_layer(sentence)
        embeds_cb = torch.cat((embeds, cb), dim=-1)
        lstm_out, _ = self.lstm(embeds_cb)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=2)

        return tag_scores


def process_data(train_data, val_data, train_percentage):
    max_len_sent = []
    # create validation set if not given:
    if val_data is None:
        train_split_lens = math.floor(int(len(train_data) * train_percentage))
        validation_lens = len(train_data) - train_split_lens
        # create list in the size of number of sentences.

        sens_train = [None] * train_split_lens
        tags_train = [None] * train_split_lens
        sens_val = [None] * validation_lens
        tags_val = [None] * validation_lens
        for i, _ in enumerate(train_data):
            s, t = zip(*train_data[i])
            if i < train_split_lens:
                sens_train[i] = s
                tags_train[i] = t
            # add to validation data
            else:
                sens_val[i % train_split_lens] = s
                tags_val[i % train_split_lens] = t
            # find max len sentence:
            if len(list(s)) > len(max_len_sent):
                max_len_sent = list(s)

        max_len_sent = len(max_len_sent)
        data = [sens_train, tags_train]
        val_data = [sens_val, tags_val]

    else:  # if validation set is given:
        sens_train = [None] * len(train_data)
        tags_train = [None] * len(train_data)
        # process train_data
        for i, _ in enumerate(train_data):
            s, t = zip(*train_data[i])
            sens_train[i] = s
            tags_train[i] = t
            if len(list(s)) > len(max_len_sent):
                max_len_sent = list(s)
        # process val_data
        sens_val = [None] * len(val_data)
        tags_val = [None] * len(val_data)
        for i, _ in enumerate(val_data):
            s, t = zip(*val_data[i])
            sens_val[i] = s
            tags_val[i] = t
            if len(list(s)) > len(max_len_sent):
                max_len_sent = list(s)

        max_len_sent = len(max_len_sent)
        data = [sens_train, tags_train]
        val_data = [sens_val, tags_val]

    return data, val_data, max_len_sent


def get_word_one_hot_vector(word):
    one_hot_vec_word = [0] * 3
    if word.isupper():  # if the whole word is upper case
        one_hot_vec_word[0] = 1
    elif word[0].isupper():  # if just first letter is upper case
        one_hot_vec_word[1] = 1
    else:  # if the word is lower case
        one_hot_vec_word[2] = 1
    return one_hot_vec_word


def sentence_to_one_hot(s):
    sent_as_list = list(s)
    one_hot_vec_word_list = [None] * len(sent_as_list)
    for i, word in enumerate(sent_as_list):
        one_hot_vec_word_list[i] = get_word_one_hot_vector(word)
    return tuple(one_hot_vec_word_list)


def process_data_case_base(train_data, val_data, train_percentage):
    max_len_sent = []
    # create validation set if not given:
    if val_data is None:
        train_split_lens = math.floor(int(len(train_data) * train_percentage))
        validation_lens = len(train_data) - train_split_lens
        # create list in the size of number of sentences.

        sens_train = [None] * train_split_lens
        tags_train = [None] * train_split_lens
        cb_train = [None] * train_split_lens

        sens_val = [None] * validation_lens
        tags_val = [None] * validation_lens
        cb_val = [None] * validation_lens

        for i, _ in enumerate(train_data):
            s, t = zip(*train_data[i])
            if i < train_split_lens:
                # generate capital letters feature
                cb_train[i] = sentence_to_one_hot(s)
                # convert the sentence to lower case
                s = tuple([w.lower() for w in list(s)])
                sens_train[i] = s
                tags_train[i] = t
            # add to validation data
            else:
                # generate capital letters feature
                cb_val[i % train_split_lens] = sentence_to_one_hot(s)
                # convert the sentence to lower case
                s = tuple([w.lower() for w in list(s)])
                sens_val[i % train_split_lens] = s
                tags_val[i % train_split_lens] = t
            # find max len sentence:
            if len(list(s)) > len(max_len_sent):
                max_len_sent = list(s)

        max_len_sent = len(max_len_sent)
        data = [sens_train, tags_train, cb_train]
        val_data = [sens_val, tags_val, cb_val]

    else:  # if validation set is given:          word_case = get_word_one_hot_vector(word)

        # process train_data
        sens_train = [None] * len(train_data)
        tags_train = [None] * len(train_data)
        cb_train = [None] * len(train_data)
        for i, _ in enumerate(train_data):
            s, t = zip(*train_data[i])
            # generate capital letters feature
            cb_train[i] = sentence_to_one_hot(s)
            # convert the sentence to lower case
            s = tuple([w.lower() for w in list(s)])
            sens_train[i] = s
            tags_train[i] = t
            if len(list(s)) > len(max_len_sent):
                max_len_sent = list(s)
        # process val_data
        sens_val = [None] * len(val_data)
        tags_val = [None] * len(val_data)
        cb_val = [None] * len(val_data)

        for i, _ in enumerate(val_data):
            s, t = zip(*val_data[i])
            # generate capital letters feature
            cb_val[i] = sentence_to_one_hot(s)
            # convert the sentence to lower case
            s = tuple([w.lower() for w in list(s)])
            sens_val[i] = s
            tags_val[i] = t
            if len(list(s)) > len(max_len_sent):
                max_len_sent = list(s)

        max_len_sent = len(max_len_sent)
        data = [sens_train, tags_train, cb_train]
        val_data = [sens_val, tags_val, cb_val]

    return data, val_data, max_len_sent


def eval_model_case_base(model, loss_function, validation_data, word_to_ix, tag_to_ix, max_len_sent, batch_size):
    cb_batch_list = []
    counter = 0
    number_of_batches = 0
    epoch_loss = 0
    epoch_correct_classifications = 0
    epoch_total_classifications = 0
    targets_batch_list = []
    sent_batch_list = []
    cb_batch_list = []
    counter = 0
    model.eval()
    with torch.no_grad():
        for sentence, tags, cb in zip(*validation_data):
            # split to batches by batch_size
            sentence_in = prepare_sequence(sentence, word_to_ix, max_len_sent)
            sent_batch_list.append(sentence_in)
            targets = prepare_sequence(tags, tag_to_ix, max_len_sent)
            targets_batch_list.append(targets)
            cbs = prepare_sequence_case_base(cb, tag_to_ix, max_len_sent)
            cb_batch_list.append(cbs)
            if (len(sent_batch_list) + counter) != (len(validation_data[0])):
                if len(sent_batch_list) < batch_size:
                    continue
            counter += len(sent_batch_list)
            sent_batch_list = torch.FloatTensor(sent_batch_list).to(device)
            targets_batch_list = torch.FloatTensor(targets_batch_list).to(device)
            cb_batch_list = torch.as_tensor(cb_batch_list).to(device)
            number_of_batches += 1
            tag_scores = model(sent_batch_list, cb_batch_list)
            tag_scores = tag_scores.view(tag_scores.shape[0] * tag_scores.shape[1], tag_scores.shape[2])
            targets_batch_list = torch.as_tensor(
                targets_batch_list.view(targets_batch_list.shape[0] * targets_batch_list.shape[1]), dtype=torch.long)
            loss = loss_function(tag_scores, targets_batch_list)  # .view(-1))
            # loss and acc calc:
            epoch_loss += loss

            tag_scores = torch.argmax(tag_scores, dim=1)  # , keepdim = True)

            non_pad_elements = (targets_batch_list != len(tag_to_ix))  # .nonzero()
            corrects = (tag_scores[non_pad_elements] == targets_batch_list[non_pad_elements])

            epoch_correct_classifications += sum(corrects)
            epoch_total_classifications += len(tag_scores[non_pad_elements])
            targets_batch_list = []
            sent_batch_list = []
            cb_batch_list = []

    epoch_acc = epoch_correct_classifications / epoch_total_classifications
    epoch_loss = epoch_loss / epoch_total_classifications
    return epoch_loss, epoch_acc


def prepare_sequence_case_base(seq, to_ix, max_len_sent, pad=True):
    idxs = [w for w in seq]
    if (len(idxs) < max_len_sent) and pad:
        idxs = pad_with_zeros_case_base(idxs, max_len_sent, len(to_ix))
    return idxs


def pad_with_zeros_case_base(state_to_append, max_len_sent, plus_len):
    for i in range(max_len_sent - len(state_to_append)):
        state_to_append.append([0, 0, 0])
    return state_to_append


def train_model_case_base(model, optimizer, loss_function, max_epochs, training_data, word_to_ix, tag_to_ix,
                          validation_data, max_len_sent, batch_size=128):
    print('train model : \n')
    # early stopping params
    rounds_without_improvement = 0
    patience = 5
    best_acc = -1
    for epoch in range(max_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
        # return to train mode after val mode.
        model.train()
        print('epoch number : ' + str(epoch) + '\n')
        epoch_loss = 0
        epoch_correct_classifications = 0
        epoch_total_classifications = 0
        targets_batch_list = []
        sent_batch_list = []
        cb_batch_list = []
        counter = 0
        number_of_batches = 0
        for sentence, tags, cb in zip(*training_data):
            # split to batches by batch_size
            sentence_in = prepare_sequence(sentence, word_to_ix, max_len_sent)
            sent_batch_list.append(sentence_in)
            targets = prepare_sequence(tags, tag_to_ix, max_len_sent)
            targets_batch_list.append(targets)
            cbs = prepare_sequence_case_base(cb, tag_to_ix, max_len_sent)
            cb_batch_list.append(cbs)
            if (len(sent_batch_list) + counter) != (len(training_data[0])):
                if len(sent_batch_list) < batch_size:
                    continue
            counter += len(sent_batch_list)
            sent_batch_list = torch.FloatTensor(sent_batch_list).to(device)
            targets_batch_list = torch.FloatTensor(targets_batch_list).to(device)
            cb_batch_list = torch.as_tensor(cb_batch_list).to(device)

            number_of_batches += 1
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            tag_scores = model(sent_batch_list, cb_batch_list)
            tag_scores = tag_scores.view(tag_scores.shape[0] * tag_scores.shape[1], tag_scores.shape[2])
            targets_batch_list = torch.as_tensor(
                targets_batch_list.view(targets_batch_list.shape[0] * targets_batch_list.shape[1]), dtype=torch.long)
            loss = loss_function(tag_scores, targets_batch_list)  # .view(-1))
            loss.backward()
            optimizer.step()
            # loss and acc calc:
            epoch_loss += loss

            tag_scores = torch.argmax(tag_scores, dim=1)  # , keepdim = True)

            non_pad_elements = (targets_batch_list != len(tag_to_ix))  # .nonzero()
            corrects = (tag_scores[non_pad_elements] == targets_batch_list[non_pad_elements])

            epoch_correct_classifications += sum(corrects)
            epoch_total_classifications += len(tag_scores[non_pad_elements])
            targets_batch_list = []
            sent_batch_list = []
            cb_batch_list = []
        epoch_acc = epoch_correct_classifications / epoch_total_classifications
        epoch_loss = epoch_loss / epoch_total_classifications
        print('train loss : ' + str(epoch_loss.item()) + ', train acc : ' + str(epoch_acc.item()) + '\n')
        if validation_data:
            val_loss, val_acc = eval_model_case_base(model, loss_function, validation_data, word_to_ix, tag_to_ix,
                                                     max_len_sent, batch_size)
            print('validation loss : ' + str(val_loss.item()) + ', validation acc : ' + str(val_acc.item()) + '\n')
            stop_train, best_acc, rounds_without_improvement = check_stop_condition(val_acc.item(),
                                                                                    rounds_without_improvement,
                                                                                    best_acc, patience)

            if stop_train:
                print('\n Stop training contidion was met \n')
                break
    if validation_data:
        return val_loss, val_acc
    # if not specifying validation data return train performances
    return epoch_loss, epoch_acc
