from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import numpy as np
import matplotlib.pyplot as plt
import csv

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Word2Vec
from gensim.models import FastText, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from gensim.test.utils import get_tmpfile

from tensorflow.keras.preprocessing.text import text_to_word_sequence

from datetime import datetime
import os

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

np.random.seed(1)

def read_glove_vector(glove_vec):

    """Reads glove word to embedding vector mapping from save file 
        
        Args:
            
            glove_vec (str) : the path to the glove word vectors file

        Returns:
            word_to_vec_map (dict) : ditctionary mapping word strings to np.ndarray word vectors 
    """


    with open(glove_vec, 'r', encoding='UTF-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            w_line = line.split()
            curr_word = w_line[0]
            word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)

    return word_to_vec_map

def train_ftxt(X_train, embed_dim=100, save_suffix ="", save_full_model=False, 
    max_vocab=100000, ckpt_folder="checkpoints"):

    """Trains and saves Fasttext embedding model (or loads from file)
        
        Args:
            
            X_train (np.ndarray) : array of input tweet strings
            embed_dim (int) : desired size of the word vectors 
            save_suffix (str) : any identifier to distinguish current fasttext model when saving it to file
            save_full_model (bool) : whether to save (and load) only final word vectors or the entire model
            max_vocab (int) : maximum vocabulary size allowed for fasttext to use
            ckpt_folder (str) : the folder path where trained Fasttext models will be saved and loaded from

        Returns:
            ftxt_model or kv : returns a full fasttext model object or only keyed vectors 
    """

    epochs=15

    #constructing save file path
    save_name = f'./{ckpt_folder}/ftxt_{embed_dim}_{save_suffix}'

    # attempting to load Fasttext model from save file

    if not save_full_model:
        if os.path.exists(save_name+".bin"):
            kv = KeyedVectors.load_word2vec_format(save_name+".bin", binary=True)
            print("Fasttext model loaded.")
            return kv
    else:
        if os.path.exists(save_name+".model"):
            fname = get_tmpfile(save_name+'.model')
            model = FastText.load(fname)
            print("Fasttext model loaded.")
            return model

    X_train_tokenized = [text_to_word_sequence(s) for s in X_train]


    class callback(CallbackAny2Vec):
        ''' callback function to print current epoch number'''
        def __init__(self):
            self.epoch = 0
        def on_epoch_end(self, model):
            print(f'Finished epoch {self.epoch}')
            self.epoch += 1

    #training the model
    ftxt_model = FastText(X_train_tokenized, workers = 8, 
               vector_size=embed_dim,   
               callbacks=[callback()],
               epochs = epochs,
               max_final_vocab = max_vocab
               )
    print("FastText vocab size: " + str(len(ftxt_model.wv.key_to_index)))

    # saving trained model keyed vectors or full model
    if not save_full_model:
        # Save Model
        kv = ftxt_model.wv
        save_name = save_name+'.bin'
        kv.save_word2vec_format(save_name, binary=True)
        del ftxt_model

        return kv

    try:
        save_name = save_name+'.model'
        fname = get_tmpfile(save_name)
        ftxt_model.save(fname)
    except:
        return ftxt_model
    return ftxt_model



def make_embed_matrix( word_idx, word_to_vec_map, augment_vader=False, vader_dim=10, print_info=True):

    """ Constructs an embedding lookup table mapping token ids form the tokenizer to vectors
        
        Args:
            
            word_idx (dict) : dictionary mapping word strings to token id integers
            word_to_vec_map (dict) : dictionary mapping word strings to word embeddings generated by an embedding model 
            augment_vader (bool) : whether to extend the original word embeddings with a section for polarity scores from VADER sentiment lexicon 
            vader_dim (int) : the size of the lexicon section of the embedding - same word polarity score coppied along this size to obtain a larger relative size compared to initial embedding
            print_info (bool) : whetehr to print vocabulary and embedding vector information

        Returns:
            embed_matrix (np.ndarray): array of word vectors where row number corresponds to token id
    """

    # safeguarding against different object types for word_to_vec_map
    if type(word_to_vec_map) is dict:
        embed_vector_len = list(word_to_vec_map.values())[1].shape[0]
    else:
        embed_vector_len = word_to_vec_map[1].shape[0]

    vocab_len = len(word_idx)

    if print_info:
        print(f"Vocab Len: {vocab_len}")
        if augment_vader:
            print(f"Embedding Vector + Vader Len: {embed_vector_len+vader_dim}")
        else:
            print(f"Embedding Vector Len: {embed_vector_len}")

    # increase dimension to add lexicon polarity score section 
    if augment_vader:
        embed_matrix = np.zeros((vocab_len+1, embed_vector_len+vader_dim))
        sent_analyzer = SentimentIntensityAnalyzer()
    else:
        embed_matrix = np.zeros((vocab_len+1, embed_vector_len))

    for word, index in word_idx.items():

        if type(word_to_vec_map) is dict:

            embed_vector = word_to_vec_map.get(word)
            if embed_vector is not None:
                embed_matrix[index, :embed_vector_len] = embed_vector
            else:
                # if word is out of vocabulary, a arandom vector is assigned in its place
                embed_matrix[index, :embed_vector_len] = np.random.uniform(-5, 5, embed_vector_len)
        else:
            try:
                embed_vector = word_to_vec_map[word]
                embed_matrix[index, :embed_vector_len] = embed_vector
            except KeyError:
                # if word is out of vocabulary, a arandom vector is assigned in its place
                embed_matrix[index, :embed_vector_len] = np.random.uniform(-5, 5, embed_vector_len)
        
        #adding a word polarity score of size "vader_dim", - one number coppied over this length
        if augment_vader:
            embed_matrix[index, embed_vector_len+1:] = sent_analyzer.polarity_scores(word)['compound']

    return embed_matrix

def make_combined_embed_matrix( word_idx, glove_word_to_vec_map,  ftxt_word_to_vec_map, 
                                glove_embed_dim=100 , ftxt_embed_dim=100 , vader_dim=20 , 
                                augment_vader=False, print_info=True):


    """ Constructs an embedding lookup table mapping token ids form the tokenizer to vectors - vectors combine Fasttext and Glove vectors (and optionally lexicon scores)
        
        Args:
            
            word_idx (dict) : dictionary mapping word strings to token id integers
            glove_word_to_vec_map (dict) : dictionary mapping word strings to word embeddings generated by Glove
            ftxt_word_to_vec_map (dict)  : dictionary mapping word strings to word embeddings generated by Fasttext
            glove_embed_dim : Glove    word embedding size
            ftxt_embed_dim  : Fasttext word embedding size

            vader_dim (int) : the size of the lexicon section of the embedding - same word polarity score coppied along this size to obtain a larger relative size compared to initial embedding
            augment_vader (bool) : whether to extend the original word embeddings with a section for polarity scores from VADER sentiment lexicon 
            print_info (bool) : whetehr to print vocabulary and embedding vector information

        Returns:
            embed_matrix (np.ndarray): array of word vectors where row number corresponds to token id
    """


    vocab_len = len(word_idx)
    
    total_embed_dim =  glove_embed_dim + ftxt_embed_dim + vader_dim


    embed_matrix = np.zeros((vocab_len+1, total_embed_dim))
    
    sent_analyzer = SentimentIntensityAnalyzer()


    for word, index in word_idx.items():
        
        # GLOVE

        if type(glove_word_to_vec_map) is dict:

            glove_embed_vector = glove_word_to_vec_map.get(word)
            if glove_embed_vector is not None:
                embed_matrix[index, :glove_embed_dim] = glove_embed_vector
            else:
                embed_matrix[index, :glove_embed_dim] = np.random.uniform(-5, 5, glove_embed_dim)
        else:
            try:
                glove_embed_vector = glove_word_to_vec_map[word]
                embed_matrix[index, :glove_embed_dim] = glove_embed_vector
            except KeyError:
                embed_matrix[index, :glove_embed_dim] = np.random.uniform(-5, 5, glove_embed_dim)
                
        # FASTTEXT

        if type(ftxt_word_to_vec_map) is dict:

            ftxt_embed_vector = ftxt_word_to_vec_map.get(word)
            if ftxt_embed_vector is not None:
                embed_matrix[index, glove_embed_dim:glove_embed_dim+ftxt_embed_dim] = ftxt_embed_vector
            else:
                embed_matrix[index, glove_embed_dim:glove_embed_dim+ftxt_embed_dim] = np.random.uniform(-5, 5, ftxt_embed_dim)
        else:
            try:
                ftxt_embed_vector = ftxt_word_to_vec_map[word]
                embed_matrix[index, glove_embed_dim:glove_embed_dim+ftxt_embed_dim] = ftxt_embed_vector
            except KeyError:
                embed_matrix[index, glove_embed_dim:glove_embed_dim+ftxt_embed_dim] = np.random.uniform(-5, 5, ftxt_embed_dim)

                          
        embed_matrix[index, total_embed_dim-vader_dim:] = sent_analyzer.polarity_scores(word)['compound']

    return embed_matrix
