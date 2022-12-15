from gensim.models import LdaMulticore, TfidfModel
from gensim.corpora import Dictionary
from gensim.test.utils import datapath, get_tmpfile
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import numpy as np
import os 

class LDA_Model:
    def __init__(self,  num_topics = 10, ckpt_folder="checkpoints/lda", save_suffix=""):

        """Latent Dirichlet Aiiocation Topic Model Constructor
        
        Args:

            num_topics    (int) : Number of topics the topic model will create a distribution for
            ckpt_folder   (str) : The folder path for LDA model + LDA model dictionary save files
            save_suffix   (str) : An identifier used do distinguish save files for a specific run of the lda model - generally including the type of preprocessing done to the input data

        """

        self.num_topics = num_topics
        
        self.lda_path  = f"{ckpt_folder}/lda_{save_suffix}"
        self.dict_path = f"{ckpt_folder}/dict_{save_suffix}.txt"
        
        
    def train(self, X_train, save_model=True):

        """Model training funct
        
        Args:

            X_train    (np.ndarray) : Array of tweet strings to train the LDA Model
            save_model (bool) : Whether to save the LDA model to file

        Returns:
            topic_vectors (np.ndarray) : Vectors of topic distribution for each tweet in the input data
        """
        
        X_train_words = [text_to_word_sequence(s) for s in X_train]

        if  self.load_lda():
            bow_corpus = [self.dictionary.doc2bow(doc) for doc in X_train_words]
            save_model=False
            
            print("LDA model successfuly loaded.")
            
        else:
        
            self.dictionary = Dictionary(X_train_words)
            self.dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
            bow_corpus = [self.dictionary.doc2bow(doc) for doc in X_train_words]
            # Build LDA model
            self.lda_model = LdaMulticore(bow_corpus, num_topics=10, id2word=self.dictionary, passes=2, workers=8)
        
        topic_vectors = np.zeros( (len(X_train),self.num_topics) )
        
        for idx, doc in enumerate(bow_corpus):
            
            topic_vector = np.zeros( (self.num_topics) )
            topics = np.array( self.lda_model[doc] )
            
            topic_vector[topics[:,0].astype(int)] = topics[:,1]     
            topic_vectors[idx] = topic_vector
            
        if save_model:
            try:
                self.save_lda()
            except:
                print("LDA model could not be saved.")
                return topic_vectors
            
        return topic_vectors
        
        
    def process_new_data(self, X_new ):

        """Model training function
        
        Args:

            X_new     (np.ndarray) : Array of tweet strings to process

        Returns:
            topic_vectors (np.ndarray) : Vectors of topic distribution for each tweet in the input data
        """
        
        X_new_words = [text_to_word_sequence(s) for s in X_new]
        new_bow_corpus = [self.dictionary.doc2bow(doc) for doc in X_new_words]
        
        topic_vectors = np.zeros( (len(X_new),self.num_topics) )
        
        for idx, doc in enumerate(new_bow_corpus):
            
            topic_vector = np.zeros( (self.num_topics) )
            topics = np.array( self.lda_model[doc] )
            
            topic_vector[topics[:,0].astype(int)] = topics[:,1]     
            topic_vectors[idx] = topic_vector
            
        return topic_vectors
            
        
    def load_lda(self):

        """Model loading function

        Returns:
            (bool) : whether the model could be loaded / save files exist
        """
        if os.path.exists(self.lda_path) and os.path.exists(self.dict_path):

            self.lda_model = LdaMulticore.load(self.lda_path)
            self.dictionary = Dictionary.load_from_text(self.dict_path)
            return True
        else:
            return False
    
    def save_lda(self):

        """Saves model to save files"""
    
        self.lda_model.save(self.lda_path)

        self.dictionary.save_as_text(self.dict_path)
        
        

