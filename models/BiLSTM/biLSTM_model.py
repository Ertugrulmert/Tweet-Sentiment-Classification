from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from models.BiLSTM.embed_utils import *
from models.utils import *
from models.lda import LDA_Model

import os, io, json 
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Activation, Dropout, Dense, Input, Bidirectional,Conv1D,MaxPool1D,Flatten,Embedding, Lambda, Concatenate
from tensorflow.keras import models, backend
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint




class biLSTM_model:


    
    def __init__(self, save_suffix, embed_type="ftxt", embed_dim=100,
                    augment_vader=False, augment_lda=False, augment_stats=False,
                    max_len=150,  
                    data_path = 'twitter-datasets/data',
                    embed_path = 'twitter-datasets/glove_twitter',
                    ckpt_folder = "checkpoints"):

        """Class constructor
        
        Args:

            save_suffix   (str) : An identifier used do distinguish save files for a specific run of the bilstm model, tokenizer, lda model
            embed_type    (str) : choose an embedding model type between the following options: "ftxt" , "glove"
            embed_dim     (int) : The dimension of the word embeddings produced by the embedding model
            augment_vader (bool): Whether to extend each word embedding by its sentiment score according to the sentiment lexicon "vader"
            augment_lda   (bool): Whether to apply Latent Dirichlet Allocation topic modeling to the data, 
                                  produce topic distributions, and use them as an auxiliary input vector in the model
            augment_stats (bool): Whether to add a vector of sentence statistics about the frequency of certain identifiers as an auxiliary input 
            max_len       (int) : The maximum number of tokens allowed for an input tweet, all tweets truncated or padded up to this value
            data_path     (str) : The folder path where training and test datasets are stored
            embed_path    (str) : The folder path where pretrained glove embeddings are stored and new fasttext embeddings are saved
            ckpt_folder   (str) : The folder path including the subfolders for different models' save files / checkpoints

        """
    
        self.embed_path = embed_path
        self.ckpt_folder = ckpt_folder
        self.save_suffix = save_suffix
        self.data_path   = data_path
        
        tokenizer_name = f"tokenizer_{self.save_suffix}.json"
        self.tokenizer_path = os.path.join(self.ckpt_folder, "tokenizer", tokenizer_name)
        self.lda_folder = os.path.join(self.ckpt_folder, "lda")
        
        self.tokenizer = None
        self.model = None
        self.ftxt_model = None
        self.glove_model = None
        self.lda_model = None
 
        
        self.model_path = ""
        self.embed_dim        = embed_dim
        self.embed_type       = embed_type

        self.augment_stats    = augment_stats
        self.augment_vader    = augment_vader
        self.augment_lda      = augment_lda
        self.max_len          = max_len
        
        # checkpoint name construction
                    
        if embed_type == "glove":
            self.model_path = os.path.join(self.ckpt_folder, "biLSTM", f"biLSTM_glove_d{self.embed_dim}")           
        elif embed_type == "ftxt":
            self.model_path = os.path.join(self.ckpt_folder, "biLSTM", f"biLSTM_ftxt_d{self.embed_dim}")  
        #else:
        #    self.model_path = f"{ckpt_folder}/biLSTM_concat_d{embed_dim}x2"
        
        if self.augment_vader:

            self.model_path += "_vader"

        if self.augment_lda:

            self.model_path += "_lda"
            
        if self.augment_stats:
         
            self.model_path += "_stats"
        
        self.model_path += f"_{self.save_suffix}"
    
    
    def build_model(self, embed_matrix, num_LSTM = 1, num_conv1D = 0, conv_dim=16,
                       num_dense= 2, dense_dim=16, cell_size = 100, 
                       dropout= 0.5):


    


        X_in = Input((self.max_len,))

        input_dim, embed_vector_len = embed_matrix.shape

        embedding_layer = Embedding(input_dim=input_dim, 
                                output_dim=embed_vector_len, 
                                input_length=self.max_len, 
                                mask_zero=   True,
                                weights = [embed_matrix], 
                                trainable=False)
    
        embeddings = embedding_layer(X_in)

        if num_LSTM:
            
            X = Bidirectional(LSTM(cell_size, activation='tanh', return_sequences=bool(num_conv1D)))(embeddings)
            X = Dropout(dropout)(X)

            #if there are more than 2 lstm cells

            for i in range(num_LSTM-1):
                
                if not num_conv1D and i == num_LSTM-2:
                    X = Bidirectional(LSTM(cell_size//2, activation='tanh', return_sequences=False))(X)
                else:
                    X = Bidirectional(LSTM(cell_size//2, activation='tanh', return_sequences=True))(X)
                X = Dropout(dropout)(X)
    
        else:
            X = embeddings
            
        # if desired, adding Conv1D + dropout + maxpool layers
        if num_conv1D:

            for _ in range(num_conv1D):

                X = Conv1D(filters=conv_dim, kernel_size=3, strides=1, padding="valid", activation='relu')(X)
                X = Dropout(dropout)(X)
                X = MaxPool1D(pool_size=2, strides=2, padding="valid")(X)
    
            X = Flatten()(X)


        # adding final fully connected layers
        for _ in range(num_dense-1):
            X = Dense(dense_dim, activation='relu')(X)

        #Adding the sentence length information before final FC layer
        input_list = [X_in]

        if self.augment_stats:
            X_stats = Input((10,))
            X = Concatenate()([X, X_stats])
            input_list.append(X_stats)

        if self.augment_lda:
            X_lda = Input((10,))
            X = Concatenate()([X, X_lda])
            input_list.append(X_lda)
       


        X = Dense(1, activation='sigmoid')(X)

        self.model = models.Model(inputs=input_list, outputs=X)

    
                   
              
    
    def train_model(self, X, Y,
                    num_LSTM = 1, num_conv1D = 0, conv_dim=32,
                    num_dense= 2, dense_dim=16, cell_size = 100, dropout= 0.5, 
                    # training params
                    batch_size = 500, lr= 0.0005, epochs=20, optim="adam", 
                    random_state=0 ):

        """Model training function
        
        Args:
            -- Inputs
            X (np.ndarray) : input tweets, array of strings
            Y (np.ndarray) : input labels, array of integers

            -- Architecture Parameters
            num_LSTM   (int) : Number of Bidirectional LSTM layers
            num_conv1D (int) : Number of 1 dimensional convolution layers to apply sequentially to sequential input
            conv_dim   (int) : Number of convolutional filters at each 1D Conv layer
            num_dense  (int) : Number of fully connected layers 
            dense_dim  (int) : Size of fully connected layers
            cell_size  (int) : Size of the internal cell of the LSTMs in the Bidirectional LSTM
            dropout    (float) : Dropout probability

            -- Training Parameters
            batch_size   (int) : size of training batches 
            lr           (float) : learning rate
            epochs       (int) : number of training epochs
            optim        (str) : type of optimizer for training - options: "adam", "sgd", "rmsprop"
            random_state (int) : the seed used to condition the random numbers of tensorflow and numpy for reproducible results

        """



        # setting random seeds for reproducable results
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

        #preparing model training inputs
        #X_train, X_dev -> for embedding generation
        train_inputs, dev_inputs, X_train, Y_train, Y_dev = self._prepare_inputs(X,Y,  random_state = random_state )
        
        word_idx = self.tokenizer.word_index
       
        
        # choosing embedding type and forming a lookup matrix for token embeddings
        if self.embed_type == "glove":
            glove_path = f"{self.embed_path}/glove.twitter.27B.{self.embed_dim}d.txt"
            word_to_vec_map = read_glove_vector(glove_path)
            embed_matrix = make_embed_matrix( word_idx, word_to_vec_map, augment_vader=self.augment_vader)
            
        elif self.embed_type == "ftxt":

            word_to_vec_map = train_ftxt(X_train, embed_dim=self.embed_dim, save_suffix=self.save_suffix, ckpt_folder=self.embed_path)
            embed_matrix = make_embed_matrix( word_idx, word_to_vec_map, augment_vader=self.augment_vader)
            
        # building model architecture
        self.build_model(embed_matrix=embed_matrix, 
        cell_size = cell_size, num_conv1D=num_conv1D, conv_dim=conv_dim, dropout= dropout, num_LSTM=num_LSTM, dense_dim=dense_dim, num_dense=num_dense)
                
        # setting callback functions to automatically lower learning rate and then stop when at risk of overfitting
        early = EarlyStopping(monitor="val_loss", patience=5, verbose=1)
        redonplat = ReduceLROnPlateau(monitor="val_loss", patience=2, verbose=1, factor=0.5)
        checkpoint = ModelCheckpoint( filepath=self.model_path, save_weights_only=False, 
                                  monitor='val_accuracy', mode='max', save_best_only=True)

        callbacks_list = [early, redonplat,checkpoint] 
        
        #choosing optimizer 
        if  optim=="adam":
            optim = Adam(learning_rate = lr)
        elif  optim=="sgd":
            optim = SGD(learning_rate = lr)
        else:
            optim = RMSprop(learning_rate = lr)
            

        self.model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])

        self.model.summary()
    
        self.model.fit(x=train_inputs, y=Y_train, batch_size=batch_size, epochs=epochs,shuffle=True,
              validation_data=(dev_inputs,Y_dev), callbacks=callbacks_list )
        
        # evaluating on dev set
        eval_results = self.evaluate(dev_inputs, Y_dev, prepare_input=False)

        self.save_model()

        return eval_results


              
                    
    def predict(self, X, prepare_input=True):  


        """Model prediction function 
        
        Args:
            
            X (np.ndarray) OR list of np.ndarray : either prepared list of all model inputs or unprepared array of tweet strings
            prepare_input (bool) : whether to prepare the input by tokenizing, extracting statistics vectors, extracting topic distributions if required

        Returns:
            Y_pred (np.ndarray) : model predictions

        """
                   
        if prepare_input: 
            X_tokenized = self._tokenize(X)
            X_input = [X_tokenized]

            if self.augment_stats:
                X_stats = np.array( sentence_statistics(X) )
                X_input.append(X_stats)

            if self.augment_lda:
                X_topics   = self.lda_model.process_new_data(X)
                X_input.append(X_topics)

        else:
            X_input = X

        Y_pred = self.model.predict(x=X_input)
        Y_pred = (Y_pred[:,0] > 0.5).astype(np.int)
        return Y_pred
                    
    
    def evaluate(self, X, Y, prepare_input=True):

        """Generates evaluation metrics for given data and ground truth labels
        
        Args:
            
            X (np.ndarray) OR list of np.ndarray : either prepared list of all model inputs or unprepared  array of tweet strings
            Y (np.ndarray)  : ground truth labels 
            prepare_input (bool) : whether to prepare the input by tokenizing, extracting statistics vectors, extracting topic distributions if required

        Returns:
            accuracy, f1 score, precision, recall : evaluation metrics generated from model predictions and ground truth labels

        """

        Y_pred = self.predict(X, prepare_input=prepare_input)
        return calc_metrics(Y, Y_pred, print_metrics=True)
    
    def save_model(self):

        """Saves the model to the checkpoint location"""

        self.model.save(self.model_path)

        if self.augment_lda:
            self.lda_model.save_lda()
        print("Model saved.")
                    
    def load_model(self):

        """If possible, loads the model from save files 

        Returns:
            (bool) : whether the model could be loaded / save files exist

        """
        if os.path.exists(self.tokenizer_path) and os.path.exists(self.model_path):
            self.model = models.load_model(self.model_path)
            self._load_tokenizer()
        else: return False

        if self.augment_lda:
            self.lda_model = LDA_Model( ckpt_folder=self.lda_folder, save_suffix=self.save_suffix)
            return self.lda_model.load_lda()

        print("Model loaded.")
        return True
    
    def _train_tokenize(self, X, filters='"#$%&@123456789' ):

        """Trains the tokenizer on input data and saves it to a save file.
        
        Args:
            
            X (np.ndarray) : input array of tweet strings
            filters : the symbols that the tokenizer will eliminate while processing the data

        Returns:
            X_tokenized (np.ndarray) : input tweets converted to rows of token ids from the tokenizer

        """
        
        self.tokenizer = Tokenizer( filters=filters)
        self.tokenizer.fit_on_texts(X)
        
        X_tokenized = self.tokenizer.texts_to_sequences(X)
        X_tokenized = pad_sequences(X_tokenized, maxlen=self.max_len, padding='post')
        
        self._save_tokenizer()
        
        return X_tokenized
            
    def _tokenize(self, X):

        """Tokenizes input data.
        
        Args:
            
            X (np.ndarray) : input array of tweet strings
        Returns:
            X_tokenized (np.ndarray) : input tweets converted to rows of token ids from the tokenizer

        """
        
        X_tokenized = self.tokenizer.texts_to_sequences(X)
        X_tokenized = pad_sequences(X_tokenized, maxlen=self.max_len, padding='post')
        return X_tokenized
        
    
    def _save_tokenizer(self):
        """Saves the tokenizer to save file"""
    
        tokenizer_json = self.tokenizer.to_json()
        
        with io.open(self.tokenizer_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))
        print("Tokenizer saved to json file.")
    
    def _load_tokenizer(self):
        """Loads the tokenizer from save file"""
        
        with open(self.tokenizer_path) as f:
            tokenizer_json = json.load(f)
            self.tokenizer = tokenizer_from_json(tokenizer_json)
        print("Tokenizer loaded from save file.")
            
    def _load_stats(self):
        """Loads statistics vectors corresponding to the frequency of potentially meaningful indicators in the original unprocessed training data"""
         
        path_pos_stat =  os.path.join(self.data_path, "pos_stats.csv")
        path_neg_stat =  os.path.join(self.data_path, "neg_stats.csv")
        
        return load_stats(path_pos_stat, path_neg_stat)
    
    
    
    def _prepare_inputs(self, X, Y, random_state=0):

        """Prepares final model inputs depending on model configuration 
        
        Args:
            
            X (np.ndarray) : array of tweet strings
            Y (np.ndarray) : ground truth labels 
            random_state (int) : the seed used to condition random numbers for reproducible results

        Returns:
            train_inputs (list of np.ndarrays) : list of all model training input arrays
            dev_inputs   (list of np.ndarrays) : list of all model training input arrays 
            X_train  (np.ndarray) : unprocessed train input to train the embedding model
            Y_train  (np.ndarray) : train ground truth labels  
            Y_dev    (np.ndarray) : dev ground truth labels 

        """
        
        if self.augment_stats:    
                                                                   
            #loading statistics vectors  
            try:                                                      
                X_stats = self._load_stats()  
                print("Stats vectors extracted.")
            except:
                print("Stats vectors could not be loaded. Extracting stats.")
                X_stats = np.array( sentence_statistics(X) )
                                                                   
            #preparing model training and dev inputs
            X_train, X_dev, Y_train, Y_dev, X_train_stats, X_dev_stats = train_test_split(X, Y, X_stats, test_size=0.1, random_state = random_state)
            
        else:
            X_train, X_dev,Y_train, Y_dev = train_test_split(X, Y, test_size=0.1, random_state = random_state)
        
        # Tokenizing the dataset
        X_train_tokenized = self._train_tokenize(X_train)
        X_dev_tokenized = self._tokenize( X_dev)
        
        #creating input lists for keras model
        train_inputs = [X_train_tokenized]
        dev_inputs   = [X_dev_tokenized]

        #adding extra inputs for model extensions
        
        if self.augment_stats:
            train_inputs.append(X_train_stats)
            dev_inputs.append(X_dev_stats)

        if self.augment_lda:
            self.lda_model = LDA_Model(num_topics = 10, ckpt_folder=self.lda_folder, save_suffix=self.save_suffix)
            print("Training LDA Model")
            X_train_topics = self.lda_model.train(X_train)
            X_dev_topics   = self.lda_model.process_new_data(X_dev)
                                                                   
            train_inputs.append(X_train_topics)
            dev_inputs.append(X_dev_topics)
            
        return train_inputs, dev_inputs, X_train, Y_train, Y_dev
            
        
        
        
            
    

    
    
 