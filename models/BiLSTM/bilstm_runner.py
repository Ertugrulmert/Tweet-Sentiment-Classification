from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from models.BiLSTM.embed_utils import * 
import models.utils as utils
from models.BiLSTM.biLSTM_model import *

from models import Runner
import configparser

import os
import tensorflow as tf
import argparse
import wandb
from keras import backend as K



class BiLSTMRunner(Runner):
    def __init__(self):
        super().__init__()
        
        # loading config
        self.config_ini = configparser.ConfigParser()
        dir = os.path.dirname(os.path.realpath(__file__))
        print("\n\n\nLoading config:\n")
        print(os.path.join(dir, 'config.ini'))
        self.config_ini.read(os.path.join(dir, 'config.ini'))
        print(json.dumps({section: dict(self.config_ini[section]) for section in self.config_ini.sections()}, indent=4))
        print("\n\n")

        
        self.model = biLSTM_model( save_suffix= self.config_ini['Paths']['CheckpointSuffix'],  
                                   #embedding params
                                   embed_dim  = int(self.config_ini['Model']['EmbedDim']),
                                   embed_type =     self.config_ini['Model']['EmbedType'], 
                                   max_len    = int(self.config_ini['Model']['MaxSentenceLen']),
                                   
                                   #model extension options
                                   augment_lda  = self.config_ini['Extensions'].getboolean('UseLDA'),
                                   augment_stats= self.config_ini['Extensions'].getboolean('UseStats'), 
                                   augment_vader= self.config_ini['Extensions'].getboolean('UseVader'), 
                                   
                                   #path params
                                   ckpt_folder = self.config_ini['Paths']['CheckpointDir'],
                                   embed_path  = self.config_ini['Paths']['EmbedDir'], 
                                   data_path   = self.config_ini['Paths']['DataDir']
                                  )
        
        

    def train(self):
        print("loading data for training...")
        tweets, labels = utils.load_tweets(
            os.path.join(
                path_root, self.config_ini['Paths']['DataDir'], self.config_ini['Paths']['DatasetPos']),
            os.path.join(
                path_root, self.config_ini['Paths']['DataDir'], self.config_ini['Paths']['DatasetNeg'])
        )

        
        device_name = tf.test.gpu_device_name()
        if device_name != "/device:GPU:0":
            device_name = "/cpu:0"
        print('Found device at: {}'.format(device_name))

        print("Training model...")
        
        with tf.device(device_name):
        
            accuracy, f1, precision, recall = self.model.train_model( tweets, labels, 
                                                            
                    random_state = int(   self.config_ini['Model']['RandomState']),
                    
                              
                    #training params
                    batch_size = int(   self.config_ini['Model']['BatchSize']), 
                    lr         = float( self.config_ini['Model']['LR']), 
                    epochs     = int(   self.config_ini['Model']['MaxEpochs']), 
                    optim      =        self.config_ini['Model']['Optimizer'],                                 
                                                            
                    dropout    = float( self.config_ini['Model']['Dropout']), 

                    #LSTM
                    cell_size  = int(   self.config_ini['Model']['CellSize']), 
                    num_LSTM   = int(   self.config_ini['Model']['NumLSTM']), 
                    #Conv
                    num_conv1D = int(   self.config_ini['Model']['CellSize']), 
                    conv_dim   = int(   self.config_ini['Model']['ConvDim']), 
                    #Dense
                    dense_dim  = int(   self.config_ini['Model']['DenseDim']), 
                    num_dense  = int(   self.config_ini['Model']['NumDense'])  ) 
        
        
      

    def evaluate(self):
        print("evaluating on test data...")
        tweets = utils.load_test_data(os.path.join(path_root, self.config_ini['Paths']['DataDir'], self.config_ini['Paths']['TestData']))
        
        device_name = tf.test.gpu_device_name()
        if device_name != "/device:GPU:0":
            device_name = "/cpu:0"
        print('Found device at: {}'.format(device_name))
        
        if not self.config_ini['Paths'].getboolean('LoadCheckpoint') or not self.model.load_model():
            print("Training model...")
            self.train()
             
        
        with tf.device(device_name):
            predictions = self.model.predict(tweets, prepare_input=True)


        utils.create_submission_csv(f"biLSTM_submission.csv", predictions)
