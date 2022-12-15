
from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from models.BiLSTM.embed_utils import *
from models.utils import *
from models.BiLSTM.biLSTM_model import *
import os
import tensorflow as tf
import argparse
import wandb
from keras import backend as K

nltk.download('vader_lexicon', quiet=True)



preprocess_types = ["","hashtags_" ,"repeated_chars_" , "contractions_" ,"all_simple_methods_" , "all_simple_methods_s2s_", "all_simple_methods_and_dictionary_s2s_" , "emoticons_and_haha_all_simple_methods_and_dictionary_s2s_"]



if __name__ == "__main__":

    """
        Runs the basic biLSTM model for all preprocessed data configurations and logs outputs
    """


    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder',      default='twitter-datasets/data/')
    parser.add_argument('--ckpt_folder', default='checkpoints')
    parser.add_argument('--glove_folder', default= 'twitter-datasets/glove_twitter/')

    parser.add_argument('--embed_type',        default="ftxt")
    parser.add_argument('--vader',            default=False, type=bool)
    parser.add_argument('--augment_stats',    default=False, type=bool)
    parser.add_argument('--augment_lda',      default=False, type=bool)

    parser.add_argument('--batch_size',       default=500, type=int)
    parser.add_argument('--lr',               default=0.0005, type=float)
    parser.add_argument('--epochs',           default=30, type=int)
    parser.add_argument('--embedding_dim',    default=100, type=int)
    parser.add_argument('--max_len',          default=150, type=int)
    parser.add_argument('--dropout',          default=0, type=float)

    parser.add_argument('--num_LSTM',         default=1, type=int)
    parser.add_argument('--cell_size',        default=100, type=int)

    parser.add_argument('--num_conv1D',        default=0, type=int)
    parser.add_argument('--conv_dim',          default=32, type=int)
    
    parser.add_argument('--num_dense',          default=2, type=int)
    parser.add_argument('--dense_dim',          default=16, type=int)

    hyperparams = parser.parse_args()

    wandb.init(project="cil-maml" , config=vars(hyperparams))


    device_name = tf.test.gpu_device_name()
    if device_name != "/device:GPU:0":
        device_name = "/cpu:0"
    print('Found device at: {}'.format(device_name))
    
    logging_file = os.path.join( wandb.config['ckpt_folder'], "param_search_logs", "dataset_validation_logs.txt") 
    line_str = "----------------------------------------\n"
    star_str = "****************************************\n"


    
    for prefix in preprocess_types:
        with tf.device(device_name):
            print("----------*----------*----------*----------")
            print(f"Running with dataset: {prefix}")
            print("----------*----------*----------*----------")


            #logging current experiment details to text file
            model_config = f'dataset: {prefix}\n'
            df=open(logging_file,'a')
            df.write(star_str )
            df.write(model_config )
            df.write(line_str )  
            df.close()

            data_prefix = wandb.config['data_folder']


            path = data_prefix + prefix + "train_"
            path_pos = path + "pos_full.txt"
            path_neg = path + "neg_full.txt"

            tweets, labels = load_tweets(path_pos, path_neg)
            
            
            model = biLSTM_model(save_suffix=prefix, 
                                   #embedding params
                                   embed_dim = wandb.config['embedding_dim'],
                                   embed_type=wandb.config['embed_type'], 
                                   max_len= wandb.config['max_len'],  
                                   
                                   #model extension options
                                   augment_lda  = wandb.config['augment_lda'], 
                                   augment_stats= wandb.config['augment_stats'] , 
                                   augment_vader= wandb.config['vader'],
                                   
                                   #path params
                                   ckpt_folder = wandb.config['ckpt_folder'],
                                   embed_path  = wandb.config['glove_folder'], 
                                  )
          
        
            accuracy, f1, precision, recall = model.train_model( tweets, labels, 
                              
                    # training params
                    batch_size = wandb.config['batch_size'], 
                    lr         = wandb.config['lr'], 
                    epochs     = wandb.config['epochs'], 
                    dropout   = wandb.config['dropout'], 

                    #LSTM
                    cell_size  = wandb.config['cell_size'],  num_LSTM = wandb.config['num_LSTM'],
                    #Conv
                    num_conv1D = wandb.config['num_conv1D'], conv_dim = wandb.config['conv_dim'],
                    #Dense
                    dense_dim  = wandb.config['dense_dim'], num_dense = wandb.config['num_dense'] )
            
            model_result = f"accuracy: {accuracy}, f1: {f1}, precision: {precision}, recall: {recall}\n"


            #logging current experiment results to text file
            df=open(logging_file,'a')
            df.write(model_result )
            df.close()
            
            
            del model
            del tweets
            del labels
            K.clear_session()
                     

            


