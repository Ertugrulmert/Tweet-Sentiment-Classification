import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import csv
import os
from sklearn.metrics import classification_report, f1_score, accuracy_score, plot_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from preprocessing.preprocessing import sentence_statistics_from_file




def load_tweets(filename_pos, filename_neg):

    """Loads dataset to numpy arrays
        
        Args:
            
            filename_pos (str) : the path to the txt file storing dataset tweets with ground truth label: positive
            filename_neg (str) : the path to the txt file storing dataset tweets with ground truth label: negative

        Returns:
            tweets (np.ndarray) : array of tweet strings (not shuffled)  
            labels (np.ndarray) : array of ground truth labels (0 or 1) 

    """


    tweets = []
    labels = []

    with open(filename_pos, 'r', encoding='utf-8') as f:
        for line in f:
            tweets.append(line.rstrip())
            labels.append(1)

    with open(filename_neg, 'r', encoding='utf-8') as f:
        for line in f:
            tweets.append(line.rstrip())
            labels.append(0)

    tweets = np.array(tweets)
    labels = np.array(labels)

    return tweets, labels

def make_stats(filename_pos, filename_neg):

    """Extracts statistical vectors of language specific indicator frequencies (e.g number of emoticons, contractions, "haha" variations...)
          given dataset file paths
        
        Args:
            
            filename_pos (str) : the path to the txt file storing dataset tweets with ground truth label: positive
            filename_neg (str) : the path to the txt file storing dataset tweets with ground truth label: negative

        Returns:
            stats_arr (np.ndarray) : array of statistical vectors of language specific indicator frequencies, dimensions: (dataset size, 10)

    """

    stats = sentence_statistics_from_file(filename_pos)

    stats.extend( sentence_statistics_from_file(filename_neg) )

    stats_arr = np.array(stats)

    return stats_arr

def load_stats(filename_pos, filename_neg):


    """Loads statistical vectors of language specific indicator frequencies (e.g number of emoticons, contractions, "haha" variations...)
          given file paths of previously saved statistics data
        
        Args:
            
            filename_pos (str) : the path to the txt file storing statistic vectors of dataset with ground truth label: positive
            filename_neg (str) : the path to the txt file storing statistic vectors of dataset with ground truth label: negative

        Returns:
            stats_arr (np.ndarray) : array of statistical vectors of language specific indicator frequencies, dimensions: (dataset size, 10)

    """

    stats_pos = pd.read_csv(filename_pos).to_numpy()[:,1:]

    stats_neg = pd.read_csv(filename_neg).to_numpy()[:,1:]

    stats_arr = np.append(stats_pos, stats_neg ,axis=0)

    return stats_arr

def calc_metrics(Y_true, Y_pred, print_metrics=True):

    """Calculates evaluation metrics given ground truth labels and model predictions
        
        Args:
            
            Y_true (str) : ground truth labels
            Y_pred (str) : model predictions
            print_metrics (bool) : whether to print metrics to the console

        Returns:
            accuracy, f1, precision, recall (floats) : evaluation metrics 

    """

    accuracy = accuracy_score(Y_true, Y_pred)
    f1 = f1_score(Y_true, Y_pred)
    precision = precision_score(Y_true, Y_pred)
    recall = recall_score(Y_true, Y_pred)

    if print_metrics:
        print("Accuracy: " ,accuracy)
        print("Weighted F1 Score: " ,f1 )
        print("Precision: " ,precision)
        print("Recall: " ,recall )     

    return accuracy, f1, precision, recall

def parse_vader(tweets):

    """Retrieves sentence polarity scores of tweet strings acfording to VADER sentiment lexicon
        
        Args:
            
            tweets (np.ndarray or list) : iterable of tweet strings

        Returns:
            (np.ndarray of float) sentence polarity scores

    """

    sid = SentimentIntensityAnalyzer()
    return np.array([sid.polarity_scores(tweet)["compound"] for tweet in tweets])

def load_test_data(filename):
    """Method for loading test data

    Loads test data from a .txt file and returns a list of strings corresponding to the test tweets
    
    Args:
        filename (str): Path to the file containing the test data
    
    Returns:
        (list): List of strings corresponding to the test tweets
    """
    tweets = []

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            tweets.append(line.rstrip())
    
    return tweets

def load_test_stats(filename):


    """Loads statistical vectors of language specific indicator frequencies (e.g number of emoticons, contractions, "haha" variations...)
          for the test set given its file path
        
        Args:
            
            filename (str) : file path of test set

        Returns:
            stats (np.ndarray): array of statistical vectors of language specific indicator frequencies, dimensions: (dataset size, 10)

    """

    stats = pd.read_csv(filename).to_numpy()[:,1:]
    return stats


def create_submission_csv(filename, predictions):
    """Method for creating submission files
    
    Takes list of predictions and formats them into a csv file ready for submission

    Args:
        filename (str): Name of csv file where predictions will be stored
        predictions (str): List of predictions for the test data
    """
    header = ["Id", "Prediction"]

    with open(filename, 'w') as f:
        writer = csv.writer(f)
        
        writer.writerow(header)
        
        for row in enumerate(predictions, 1):
            writer.writerow(row)
    return