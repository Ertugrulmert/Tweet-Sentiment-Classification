from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import torch
import models.utils as utils
from tqdm import tqdm

def classify_data(model, tokenizer, tweets, pos_label=1, neg_label=0, device='cpu'):
    """Evaluate test data using a BERT-based classifier

    Loads the test data and preprocesses the tweets one-by-one using the provided model.

    Args:
        model (torch.nn.Module): Classifier model used to generate predictions
        tokenizer (transformers.PreTrainedTokenizer): BERT's tokenizer
        tweets (list): List of tweets to classify
        pos_label (int): Label of positive class
        neg_label (int): Label of negative class
        device (torch.device or str): Device on which to run the model
    """
    model = model.eval()
    with torch.no_grad():
        sm = torch.nn.Softmax()
        predictions = []
        model = model.to(device)
        for tweet in tqdm(tweets):
            encoded_tweet = tokenizer(tweet, return_tensors='pt')
            encoded_tweet = encoded_tweet.to(device)
            output = model(**encoded_tweet)
            scores = output[0][0].detach()
            scores = sm(scores)
            if scores[neg_label] < scores[pos_label]:
                predictions.append(1)
            else:
                predictions.append(-1)
    return predictions

def classify_data_bilstm(model, tokenizer, tweets, pos_label=1, neg_label=0, device='cpu'):
    """Evaluate test data using a BERT-based classifier

    Loads the test data and preprocesses the tweets one-by-one using the provided model.

    Args:
        model (torch.nn.Module): Classifier model used to generate predictions
        tokenizer (transformers.PreTrainedTokenizer): BERT's tokenizer
        tweets (list): List of tweets to classify
        pos_label (int): Label of positive class
        neg_label (int): Label of negative class
        device (torch.device or str): Device on which to run the model
    """
    model = model.eval()
    with torch.no_grad():
        sm = torch.nn.Softmax()
        predictions = []
        model = model.to(device)
        for tweet in tqdm(tweets):
            encoded_tweet = tokenizer(tweet, return_tensors='pt')
            encoded_tweet = encoded_tweet.to(device)
            output = model(**encoded_tweet)
            scores = output[0].detach()
            scores = sm(scores)
            if scores[neg_label] < scores[pos_label]:
                predictions.append(1)
            else:
                predictions.append(-1)
    return predictions

def classify_data_stats(model, tokenizer, tweets, vader=None, lda=None, stats=None, pos_label=1, neg_label = 0, device='cpu'):
    sm = torch.nn.Softmax()
    predictions = []
    model = model.to(device)
    for i, tweet in tqdm(enumerate(tweets)):
        vader_i = None
        lda_i = None
        stats_i = None
        if model.vader:
            vader_i = vader[i]
            vader_i = torch.Tensor([vader_i]*10)
            vader_i = vader_i.unsqueeze(0)
            vader_i = vader_i.to(device)
        if model.lda:
            lda_i = lda[i]
            lda_i = torch.Tensor(lda_i)
            lda_i = lda_i.unsqueeze(0)
            lda_i = lda_i.to(device)
        if model.stats:
            stats_i = stats[i]
            stats_i = torch.Tensor(stats_i)
            stats_i = stats_i.unsqueeze(0)
            stats_i = stats_i.to(device)
        encoded_tweet = tokenizer(tweet, return_tensors='pt')
        encoded_tweet = encoded_tweet.to(device)
        output = model(**encoded_tweet, vader = vader_i, lda = lda_i, stats = stats_i)
        scores = output[0].detach()
        scores = sm(scores)
        if scores[neg_label] < scores[pos_label]:
            predictions.append(1)
        else:
            predictions.append(-1)
    return predictions

def evaluate_test_data(model, tokenizer, test_data_filename, submission_filename, pos_label=1, neg_label=0, device='cpu'):
    """Evaluate test data using a BERT-based classifier

    Loads the test data and preprocesses the tweets one-by-one using the provided model.
    Saves the result as a csv file.

    Args:
        model (torch.nn.Module): Classifier model used to generate predictions
        tokenizer (transformers.PreTrainedTokenizer): BERT's tokenizer
        test_data_filename (string): File from which to load test data
        submission_filename (string): Name of the csv file to which predictions will be saved
        pos_label (int): Label of positive class
        neg_label (int): Label of negative class
        device (torch.device or str): Device on which to run the model
    """
    tweets = utils.load_test_data(test_data_filename)
    predictions = classify_data(model, tokenizer, tweets, pos_label, neg_label, device)
    utils.create_submission_csv(submission_filename, predictions)