from models.MLP.models import MLP
from models import Runner
import models.utils as utils

from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
import torch
import csv
import numpy as np

import configparser
import os
from cmath import log
from pathlib import Path
import sys
import json
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))


class SvmRunner(Runner):
    def __init__(self):
        super().__init__()

        # loading config
        self.config_ini = configparser.ConfigParser()
        dir = os.path.dirname(os.path.realpath(__file__))
        print("\n\n\nLoading config:\n")
        print(os.path.join(dir, 'config.ini'))
        self.config_ini.read(os.path.join(dir, 'config.ini'))
        print(json.dumps({section: dict(
            self.config_ini[section]) for section in self.config_ini.sections()}, indent=4))
        print("\n\n")

        if self.config_ini['Model']['EmbeddingType'] == 'glove':
            # set up model
            print("loading glove embeddings...")
            self.glove = self.__glove2dict(self.config_ini['Basic']['GlovePath'])
            print("done.")
        elif self.config_ini['Model']['EmbeddingType'] == 'tfidf':
            self.feature_extraction = TfidfVectorizer(max_features=int(self.config_ini['Model']['MaxFeatures']))

        self.model = svm.LinearSVC(verbose=True)

    def train(self):
        print("loading data for training...")
        tweets, labels = utils.load_tweets(
            os.path.join(
                path_root, self.config_ini['Basic']['DataDir'], self.config_ini['Basic']['DatasetPos']),
            os.path.join(
                path_root, self.config_ini['Basic']['DataDir'], self.config_ini['Basic']['DatasetNeg'])
        )
        self.feature_extraction.fit_transform(tweets)
        X_train_raw, X_dev_raw, Y_train, Y_dev = train_test_split(
            tweets, labels, test_size=0.1, random_state=1)

        if self.config_ini['Model']['EmbeddingType'] == 'glove':
            X_train_embedded = self.__embed(X_train_raw, self.glove, int(
                self.config_ini['Model']['EmbeddingDimension']))
            X_dev_embedded = self.__embed(X_dev_raw, self.glove, int(
                self.config_ini['Model']['EmbeddingDimension']))
        elif self.config_ini['Model']['EmbeddingType'] == 'tfidf':
            X_train_embedded = self.feature_extraction.transform(X_train_raw)
            X_dev_embedded = self.feature_extraction.transform(X_dev_raw)
        self.model.fit(X_train_embedded, Y_train)
        predictions = self.model.predict(X_dev_embedded)
        print(f'validation acc: {accuracy_score(Y_dev, predictions)}')
        print(f'validation f1: {f1_score(Y_dev, predictions)}')

    def evaluate(self):
        print("evaluating on test data...")
        tweets = utils.load_test_data(os.path.join(
            path_root, self.config_ini['Basic']['DataDir'], self.config_ini['Basic']['TestData']))
        if self.config_ini['Model']['EmbeddingType'] == 'glove':
            X_embedded = torch.tensor(self.__embed(np.array(tweets), self.glove, int(
                self.config_ini['Model']['EmbeddingDimension'])))
        elif self.config_ini['Model']['EmbeddingType'] == 'tfidf':
            X_embedded = self.feature_extraction.transform(tweets)
        predictions = self.model.predict(X_embedded)
        utils.create_submission_csv(f"SVM_submission.csv", predictions)

    def __glove2dict(self, glove_filename):
        with open(glove_filename, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
            embed = {line[0]: np.array(list(map(float, line[1:])))
                     for line in reader}
        return embed

    def __embed(self, X, glove, embedding_dim):

        X_embedded = np.zeros((X.shape[0], embedding_dim))

        idx = 0
        for tweet in X:
            avg_embedding = np.zeros((embedding_dim))
            for word in tweet.split():
                if word in glove.keys():
                    avg_embedding += glove[word]

            avg_embedding = avg_embedding / len(tweet.split())
            X_embedded[idx] = avg_embedding
            idx += 1

        return X_embedded
