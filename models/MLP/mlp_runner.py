from models.MLP.models import MLP
import models.utils as utils
from models import Runner

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import csv
import numpy as np

import os
import configparser
import imp
from pathlib import Path
import sys
import json

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))


class MlpRunner(Runner):
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

       # set up model
        print("loading glove embeddings...")
        self.glove = self.__glove2dict(self.config_ini['Basic']['GlovePath'])
        print("done.")

        self.model = MLP(
            int(self.config_ini['Model']['EmbeddingDimension']),
            int(self.config_ini['Model']['Hidden1']),
            int(self.config_ini['Model']['Hidden2']),
            int(self.config_ini['Model']['Hidden3']),
            2,
            float(self.config_ini['Model']['Dropout'])
        )

    def train(self):
        print("loading data for training...")
        tweets, labels = utils.load_tweets(
            os.path.join(
                path_root, self.config_ini['Basic']['DataDir'], self.config_ini['Basic']['DatasetPos']),
            os.path.join(
                path_root, self.config_ini['Basic']['DataDir'], self.config_ini['Basic']['DatasetNeg'])
        )
        X_train_raw, X_dev_raw, Y_train, Y_dev = train_test_split(
            tweets, labels, test_size=0.1, random_state=1)

        X_train_embedded = self.__embed(X_train_raw, self.glove, int(self.config_ini['Model']['EmbeddingDimension']))
        X_dev_embedded = self.__embed(X_dev_raw, self.glove, int(self.config_ini['Model']['EmbeddingDimension']))

        batch_size = int(self.config_ini['Model']['BatchSize'])
        num_epochs = int(self.config_ini['Model']['NumEpochs'])

        train = list(zip(X_train_embedded, Y_train))
        val = list(zip(X_dev_embedded, Y_dev))
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            val, batch_size=batch_size, shuffle=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("running on: " + str(device))
        self.model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config_ini['Model']['LR']))
        train_acc, val_acc = [], []

        print("starting with training...")
        for epoch in range(num_epochs):
            for i, (tweets, labels) in enumerate(train_loader):
                tweets = tweets.to(device).float()
                labels = labels.type(torch.LongTensor).to(device)

                pred = self.model(tweets)
                loss = criterion(pred, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i+1) % 1000 == 0:
                    print(
                        f'\r epoch {epoch+1}/{num_epochs}, step {i+1}, loss={loss.item():.5f}')
        if epoch%5==4:
                train_acc.append(self.__accuracy(self.model, train_loader))
                val_acc.append(self.__accuracy(self.model, val_loader))

    def evaluate(self):
        print("evaluating on test data...")
        tweets = utils.load_test_data(os.path.join(path_root, self.config_ini['Basic']['DataDir'], self.config_ini['Basic']['TestData']))
        X_embedded = torch.tensor(self.__embed(np.array(tweets), self.glove, int(self.config_ini['Model']['EmbeddingDimension'])))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        predictions = []
        with torch.no_grad():
            for i in range(X_embedded.shape[0]):
                tweet = X_embedded[i:i+1,].to(device).float()
                output = self.model(tweet)
                _,pred = torch.max(output,1)
                predictions.append(pred.item())
            
        utils.create_submission_csv(f"MLP_submission.csv", predictions)

    
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

            avg_embedding = avg_embedding / (len(tweet.split()) + 1e-8)
            X_embedded[idx] = avg_embedding
            idx += 1

        return X_embedded

    def __accuracy(self, model, loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        predictions = torch.Tensor().to(device)
        Y_dev = torch.Tensor().to(device)
        with torch.no_grad():
            correct = 0
            samples = 0
            for tweets, labels in loader:
                tweets = tweets.to(device).float()
                labels = labels.type(torch.LongTensor).to(device)
                outputs = model(tweets)
                _, pred = torch.max(outputs, 1)
                predictions = torch.cat((predictions, pred))
                Y_dev = torch.cat((Y_dev, labels))
                samples += labels.shape[0]
                correct += (pred == labels).sum().item()
            acc = 100.0*correct/samples
            print('accuracy: ', acc)
            print('F1: ', f1_score(predictions.cpu(), Y_dev.cpu()))
        model.train()
        return acc
