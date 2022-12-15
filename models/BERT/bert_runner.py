from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import json
import os
import random
import models.utils as utils
import configparser
import torch
import numpy as np
import models.BERT.bert_utils as bert_utils
from models import Runner
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from models.BERT.bert_dataset import TweetSentimentDataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


class BertRunner(Runner):
    """A Runner class for running off-the-shelf BERT models with MLP classification heads"""
    def __init__(self):
        super().__init__()
        self.config_ini = configparser.ConfigParser()
        dir = os.path.dirname(os.path.realpath(__file__))
        print("\n\n\nLoading config:\n")
        print(os.path.join(dir, 'config.ini'))
        self.config_ini.read(os.path.join(dir,'config.ini'))
        print(json.dumps({section: dict(
            self.config_ini[section]) for section in self.config_ini.sections()}, indent=4))
        print("\n\n")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config_ini['Basic']['ModelType'])
        self.config = AutoConfig.from_pretrained(self.config_ini['Basic']['ModelType'])
        self.model = AutoModelForSequenceClassification.from_pretrained(self.config_ini['Basic']['ModelType'], output_attentions = False, output_hidden_states = False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = float(self.config_ini['Basic']['LR']), eps = 1e-8)
        if self.config_ini.getboolean('Basic', 'LoadCheckpoint'):
            print('Loading checkpoint')
            checkpoint = torch.load(os.path.join(path_root, self.config_ini['Basic']['CheckpointDir'],self.config_ini['Basic']['CheckpointName']), map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.last_epoch = checkpoint['epoch']
        else:
            self.last_epoch = -1
        
    def train(self):
        tweets, labels = utils.load_tweets(os.path.join(path_root, self.config_ini['Basic']['DataDir'], self.config_ini['Basic']['DatasetPos']), os.path.join(path_root, self.config_ini['Basic']['DataDir'], self.config_ini['Basic']['DatasetNeg']))
        if self.config_ini['Basic']['BaseModel'] == 'roberta':
            labels[labels==1]=2
        X_train, X_dev, Y_train, Y_dev = train_test_split(tweets, labels, test_size=0.1, random_state = 1)
        dataset_train = TweetSentimentDataset(X_train.tolist(), Y_train, self.tokenizer)
        dataset_val = TweetSentimentDataset(X_dev.tolist(), Y_dev, self.tokenizer)
        train_dataloader = DataLoader(
            dataset_train,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            batch_size = int(self.config_ini['Basic']['BatchSize']))
        val_dataloader = DataLoader(
            dataset_val,
            num_workers=8,
            pin_memory=True,
            batch_size = int(self.config_ini['Basic']['BatchSize']))
        total_steps = len(train_dataloader) * int(self.config_ini['Basic']['MaxEpochs'])
        if self.config_ini['Basic'].getboolean('WarmupSteps'):
            warmup_steps = total_steps/10
        else:
            warmup_steps = 0
        scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                        num_warmup_steps = warmup_steps,
                                                        num_training_steps = total_steps,
                                                        last_epoch=self.last_epoch)
        if self.config_ini['Basic'].getboolean('OnlyClassifier'):
            if self.config_ini['Basic']['BaseModel'] == 'distilbert':
                for param in self.model.distilbert.parameters():
                    param.requires_grad = False
            elif self.config_ini['Basic']['BaseModel'] == 'roberta':
                for param in self.model.roberta.parameters():
                    param.requires_grad = False
        seed_val = 1
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        sm = torch.nn.Softmax(1)
        for epoch_i in tqdm(list(range(0, int(self.config_ini['Basic']['MaxEpochs']))), desc=" epochs", position=0):
            self.model.train()
            total_train_loss = 0
            for batch in tqdm(train_dataloader, desc=" iterations", position=0, leave=True, total=len(train_dataloader)):
                self.model.zero_grad()
                ii = batch[0].to(self.device)
                am = batch[1].to(self.device)
                l = batch[2].to(self.device)
                output = self.model(input_ids = ii, attention_mask = am ,labels=l)
                loss = output.loss
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                scheduler.step()
            avg_train_loss = total_train_loss / len(train_dataloader)
            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("")
            print("Running Validation...")
            self.model.eval()
            total_eval_loss = 0
            predictions = torch.Tensor().to(self.device)
            for batch in tqdm(val_dataloader, total = len(val_dataloader)):
                vii = batch[0].to(self.device)
                vam = batch[1].to(self.device)
                vl = batch[2].to(self.device)
                with torch.no_grad():
                    output = self.model(input_ids = vii, attention_mask = vam ,labels=vl)
                    loss = output.loss
                    logits = output.logits
                    total_eval_loss += loss.item()
                    scores = sm(logits)
                    scores = torch.argmax(scores, dim=1)
                    predictions = torch.cat((predictions, scores))
            avg_val_loss = total_eval_loss / len(val_dataloader)
            acc_score = accuracy_score(Y_dev, predictions.to('cpu'))
            f1 = f1_score(Y_dev, predictions.to('cpu'), average='weighted')
            print("Validation loss: ", avg_val_loss)
            print("Accuracy: " , acc_score)
            print("F1 Score: " , f1)
            cm = confusion_matrix(Y_dev, predictions.to('cpu'), normalize = "true")
            torch.save({
                        'epoch': epoch_i,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': avg_val_loss,
                        'conf_mat': cm,
                    }, os.path.join(path_root, self.config_ini['Basic']['CheckpointDir'],f"checkpoint_BERT_basic_{self.config_ini['Basic']['BaseModel']}_epoch_{epoch_i}.pth"))
    
    def evaluate(self):
        tweets = utils.load_test_data(os.path.join(path_root, self.config_ini['Basic']['DataDir'], self.config_ini['Basic']['TestData']))
        if self.config_ini['Basic']['BaseModel'] == 'roberta':
            pos_label = 2
        else:
            pos_label = 1
        predictions = bert_utils.classify_data(self.model, self.tokenizer, tweets, pos_label, 0, self.device)
        utils.create_submission_csv(os.path.join(path_root, self.config_ini['Basic']['SubmissionDir'],f"BERT_basic_{self.config_ini['Basic']['BaseModel']}_submission.csv"), predictions)