from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import json
import configparser
import os
import torch
import models.utils as utils
from models import Runner
import random
import numpy as np
import models.BERT.bert_utils as bert_utils
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel, RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup
from models.BERT.models import ClassifierBert, BiLSTMCNNHead, BiLSTMHead
from models.BERT.bert_dataset import TweetSentimentDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

class BiLSTMBertRunner(Runner):
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
        if self.config_ini['BiLSTM']['BaseModel'] == 'distilbert':
            print('Instantiating distilbert')
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased", output_attentions = False, output_hidden_states = False)
        elif self.config_ini['BiLSTM']['BaseModel'] == 'roberta':
            print('Instantiating roberta')
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.bert = RobertaModel.from_pretrained('roberta-base', output_attentions = False, output_hidden_states = False)
        if self.config_ini['BiLSTM'].getboolean('UseCNN'):
            print('Instantiating CNN head')
            head = BiLSTMCNNHead()
        else:
            print('Instantiating MLP head')
            head = BiLSTMHead()
        self.model = ClassifierBert(self.bert, head)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = float(self.config_ini['BiLSTM']['LR']), eps = 1e-8)
        if self.config_ini.getboolean('BiLSTM', 'LoadCheckpoint'):
            print('Loading checkpoint')
            checkpoint = torch.load(os.path.join(path_root, self.config_ini['BiLSTM']['CheckpointDir'],self.config_ini['BiLSTM']['CheckpointName']), map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.last_epoch = checkpoint['epoch']
        else:
            self.last_epoch = -1
    
    def train(self):
        tweets, labels = utils.load_tweets(os.path.join(path_root, self.config_ini['BiLSTM']['DataDir'], self.config_ini['BiLSTM']['DatasetPos']), os.path.join(path_root, self.config_ini['BiLSTM']['DataDir'], self.config_ini['BiLSTM']['DatasetNeg']))
        X_train, X_dev, Y_train, Y_dev = train_test_split(tweets, labels, test_size=0.1, random_state = 1)
        dataset_train = TweetSentimentDataset(X_train.tolist(), Y_train, self.tokenizer)
        dataset_val = TweetSentimentDataset(X_dev.tolist(), Y_dev, self.tokenizer)
        train_dataloader = DataLoader(
            dataset_train,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            batch_size = int(self.config_ini['BiLSTM']['BatchSize']))
        val_dataloader = DataLoader(
            dataset_val,
            num_workers=8,
            pin_memory=True,
            batch_size = int(self.config_ini['BiLSTM']['BatchSize']))
        total_steps = len(train_dataloader) * int(self.config_ini['BiLSTM']['MaxEpochs'])
        if self.config_ini['BiLSTM'].getboolean('WarmupSteps'):
            warmup_steps = total_steps/10
        else:
            warmup_steps = 0
        scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                    num_warmup_steps = warmup_steps,
                                                    num_training_steps = total_steps,
                                                    last_epoch=self.last_epoch)
        if self.config_ini['BiLSTM'].getboolean('OnlyClassifier'):
            for param in self.model.bert.parameters():
                param.requires_grad = False
        seed_val = 1
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        sm = torch.nn.Softmax(1)
        criterion = torch.nn.BCELoss()
        for epoch_i in tqdm(list(range(0, int(self.config_ini['BiLSTM']['MaxEpochs']))), desc=" epochs", position=0):
            self.model.train()
            total_train_loss = 0
            for batch in tqdm(train_dataloader, desc=" iterations", position=0, leave=True, total=len(train_dataloader)):
                self.model.zero_grad()
                ii = batch[0].to(self.device)
                am = batch[1].to(self.device)
                l = batch[2].to(self.device)
                output = self.model(input_ids = ii, attention_mask = am)
                loss = criterion(sm(output),torch.nn.functional.one_hot(l).float())
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
                    output = self.model(input_ids = vii, attention_mask = vam)
                    loss = criterion(sm(output),torch.nn.functional.one_hot(vl).float())
                    total_eval_loss += loss.item()
                    scores = sm(output)
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
                    }, os.path.join(path_root, self.config_ini['BiLSTM']['CheckpointDir'],f"checkpoint_BERT_BiLSTM_{self.config_ini['BiLSTM']['BaseModel']}_epoch_{epoch_i}.pth"))
    
    def evaluate(self):
        tweets = utils.load_test_data(os.path.join(path_root, self.config_ini['BiLSTM']['DataDir'], self.config_ini['BiLSTM']['TestData']))
        predictions = bert_utils.classify_data_bilstm(self.model, self.tokenizer, tweets, 1, 0, self.device)
        utils.create_submission_csv(os.path.join(path_root, self.config_ini['BiLSTM']['SubmissionDir'],f"BERT_BiLSTM_{self.config_ini['BiLSTM']['BaseModel']}_submission.csv"), predictions)
