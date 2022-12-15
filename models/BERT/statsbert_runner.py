from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import json
import configparser
from models import Runner
import os
import torch
import models.utils as utils
import random
import numpy as np
import models.BERT.bert_utils as bert_utils
from tqdm import tqdm
from models.lda import LDA_Model
from transformers import DistilBertTokenizer, DistilBertModel, RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from models.BERT.models import StatsBert
from models.BERT.bert_dataset import TweetSentimentDatasetStats
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

class StatsBertRunner(Runner):
    def __init__(self):
        self.config_ini = configparser.ConfigParser()
        dir = os.path.dirname(os.path.realpath(__file__))
        print("\n\n\nLoading config:\n")
        print(os.path.join(dir, 'config.ini'))
        self.config_ini.read(os.path.join(dir,'config.ini'))
        print(json.dumps({section: dict(
            self.config_ini[section]) for section in self.config_ini.sections()}, indent=4))
        print("\n\n")
        if self.config_ini['Stats']['BaseModel'] == 'distilbert':
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased", output_attentions = False, output_hidden_states = False)
        elif self.config_ini['Stats']['BaseModel'] == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.bert = RobertaModel.from_pretrained('roberta-base', output_attentions = False, output_hidden_states = False)
        self.model = StatsBert(self.bert, self.config_ini['Stats'].getboolean('UseVader'), self.config_ini['Stats'].getboolean('UseLDA'), self.config_ini['Stats'].getboolean('UseStats'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = float(self.config_ini['Stats']['LR']), eps = 1e-8)
        if self.config_ini.getboolean('Stats', 'LoadCheckpoint'):
            print("Loading checkpoint")
            checkpoint = torch.load(os.path.join(path_root, self.config_ini['Stats']['CheckpointDir'],self.config_ini['Stats']['CheckpointName']), map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.last_epoch = checkpoint['epoch']
            self.lda = LDA_Model(save_suffix='final')
            self.lda.load_lda()
        else:
            self.last_epoch = -1
    
    def train(self):
        tweets, labels = utils.load_tweets(os.path.join(path_root, self.config_ini['Stats']['DataDir'], self.config_ini['Stats']['DatasetPos']), os.path.join(path_root, self.config_ini['Stats']['DataDir'], self.config_ini['Stats']['DatasetNeg']))
        if self.config_ini['Stats'].getboolean('UseStats'):
            stats = utils.load_stats(os.path.join(path_root, self.config_ini['Stats']['DataDir'], self.config_ini['Stats']['StatsPos']), os.path.join(path_root, self.config_ini['Stats']['DataDir'], self.config_ini['Stats']['StatsNeg']))
            X_train, X_dev, stats_train, stats_dev, Y_train, Y_dev = train_test_split(tweets, stats, labels, test_size=0.1, random_state = 1)
        else:
            X_train, X_dev,Y_train, Y_dev = train_test_split(tweets, labels, test_size=0.1, random_state = 1)
            stats_train = None
            stats_dev = None
        if self.config_ini['Stats'].getboolean('UseVader'):
            vader_train = utils.parse_vader(X_train)
            vader_dev = utils.parse_vader(X_dev)
        else:
            vader_train = None
            vader_dev = None
        if self.config_ini['Stats'].getboolean('UseLDA'):
            self.lda = LDA_Model(save_suffix='runner')
            lda_train = self.lda.train(X_train)
            lda_dev = np.array(self.lda.process_new_data(X_dev))
        else:
            lda_train = None
            lda_dev = None
        dataset_train = TweetSentimentDatasetStats(X_train.tolist(), Y_train, self.tokenizer, vader_train, lda_train, stats_train)
        dataset_val = TweetSentimentDatasetStats(X_dev.tolist(), Y_dev, self.tokenizer, vader_dev, lda_dev, stats_dev)
        train_dataloader = DataLoader(
            dataset_train,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            batch_size = int(self.config_ini['Stats']['BatchSize']))
        val_dataloader = DataLoader(
            dataset_val,
            num_workers=8,
            pin_memory=True,
            batch_size = int(self.config_ini['Stats']['BatchSize']))
        total_steps = len(train_dataloader) * int(self.config_ini['Stats']['MaxEpochs'])
        if self.config_ini['Stats'].getboolean('WarmupSteps'):
            warmup_steps = total_steps/10
        else:
            warmup_steps = 0
        scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                    num_warmup_steps = warmup_steps,
                                                    num_training_steps = total_steps,
                                                    last_epoch=self.last_epoch)
        if self.config_ini['Stats'].getboolean('OnlyClassifier'):
            for param in self.model.bert.parameters():
                param.requires_grad = False
        seed_val = 1
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        sm = torch.nn.Softmax(1)
        criterion = torch.nn.BCELoss()
        for epoch_i in tqdm(list(range(0, int(self.config_ini['Stats']['MaxEpochs']))), desc=" epochs", position=0):
            self.model.train()
            total_train_loss = 0
            for batch in tqdm(train_dataloader, desc=" iterations", position=0, leave=True, total=len(train_dataloader)):
                self.model.zero_grad()
                ii = batch[0].to(self.device)
                am = batch[1].to(self.device)
                vader = batch[2].to(self.device)
                lda_b = batch[3].to(self.device)
                stats_b = batch[4].to(self.device)
                l = batch[5].to(self.device)
                output = self.model(input_ids = ii, attention_mask = am, vader=vader, lda=lda_b, stats=stats_b)
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
                vader = batch[2].to(self.device)
                lda_b = batch[3].to(self.device)
                stats_b = batch[4].to(self.device)
                vl = batch[5].to(self.device)
                with torch.no_grad():
                    output = self.model(input_ids = vii, attention_mask = vam, vader=vader, lda=lda_b, stats=stats_b)
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
                    }, os.path.join(path_root, self.config_ini['Stats']['CheckpointDir'],f"checkpoint_BERT_stats_{self.config_ini['Stats']['BaseModel']}_epoch_{epoch_i}.pth"))
    
    def evaluate(self):
        tweets = utils.load_test_data(os.path.join(path_root, self.config_ini['Stats']['DataDir'], self.config_ini['Stats']['TestData']))
        if self.config_ini['Stats'].getboolean('UseVader'):
            vader = utils.parse_vader(tweets)
        else:
            vader = None
        if self.config_ini['Stats'].getboolean('UseLDA'):
            lda_in = np.array(self.lda.process_new_data(tweets))
        else:
            lda_in = None
        if self.config_ini['Stats'].getboolean('UseStats'):
            stats = utils.load_test_stats(os.path.join(path_root, self.config_ini['Stats']['DataDir'],self.config_ini['Stats']['StatsTest']))
        else:
            stats = None
        predictions = bert_utils.classify_data_stats(self.model, self.tokenizer, tweets, vader, lda_in, stats, 1, 0, self.device)
        utils.create_submission_csv(os.path.join(path_root, self.config_ini['Stats']['SubmissionDir'],f"BERT_stats_{self.config_ini['Stats']['BaseModel']}_submission.csv"), predictions)
