import torch
from torch.utils.data import Dataset

class TweetSentimentDataset(Dataset):
    """A pytorch Dataset class for working with BERT-based classifier models"""
    def __init__(self, tweets, labels, tokenizer):
        """Class Constructor

        Args:
            tweets (list): List of tweets that constitute the dataset
            labels (list): List of labels corresponding to the tweets
            tokenizer (transformers.PreTrainedTokenizer): BERT's tokenizer
        """
        self.labels = labels
        self.tokenizer = tokenizer
        self.tokenized_tweets = self.tokenizer(tweets, return_tensors='pt', max_length = 64, padding = 'max_length', truncation=True)

    def __len__(self):
        return len(self.tokenized_tweets['input_ids'])

    def __getitem__(self, idx):
          return self.tokenized_tweets['input_ids'][idx], self.tokenized_tweets['attention_mask'][idx], self.labels[idx]

class TweetSentimentDatasetStats(TweetSentimentDataset):
    """
    A pytorch Dataset class for working with BERT-based classifier models that utilize statistical features,
    LDA topic vectors and lexical features
    """
    def __init__(self, tweets, labels, tokenizer, vader = None, lda = None, stats = None):
        """Class Constructor

        Args:
            tweets (list): List of tweets that constitute the dataset
            labels (list): List of labels corresponding to the tweets
            tokenizer (transformers.PreTrainedTokenizer): BERT's tokenizer
            vader (iterable): Iterable of vader intensity scores corresponding to tweets
            lda (iterable): Iterable of LDA topic vectors corresponding to tweets
            stats (iterable): Iterable of statistical features corresponding to tweets
        """
        super().__init__(tweets, labels, tokenizer)
        self.vader = False
        self.lda = False
        self.stats = False
        if not vader is None:
            self.vader = True
            self.vader_labels = vader
        if not lda is None:
            self.lda = True
            self.lda_labels = lda
        if not stats is None:
            self.stats = True
            self.stats_labels = stats

    def __len__(self):
        return len(self.tokenized_tweets['input_ids'])

    def __getitem__(self, idx):
        input_ids, attention_mask, label = super().__getitem__(idx)
        vader = 0
        lda = 0
        stats = 0
        if self.vader:
            vader = self.vader_labels[idx]
        if self.lda:
            lda = self.lda_labels[idx]
        if self.stats:
            stats = self.stats_labels[idx]
        return input_ids, attention_mask, torch.Tensor([vader]*10), lda, stats, label
