import torch

class ClassifierBert(torch.nn.Module):
    """Basic BERT classifier model"""
    def __init__(self, bert, head):
        """Class constructor
        
        Args:
            bert (torch.nn.Module): A BERT-based model used to generate embeddings
            head (torch.nn.Module): A BiLSTM-based model used to generate predictions
        """
        super().__init__()
        self.bert = bert
        self.head = head
    
    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids, attention_mask)[0]
        x = self.head(x)
        return x

class StatsBert(torch.nn.Module):
    """BERT classifier model using dataset statistics, lexical features and LDA topic vectors"""
    def __init__(self, bert, vader=False, lda=False, stats=False, dropout = 0.5, input_size = 768, hidden_size = 128, preclassifier_size = 32, output_dim = 2):
        """Class constructor
        
        Args:
            bert (torch.nn.Module): A BERT-based model used to generate embeddings
            vader (bool): Boolean value which indicates whether the model should use lexical features
            lda (bool): Boolean value which indicates whether the model should use LDA topic vectors
            stats (bool): Boolean value which indicates whether the model should use statistical features
            dropout (float): Dropout probability
            input_size (int): Size of the embedding vectors provided to the BiLSTM
            hidden_size (int): Size of the hidden state in the BiLSTM cell
            preclassifier_size (int): Size of the output of the first fully connected layer
            output_dim (int): Dimension of the output
        """
        
        super().__init__()
        self.bert = bert
        self.bert_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.preclassifier_size = preclassifier_size
        self.final_linear = self.preclassifier_size
        self.vader = vader
        self.lda = lda
        self.stats = stats
        if vader:
            self.final_linear+=10
        if lda:
            self.final_linear+=10
        if stats:
            self.final_linear+=10
        self.bilstm = torch.nn.LSTM(input_size = self.bert_size, hidden_size = self.hidden_size, batch_first = True, dropout = dropout, bidirectional = True)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p = dropout)
        self.preclassifier = torch.nn.Linear(2*self.hidden_size, self.preclassifier_size)
        self.classifier = torch.nn.Linear(self.final_linear, self.output_dim)
    
    def forward(self, input_ids, attention_mask, vader=None, lda=None, stats=None):
        x = self.bert(input_ids = input_ids, attention_mask = attention_mask)[0]
        _, (_, x)= self.bilstm(x)
        x = x.permute(1,0,2)
        x = x.contiguous().view(-1, 2*self.hidden_size)
        x = self.dropout(x)
        x = self.preclassifier(x)
        x = self.relu(x)
        x = self.dropout(x)
        if self.vader:
            x = torch.cat((x,vader), 1)
        if self.lda:
            x = torch.cat((x,lda), 1)
        if self.stats:
            x = torch.cat((x,stats), 1)
        x = x.float()
        x = self.classifier(x)
        return x

class BiLSTMHead(torch.nn.Module):
    """BiLSTM classification head with a fully-connected layer"""
    def __init__(self, dropout=0.3, input_size = 768, hidden_size = 128, output_dim = 2):
        """Class constructor
        
        Args:
            dropout (float): Dropout probability
            input_size (int): Size of the embedding vectors provided to the BiLSTM
            hidden_size (int): Size of the hidden state in the BiLSTM cell
            output_dim (int): Dimension of the output
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.bilstm = torch.nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, batch_first = True, dropout = dropout, bidirectional = True)
        self.dropout = torch.nn.Dropout(p = dropout)
        self.linear = torch.nn.Linear(2*self.hidden_size, self.output_dim)
    
    def forward(self, input):
        _, (_, x)= self.bilstm(input)
        x = x.permute(1,0,2)
        x = x.contiguous().view(-1, 2*self.hidden_size)
        x = self.dropout(x)
        x = self.linear(x)
        return x

class BiLSTMCNNHead(torch.nn.Module):
    """BiLSTM classification head with a convolutional layers and a fully-connected layer"""
    def __init__(self, dropout=0.3, input_size = 768, hidden_size = 128, output_dim = 2):
        """Class constructor
        
        Args:
            dropout (float): Dropout probability
            input_size (int): Size of the embedding vectors provided to the BiLSTM
            hidden_size (int): Size of the hidden state in the BiLSTM cell
            output_dim (int): Dimension of the output
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.bilstm = torch.nn.LSTM(input_size = self.bert_size, hidden_size = self.hidden_size, batch_first = True, dropout = dropout, bidirectional = True)
        self.conv1 = torch.nn.Conv1d(2, 16, 3, 2)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p = dropout)
        self.pool1 = torch.nn.MaxPool1d(2,2)
        self.conv2 = torch.nn.Conv1d(16, 32, 3, 2)
        self.pool2 = torch.nn.MaxPool1d(2,2)
        self.linear = torch.nn.Linear(32*7, self.output_dim)
    
    def forward(self, input):
        _, (_, x)= self.bilstm(input)
        x = x.permute(1,0,2)
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool2(x)
        x = x.contiguous().view(-1, 32*7)
        x = self.linear(x)
        return x