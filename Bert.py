import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, AutoTokenizer
from sklearn import metrics
import pandas as pd
import numpy as np

# fix random seed for reproducibility
seed = 30
#random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# check if you have a gpu or not
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# this class is defined to accept dataframe, tokenizer and max_len as input
# and generated tokenized output and tags that is used by the Bert for training
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len

    # return the size of the dataset
    def __len__(self):
        return self.len

    # support the indexing, dataset[i] is the ith sample
    def __getitem__(self, index):
        text = str(self.data.text[index])
        text = " ".join(text.split())
        # perfrom tokenization and generate the necessary outputs, namely
        # input_ids, attention_mask, token_type_ids
        inputs = self.tokenizer.encode_plus(
            text,  # the first sequence to be encoded, can be a string, a list of strings(tokenized)
            None,  # the second sequence to be encoded
            add_special_tokens=True,  # whether or not to addd the special tokens
            max_length=self.max_len,  # the maxinum length of the sequence
            padding='max_length',  # whether or not to pad the sequence to the max length
            truncation=True,  # whether or not to truncate sequence to max length
            return_token_type_ids=True  # whether to retun token type ids
        )
        ids = inputs['input_ids']  # indices of input sequence tokens in the vocab
        mask = inputs['attention_mask']  # mask to avoid performing attention pm padding token indices
        token_type_ids = inputs[
            'token_type_ids']  # segment token indices to indicate first and second portions of the inputs
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.data.label[index], dtype=torch.long)  # list of labels

        }


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, monitor='val_loss', delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last improvement.
                            Default: 7
            verbose (bool): If True, prints a message for each improvement.
                            Default: False
            monitor (string): The metric to qualify the performance of the model.
                            Default: val_loss
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_best = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val, model):
        if self.monitor == 'val_loss':
            score = -val
        else:
            score = val

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val, model)
            self.counter = 0

    def save_checkpoint(self, val, model):
        """Saves model when encountering an improvement ."""
        if self.verbose:
            if self.monitor == 'val_loss':
                self.trace_func(f'{self.monitor} decreased ({self.val_best:.6f} --> {val:.6f}).  Saving model ...')
            else:
                self.trace_func(f'{self.monitor} increased ({self.val_best:.6f} --> {val:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_best = val

# use different pretrained weights to initialize a bert model
def BertModel(pretrained_weights, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_weights)
    model = BertForSequenceClassification.from_pretrained(pretrained_weights,
                                                          num_labels=num_labels,
                                                          output_attentions=False,
                                                          output_hidden_states=False,
                                                          return_dict=False)

    return tokenizer, model

'''
since all the data from the dataset cannot be loaded to memory at once,
the amount of data loaded to to the memory and then passed to the NN 
should be controlled, this control is achived by using the parameters such
as batch_size and max_len
'''
def convert_data_into_features(data, tokenizer, max_len, batch_size, shuffle=True):
    dataset = CustomDataset(data, tokenizer, max_len)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

# train and save the best model on val set
def train(dataloader_train, dataloader_val, model, optimizer, train_args, early_stop_args):
    early_stopping = EarlyStopping(patience=early_stop_args['patience'], verbose=early_stop_args['verbose'],
                                   monitor=early_stop_args['monitor'], path=early_stop_args['model_path'])

    for epoch in range(1, train_args['epochs'] + 1):
        print(f'Epoch {epoch}')
        train_for_single_epoch(dataloader_train, model, optimizer)
        predictions, true_vals = evaluate(dataloader_val, model)
        val_f1 = metrics.f1_score(true_vals, predictions, average='macro')
        print(f'F1 score (macro) on val set: {val_f1}')
        early_stopping(val_f1, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break


def train_for_single_epoch(dataloader_train, model, optimizer):
    model.train()
    for i, data in enumerate(dataloader_train, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)
        outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids, labels=targets)
        loss = outputs[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


# evaluate the model
def evaluate(dataloader_val, model):
    model.eval()
    predictions, gold_standard = [], []
    for i, data in enumerate(dataloader_val, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)
        with torch.no_grad():
            outputs = model(ids, mask, token_type_ids, labels=targets)
    _, big_idx = torch.max(outputs[1].data, dim=1)
    predictions.extend(big_idx.tolist())
    gold_standard.extend(targets.tolist())
    return predictions, gold_standard

# use macro F1 score to evaluate the performance of the model
def show_performance(truth, pred):
    f1_score_macro = metrics.f1_score(truth, pred, average='macro')
    return f1_score_macro

# load and shuffle data
data = pd.read_csv('IMDB Dataset.csv')[:1000]
shuffled_data = data.sample(frac=1).reset_index(drop=True)
shuffled_data.rename(columns={'review':'text','sentiment':'label'},inplace=True)
shuffled_data.replace({'positive':1,'negative':0},inplace=True)
train_val_split = 0.8
train_set = shuffled_data[:int(len(shuffled_data)*train_val_split)]
val_set = shuffled_data[int(len(shuffled_data)*train_val_split):].reset_index(drop=True)

# define the parameters for training and early stop
early_stop_args = {'monitor':'val_f1','verbose':True,'patience':6,'model_path':'checkpoint.pt'}
train_args = {'learning_rate':1e-5,'batch_size':20,'epochs':2}
# define different pretrained weights
bert_pretrained_weights = 'bert-base-cased'
sciBert_pretrained_weights = 'allenai/scibert_scivocab_cased'
# the number of classes in your task
num_labels = 2
# max length of your sequence
max_len=150

tokenizer,model = BertModel(bert_pretrained_weights,num_labels)
optimizer = AdamW(model.parameters(),lr=train_args['learning_rate'],eps=1e-8)
model.to(device)

dataloader_train = convert_data_into_features(train_set,tokenizer,max_len=max_len,batch_size=train_args['batch_size'])
dataloader_val = convert_data_into_features(val_set,tokenizer,max_len=max_len,batch_size=train_args['batch_size'])

train(dataloader_train,dataloader_val,model,optimizer,train_args,early_stop_args)
# load the best model
model.load_state_dict(torch.load(early_stop_args['model_path'], map_location=torch.device('cpu')))
predictions, true_vals = evaluate(dataloader_train, model)