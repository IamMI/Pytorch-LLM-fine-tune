import pandas as pd
from datasets import Dataset, DatasetDict
import os
import numpy as np
from sklearn.model_selection import train_test_split

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

def build_datasets(task, tokenizer, train_num: int=3200, test_ratio: float=0.5):
    '''
        task: str, one of ['RTE', 'CoLA', 'SST']
        tokenizer: tokenizer object
        train_num: int, number of training samples
        test_ratio: float, ratio of test samples
        return: tokenized_datasets
    '''
    try:
        assert task in ['RTE', 'CoLA', 'SST']
    except AssertionError:
        raise ValueError(f"Task {task} is not supported. Task must be one of ['RTE', 'CoLA', 'SST-2']")
    
    task_path = os.path.join(FILE_DIR, 'tasks', task)
    if task == 'RTE':
        return RTE_datasets(task_path, tokenizer, train_num, test_ratio)
    elif task == 'CoLA':
        return CoLA_datasets(task_path, tokenizer, train_num, test_ratio)
    elif task == 'SST':
        return SST_datasets(task_path, tokenizer, train_num, test_ratio)
    

def df2datasets(train_df, val_df, test_df):
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    dataset = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    return dataset

def RTE_datasets(task_path, tokenizer, train_num: int=3200, test_ratio: float=0.5):
    # 读取 train.tsv 和 dev.tsv
    train_df = pd.read_csv(os.path.join(task_path, 'train.tsv'), delimiter='\t')
    dev_df = pd.read_csv(os.path.join(task_path, 'dev.tsv'), delimiter='\t')

    label_map = {'entailment': 0, 'not_entailment': 1}
    train_df['label'] = train_df['label'].map(label_map)
    dev_df['label'] = dev_df['label'].map(label_map)
    train_df = train_df.dropna(subset=['label'])
    dev_df = dev_df.dropna(subset=['label'])
    train_df['label'] = train_df['label'].astype(int)
    dev_df['label'] = dev_df['label'].astype(int)

    train_df = train_df.head(train_num)  
    val_df, test_df = train_test_split(dev_df, test_size=test_ratio, random_state=42)
    
    dataset = df2datasets(train_df, val_df, test_df)
    
    def preprocess_function(examples):
        return tokenizer(examples['sentence1'], examples['sentence2'], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    tokenized_datasets= tokenized_datasets.remove_columns(["__index_level_0__"])
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "index"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    
    return tokenized_datasets
    
def CoLA_datasets(task_path, tokenizer, train_num: int=3200, test_ratio: float=0.5):
    # 读取 train.tsv 和 dev.tsv
    train_df = pd.read_csv(os.path.join(task_path, 'train.tsv'), delimiter='\t', header=None, names=['sentence_id', 'label', 'star', 'sentence'])
    dev_df = pd.read_csv(os.path.join(task_path, 'dev.tsv'), delimiter='\t', header=None, names=['sentence_id', 'label', 'star', 'sentence'])

    train_df = train_df.head(train_num)  
    val_df, test_df = train_test_split(dev_df, test_size=test_ratio, random_state=42)

    train_df = train_df[['label', 'sentence']]
    val_df = val_df[['label', 'sentence']]
    test_df = test_df[['label', 'sentence']]

    dataset = df2datasets(train_df, val_df, test_df)
    
    def preprocess_function(examples):
        return tokenizer(examples['sentence'], padding="max_length", truncation=True)
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    tokenized_datasets['validation'] = tokenized_datasets['validation'].remove_columns(["__index_level_0__"])
    tokenized_datasets['test'] = tokenized_datasets['test'].remove_columns(['__index_level_0__'])
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    
    return tokenized_datasets

def SST_datasets(task_path, tokenizer, train_num: int=3200, test_ratio: float=0.5):
    # 读取 train.tsv 和 dev.tsv
    train_df = pd.read_csv(os.path.join(task_path, 'train.tsv'), delimiter='\t')
    dev_df = pd.read_csv(os.path.join(task_path, 'dev.tsv'), delimiter='\t')

    train_df = train_df.head(train_num)  
    val_df, test_df = train_test_split(dev_df, test_size=test_ratio, random_state=42)

    train_df = train_df[['label', 'sentence']]
    val_df = val_df[['label', 'sentence']]
    test_df = test_df[['label', 'sentence']]

    dataset = df2datasets(train_df, val_df, test_df)
    
    def preprocess_function(examples):
        return tokenizer(examples['sentence'], padding="max_length", truncation=True)
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    tokenized_datasets['validation'] = tokenized_datasets['validation'].remove_columns(["__index_level_0__"])
    tokenized_datasets['test'] = tokenized_datasets['test'].remove_columns(['__index_level_0__'])
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    
    return tokenized_datasets

if __name__=='__main__':
    build_datasets('RTE', None)