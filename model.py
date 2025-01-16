from transformers import TrainingArguments,Trainer
from transformers import BertForSequenceClassification,BertTokenizer,BertConfig
from transformers import T5Tokenizer, T5ForSequenceClassification,T5Config
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification,GPT2Config
from transformers import ElectraTokenizer, ElectraForSequenceClassification,ElectraConfig

import torch
import os


FILE_DIR = os.path.dirname(os.path.abspath(__file__))

def import_model(model_name, tasks, train=True, eval=True):
    '''
        model_name: str, one of ['bert', 't5', 'gpt2', 'electra', 'tiny-bert']
        tasks: str, one of ['RTE', 'CoLA', 'SST']
        train: bool, whether to train
        eval: bool, whether to evaluate
        return: model, tokenizer, config, sizeof_model
    '''
    try:
        assert model_name in ['bert', 't5', 'gpt2', 'electra', 'tiny-bert']
    except AssertionError:
        raise ValueError(f"Model {model_name} is not supported. Model must be one of ['bert', 't5', 'gpt2', 'electra', 'tiny-bert']")
    
    if not train and not eval:
        return None
    
    if train:
        model_path = os.path.join(FILE_DIR, 'checkpoints', model_name, 'config')
        vocab_path = model_path
    else:
        model_path = os.path.join(FILE_DIR, 'checkpoints', model_name, f'saved_model_{tasks}')
        vocab_path = os.path.join(FILE_DIR, 'checkpoints', model_name, 'config')
        
    if model_name == 'bert':
        model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
        tokenizer = BertTokenizer.from_pretrained(vocab_path)
        config = BertConfig.from_pretrained(model_path)
    
    elif model_name == 't5':
        config = T5Config.from_pretrained(model_path)
        config.num_labels = 2
        tokenizer = T5Tokenizer.from_pretrained(vocab_path, legacy=False)
        model = T5ForSequenceClassification.from_pretrained(model_path, config=config)
    
    elif model_name == 'gpt2':
        config = GPT2Config.from_pretrained(model_path)
        config.num_labels = 2
        tokenizer = GPT2Tokenizer.from_pretrained(vocab_path)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = tokenizer.get_vocab()[tokenizer.pad_token]
        config.pad_token_id = tokenizer.pad_token_id
        model = GPT2ForSequenceClassification.from_pretrained(model_path, config=config)
        model.resize_token_embeddings(len(tokenizer))
        
    elif model_name == 'electra':
        config = ElectraConfig.from_pretrained(model_path)
        config.num_labels = 2
        tokenizer = ElectraTokenizer.from_pretrained(vocab_path, legacy=False)
        model = ElectraForSequenceClassification.from_pretrained(model_path, config=config)
    
    elif model_name == 'tiny-bert':
        max_length = 512
        config = BertConfig.from_pretrained(model_path)
        config.num_labels = 2
        tokenizer = BertTokenizer.from_pretrained(vocab_path, model_max_length=max_length)
        model = BertForSequenceClassification.from_pretrained(model_path, config=config)
        
    num_params = sum(p.numel() for p in model.parameters())
    
        
    return model, tokenizer, config, num_params*4/(1024**2)