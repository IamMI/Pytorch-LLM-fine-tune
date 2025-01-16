import torch
import argparse
from torch.utils.data import DataLoader
from torch.optim import AdamW
import os

import utils
import model
import train_eval

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

def main(args):
    model_name = args.model
    task = args.task
    batch_size = args.batch_size
    ifcuda = args.cuda
    iftrain = args.train
    ifeval = args.eval
    save_path = os.path.join(FILE_DIR, 'checkpoints', model_name, f'saved_model_{task}')
    
    
    my_model, tokenizer, config, sizeof_model = model.import_model(model_name, task, iftrain, ifeval)
    dataset = utils.build_datasets(task, tokenizer)
    
    print(f"Model: {model_name}, size: {round(sizeof_model, 2)}MB, task: {task}")
    
    train_dataloader = DataLoader(dataset['train'],
                                  batch_size=batch_size, 
                                  shuffle=True)
    val_dataloader = DataLoader(dataset['validation'],
                                batch_size=batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() and ifcuda else "cpu")
    my_model.to(device)
    optimizer = AdamW(my_model.parameters(), lr=5e-5)
    
    if iftrain:
        train_eval.train(train_dataloader, my_model, optimizer, device, save_path)
    if ifeval:
        avg_eval_loss, accuracy = train_eval.eval(val_dataloader, my_model, device)
        print('-'*9 + 'Eval Finish' + '-'*9)
        print(f"Evaluation Loss: {avg_eval_loss}")
        print(f"Accuracy: {accuracy}")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=True, help='Whether fine-tune model or not')
    parser.add_argument('--eval', type=bool, default=False, help='Whether eval model or not')
    parser.add_argument('--model', type=str, default='t5', help='Choose a model from [bert, t5, gpt2, electra, tiny-bert]')
    parser.add_argument('--task', type=str, default='RTE', help='Choose a task from [RTE, CoLA, SST]')
    parser.add_argument('--cuda', type=bool, default=True, help='Whether support cuda or not')
    parser.add_argument('--batch_size', type=int, default=4, help='Choose batch_size when training')
    args = parser.parse_args()
    
    for model_name in ['bert', 'electra', 'gpt2', 't5', 'tiny-bert']:
        os.makedirs(os.path.join(FILE_DIR, 'checkpoints', model_name, 'config'), exist_ok=True)
        for task in ['SST', 'RTE', 'CoLA']:
            os.makedirs(os.path.join(FILE_DIR, 'checkpoints', model_name, f'saved_model_{task}'), exist_ok=True)
            
    for task in ['SST', 'RTE', 'CoLA']:
        os.makedirs(os.path.join(FILE_DIR, 'tasks', task), exist_ok=True)
        
    print("Please make sure you have downloaded model parameters and datasets! Otherwise please turn to README.md!")
    input("Press to continue!")
    
    
    main(args)