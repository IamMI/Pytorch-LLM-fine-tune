# Pytorch examples of LLM fine-tune
## Introduction
    Demo have built five models and used three datasets to fine-tune, evaluate. 
    1. Five models:
        - Bert
        - GPT2
        - T5
        - Electra
        - Tiny-bert
    2. Three datasets:
        - SST-2(The Stanford Sentiment Treebank)
        - CoLA(The Corpus of Linguistic Acceptability)
        - RTE(Recognizing Textual Entailment)
    
    We use dataset to fine-tune the pre-trained model and evaluate it using three different datasets respectively.


## Model parameters preparetion
    It is hard to pre-train our Large model at our host, and I recommend you to downloaded model parameters from Hugging Face! Hugging Face is a GPL website where you could obtain any pre-trained model parameters you want. The parameters in demo come from Hugging Face. 
    
    Hugging Face:<https://huggingface.co/>

    Demo needs five models, for example, when you want to downloaded parameters of Bert, please search "BERT" and click "google-bert/bert-base-uncased", turn to "Files and versions" where you can find a lot of files, which are important to initial our model.

    It is common that pre-trained parameters are stored in "pytorch_model.bin", configurations of model are stored in "config.json" and vocabulary is stored in "vocab.txt". Thus you should downloaded them and place them under "checkpoints/bert/config".

    However, some of repositories do not contain "vocab.txt" or another file, maybe the owner have renamed them or this model does not need that file, so you can ask GPT when you are confused!

## Datasets preparetion
    All datasets come from GLUE, you can visit their website and click "Tasks" to find three datasets. Make sure to downloaded and place them under "tasks/".

    GLUE:<https://gluebenchmark.com/tasks> 

## Code structure
    We could import pre-trained model using package [transformer], which is made for Hugging Face. Click below to learn more about [transformer].

    transformer:https://huggingface.co/docs/transformers/main/zh/training
    
     




    


