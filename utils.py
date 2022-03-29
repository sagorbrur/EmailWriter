import os
import re
import random
import torch
import numpy as np
import config as cfg
from transformers import AutoTokenizer, AutoConfig, AutoModelForPreTraining

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_tokenizer(special_tokens=None):
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL)

    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
    return tokenizer

def get_model(tokenizer, special_tokens=None, load_model_path=None):

    if special_tokens:
        config = AutoConfig.from_pretrained(
            cfg.MODEL, 
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            sep_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
            output_hidden_states=False
        )
    else: 
        config = AutoConfig.from_pretrained(
            cfg.MODEL,                                     
            pad_token_id=tokenizer.eos_token_id,
            output_hidden_states=False
        )    

    model = AutoModelForPreTraining.from_pretrained(cfg.MODEL, config=config)

    if special_tokens:
        #Special tokens added, model needs to be resized accordingly
        model.resize_token_embeddings(len(tokenizer))

    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))

    model.cuda()
    return model

def remove_special_token(text):
    for key, token in cfg.SPECIAL_TOKENS.items():
        text = text.replace(token, '')
    return text

def start_with_greet(text, greet):
    for g in greet:
        if text.startswith(g):
            return True, g
    return False, None

def post_processing(text, prompt):
    # remove prompt first
    text = text.replace(prompt, '')
    text = text.strip()
    gexist, greet = start_with_greet(text, cfg.GREET_TOKENS)
    if gexist:
        regex = f"^{greet}\s?(\w+)?[\.,\s]?"
        start_greet = greet + " [name]\n"
        text = re.sub(regex, start_greet, text)
        text = text + cfg.CONCLUSION_GREET
    else:
        text = cfg.START_GREET + text + cfg.CONCLUSION_GREET
    return text
