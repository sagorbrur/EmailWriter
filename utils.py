import os
import re
import random
import torch
import numpy as np
import config as cfg
from transformers import AutoTokenizer, AutoConfig, AutoModelForPreTraining

def seed_everything(seed):
    """initialize random seed

    Args:
        seed (int): random seed value
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_tokenizer(special_tokens=None):
    """Get tokenizer

    Args:
        special_tokens (dict, optional): special token dictionary. Defaults to None.

    Returns:
        obj: tokenizer object
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL)

    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
    return tokenizer

def get_model(tokenizer, special_tokens=None, load_model_path=None):
    """Get model from pretrained model

    Args:
        tokenizer (obj): tokenizer object
        special_tokens (dict, optional): special token dictionary. Defaults to None.
        load_model_path (str, optional): load model path. Defaults to None.
    
    Returns:
        obj: model object
    """
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
    """Remove special token from text

    Args:
        text (str): text to be processed

    Returns:
        _str: processed text 
    """
    for key, token in cfg.SPECIAL_TOKENS.items():
        text = text.replace(token, '')
    return text

def start_with_greet(text, greet):
    """Check if text starts with greet

    Args:
        text (str): text to be checked
        greet (list): list of greet tokens

    Returns:
        (bool, str): (True, greet token) if text starts with greet,
                    (False, None) otherwise
    """
    for g in greet:
        if text.startswith(g):
            return True, g
    return False, None

def post_processing(text, prompt):
    """Post processing generated email

    Args:
        text (str): text to be processed
        prompt (str): prompt to be replaced

    Returns:
        str: processed text
    """
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
