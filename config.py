SEED = 2020 # random seed value
MODEL = "sagorsarker/emailgenerator" # hf model name
MAXLEN = 768
# CUSTOM_MODEL_PATH = "./model/pytorch_model.bin"
SPECIAL_TOKENS  = { "bos_token": "<|BOS|>",
                    "eos_token": "<|EOS|>",
                    "unk_token": "<|UNK|>",                    
                    "pad_token": "<|PAD|>",
                    "sep_token": "<|SEP|>"}

GREET_TOKENS = ["hi", "hey", "hello", "dear"]
START_GREET = "hi [name]\n"
CONCLUSION_GREET = "\nsincerely\n[name]"
