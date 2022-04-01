# Email Writer
Email writer is a [gpt-2](https://github.com/openai/gpt-2) based email generator application. It is a fine-tune version of [gpt-2-medium](https://huggingface.co/gpt2-medium) model which is trained on [this](https://www.kaggle.com/datasets/mikeschmidtavemac/emailblog) datasets.

## API request and response schema
```py
request = {
    "prompt": str,
    "token_count": int,
    "temperature": float,
    "n_gen": int,
    "keywords": list
}

response = {
    "status": str,
    "ai_results": [
        {
            "generated_text": str,
            "text_length": int
        },
    ]
}
```

## Installation
### Method-1
environment setup using `environment.yml` file
```
conda env create -f environment.yml
conda activate email_writer_env
```
### Method-2
```bash
pip install -r requirements.txt
# install pytorch >= 1.11
conda install pytorch cudatoolkit=11.3 -c pytorch
```

## Run Application
This is application is based one [Flask](https://flask.palletsprojects.com/en/2.1.x/) framework.
Default port for this application is 5000.

```bash
python app.py
```

## Test Application
Change input `data` in `tests/test_app.py` to test different inputs.

```bash
python -m tests.test_app
```
or 

```py
import requests

data = {
    "prompt": "love to visit again", # subject of the email
    "token_count": 128,
    "temperature": 0.6,
    "n_gen": 4,
    "keywords": ['experience', 'joyfull', 'sea']
}

response = requests.post('http://localhost:5000/generate', json=data)
print(response.json())

```