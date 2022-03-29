# Email Generator
Email generator is a [gpt-2]() based email generator application. It is a fine-tune version of gpt-2-medium model which is trained on [this]() datasets.

## Installation
```bash
pip install -r requirements.txt
```
Install pytorch by

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

## Run Application
This is application is base one [Flask]() framework.
Default port for this application is 5000.

```bash
python app.py
```

## Test Application
Change input `data` in `tests/test_app.py` to test different inputs.

```bash
python -m tests.test_app
```