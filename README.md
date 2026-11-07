# Spam Detection in Messages

Web application and data-science study for automatic detection of spam
messages. The project was developed by a group of four students for the
course *Ciberataques y Ciencia de Datos para Defensa*.

The repository contains two complementary deliverables:

- A **Flask web application** that classifies any text typed by the user
  with a pretrained Hugging Face model
  ([`Goodmotion/spam-mail-classifier`](https://huggingface.co/Goodmotion/spam-mail-classifier)).
- A **Jupyter notebook** that documents, step by step, the complete data
  science methodology applied by the team: exploratory data analysis,
  preprocessing, supervised modeling, and evaluation of the pretrained
  model on the same held-out partition.

## Team members

| Name | Role |
|---|---|
| Fredy Alvarez | Member |
| Sebastian Orozco | Member |
| Cristian Yepes | Member |
| Yenny Villarreal | Member |

All four members participated in the data study, the model evaluation
and the development of the web application.

## Project structure

```
spam-detection/
├── app/
│   ├── app.py              # Flask application that serves the model
│   └── templates/
│       └── index.html      # Web interface (form and result view)
├── data/
│   └── SMSSpamCollection   # SMS Spam Collection v.1 dataset
├── notebook/
│   └── spam_detection.ipynb# Step-by-step data science process
├── requirements.txt        # Python packages with pinned versions
└── README.md
```

## How the project works

The data source is the **SMS Spam Collection v.1** (5574 labeled SMS
messages: `ham` = legitimate, `spam` = unwanted). The pretrained model is
a multilingual MiniLM encoder fine-tuned by its authors for binary spam
classification. Its labels are `SPAM` (index 1) and `NOSPAM`
(index 0); the application normalizes them to the readable values
`SPAM / NO SPAM` and shows the model confidence for each prediction.

The notebook reproduces and documents the full process: class balance,
message length distributions, characteristic vocabulary per class,
TF-IDF + naive Bayes and logistic regression baselines, evaluation of
the pretrained model on the same test partition, confusion matrices and
a quantitative discussion of the domain mismatch between e-mail (the
model's training domain) and SMS.

## How to run the project

Requirements: Python 3.11 (the exact package versions are pinned in
`requirements.txt`).

```bash
# 1. Clone the repository and enter the folder
git clone <repository-url>
cd spam-detection

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # macOS / Linux

# 3. Install the dependencies
#    The +cpu build of torch is installed from PyTorch's CPU index,
#    which is much lighter than the CUDA build.
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 4. Run the web application
python app/app.py
```

Then open `http://127.0.0.1:5000` in a browser, type a message in the
form and press the classification button. The page returns the verdict
(SPAM / NO SPAM) together with the model confidence. Example messages
are provided inside the interface.

The first request may take a few seconds while the model weights are
downloaded from Hugging Face; afterwards the model is loaded once and
reused by every request.

## How to open the notebook

```bash
pip install jupyter
jupyter notebook notebook/spam_detection.ipynb
```

The notebook can be re-executed cell by cell, with results identical to
the ones stored in the file thanks to the fixed random seed. It also
works on machines without internet access after the first run, since
the model stays in the local Hugging Face cache.

## Dataset and model credits

- Dataset: Almeida, T. A., Gomez Hidalgo, J. M., Yamakami, A. (2011),
  *Contributions to the study of SMS Spam Filtering: New Collection and
  Results*, ACM DOCENG'11.
- Pretrained model: `Goodmotion/spam-mail-classifier`, Hugging Face,
  Apache 2.0 license.
