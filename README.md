## RCT abstract analyser

### Project abstract

For the past two decades, Randomised controlled trials (RCTs) have been widely
considered the gold standard for gathering unbiased evidence to support critical decisionmaking
in the biomedical industry. However, manually extracting and tabulating information
from RCT study publications is a time-consuming and costly process.
To address this problem, this study aims to utilise the latest state-of-the-art NLP
machine learning architecture - transformer-based language representations - to develop
an NLP pipeline that can automatically generate critical evidence tabulations
of given RCT paper abstracts. We utilised a combination of distantly-supervised and
fully-supervised learning approaches to collect the dataset for training and evaluating
our natural language processing modules. The evaluation results show high system
performance across all units within the pipeline, demonstrating that automation of the
evidence tabulation procedure for systematic reviews of RCT studies is feasible with
modern NLP technologies.

### Getting started

We recommend using Python 3.9 for installing all dependencies and running the application.

Install into a fresh virtual environment by:
```shell
pip install -r requirements.txt
```

Please note that if you wish to utilize the GPU for training and evaluation, you will need to install the `torch` package with CUDA following the tutorial on [PyTorch website](https://pytorch.org/get-started/locally/).

All models fine-tuned and used in this project are available in this [google drive link](https://drive.google.com/file/d/1WRF6XMOgxwcf2AM1V9imJWYItesQC2wr/view?usp=sharing), after downloading, please make sure the `models` folder is placed in the root directory of the project.

### Running the applications
Please invoke all following commands in the root directory of the project.

To launch the demo user interface, run by:
```shell
streamlit run .\src\interface.py
```

The web app will then be automatically opened in your default web browser.
\
However, if you wish to open it in your browser of choice, start by typing in the following URL and press enter:
```shell
http://localhost:8501/
```
\
To run the CLI tabulation script, run by:
```shell
python .\src\tabulate.py
```

An example input file to the CLI script `input.csv` is provided in the `input` folder.
\
By default, output of the CLI script will be named as `output.csv` and saved in the `output` folder.


### Model training and evaluation
To run the classification dataset collection and annotation script, run by:
```shell
python .\src\classifier_dataset_collection.py
```
A stable internet connection is needed for this script to run.
A usual run of this script will take around 5-10 minutes to complete due to the large amount of requests sent to PubMed.
Collected datasets for training participant and section classifier models will be saved in the `datasets` folder, under `participant classifier datasets` and `section classifier datasets` respectively.

To train the NER model, this project utilised a manually annotated dataset introduced by study [RCT-ART](https://arxiv.org/abs/2112.05596).
The dataset is available in the corresponding [github repository](https://github.com/jetsunwhitton/rct-art). We have provided a copy of the dataset in the `datasets` folder, under `NER datasets` with the filename `RCT_ART_NER.jsonl`.
\
To prepare the NER dataset for training, run by:
```shell
python .\src\preprocess_ner_dataset.py
```
The script will generate two files `train_NER_dataset.json` and `test_NER_dataset.json` under `NER datasets` for training and evaluation respectively.

To train the participant classifier model, run by:
```shell
python .\src\train_participant_classifier.py
```

To train the section classifier model, run by:
```shell
python .\src\train_section_classifier.py
```

To train the NER model, run by:
```shell
python .\src\train_ner.py
```
Corresponding trained models will be saved under `models` folder.

To evaluate the participant classifier model, run by:
```shell
python .\src\evaluate_participant_classifier.py
```

To evaluate the section classifier model, run by:
```shell
python .\src\evaluate_section_classifier.py
```

To evaluate the NER model, run by:
```shell
python .\src\evaluate_ner.py
```