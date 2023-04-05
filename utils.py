# -*- coding: utf-8 -*-
from transformers import BertForSequenceClassification, AutoModelForSequenceClassification, \
    AutoModelForTokenClassification, BertForTokenClassification, RobertaForTokenClassification, BertTokenizer, \
    RobertaTokenizer


def load_pretrained_classifier_tokenizer_model(num_labels):
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased',
        do_lower_case=True
    )
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False
    )
    return tokenizer, model


def load_pretrained_NER_tokenizer_model(model_type, num_labels):
    if model_type == "BERT":
        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-cased',
            do_lower_case=False
        )
        model = BertForTokenClassification.from_pretrained(
            'bert-base-cased',
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False
        )
    elif model_type == "RoBERTa":  # RoBERTa has a different model structure than BERT and BioBERT
        tokenizer = RobertaTokenizer.from_pretrained(
            'roberta-base',
            do_lower_case=False
        )
        model = RobertaForTokenClassification.from_pretrained(
            'roberta-base',
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False
        )
    elif model_type == "BioBERT":  # Uses the same model structure as BERT
        tokenizer = BertTokenizer.from_pretrained(
            'dmis-lab/biobert-base-cased-v1.1',
            do_lower_case=False
        )
        model = BertForTokenClassification.from_pretrained(
            'dmis-lab/biobert-base-cased-v1.1',
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False
        )
    else:
        raise ValueError("model_type must be either 'BERT', 'RoBERTa' or 'BioBERT'.")
    return tokenizer, model


def load_fine_tuned_classification_model_tokenizer(model_filepath, tokenizer_filepath):
    model = AutoModelForSequenceClassification.from_pretrained(model_filepath, local_files_only=True)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_filepath, local_files_only=True)
    return model, tokenizer


def load_fine_tuned_NER_model_tokenizer(model_filepath, tokenizer_filepath, model_type):
    model = AutoModelForTokenClassification.from_pretrained(model_filepath, local_files_only=True)
    if model_type == "BERT" or "BioBERT":
        tokenizer = BertTokenizer.from_pretrained(tokenizer_filepath, local_files_only=True)
    elif model_type == "RoBERTa":
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_filepath, local_files_only=True)
    else:
        raise ValueError("model_type must be either 'BERT', 'RoBERTa' or 'BioBERT'.")
    return model, tokenizer
