# -*- coding: utf-8 -*-
import os

import spacy  # https://spacy.io/
from spacy.tokens import Span  # https://spacy.io/api/span
from transformers import BertForSequenceClassification, AutoModelForSequenceClassification, \
    AutoModelForTokenClassification, BertForTokenClassification, RobertaForTokenClassification, BertTokenizer, \
    RobertaTokenizer  # https://huggingface.co/docs/transformers/index


def load_pretrained_classification_tokenizer_model(num_labels):
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-cased',
        do_lower_case=True
    )
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-cased',
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False
    )
    return tokenizer, model


def load_pretrained_NER_tokenizer_model(model_type, num_labels):
    if model_type == "BERT":
        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-cased',
            do_lower_case=False,
        )
        model = BertForTokenClassification.from_pretrained(
            'bert-base-cased',
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False
        )
    elif model_type == "RoBERTa":  # RoBERTa has a different model structure and tokenizer than BERT and BioBERT
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
    elif model_type == "BioBERT":  # Uses the same model structure and tokenizer as BERT
        tokenizer = BertTokenizer.from_pretrained(
            'dmis-lab/biobert-base-cased-v1.1',
            do_lower_case=False
        )
        model = BertForTokenClassification.from_pretrained(
            'bert-base-cased',
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
    if model_type == "BERT" or model_type == "BioBERT":
        tokenizer = BertTokenizer.from_pretrained(tokenizer_filepath, local_files_only=True)
    elif model_type == "RoBERTa":
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_filepath, local_files_only=True)
    else:
        raise ValueError("model_type must be either 'BERT', 'RoBERTa' or 'BioBERT'.")
    return model, tokenizer


def init_spacy_parser():
    parser = spacy.load("en_core_web_sm", disable=["tagger", "ner", "entity"])  # just the parser
    return parser


def abstract_to_paragraphs(abstract):
    paragraphs = abstract.split("\n\n")
    return paragraphs


def paragraph_to_sentences(paragraph):
    paragraph = paragraph.strip() + " "
    sentences = paragraph.split(". ")
    # Add the period back to the end of each sentence
    sentences = [sentence + "." for sentence in sentences if sentence.strip() != ""]
    return sentences


def make_section_dict(paragraphs, labels):
    section_dict = {}
    for paragraph, label in zip(paragraphs, labels):
        if label not in section_dict:
            section_dict[label] = paragraph
        else:
            section_dict[label] += "\n\n" + paragraph
    return section_dict


def align_spacy_doc_entity(doc, labels):
    entities = []
    cur_label = None
    label_start = None
    for i, label in enumerate(labels):
        if label == "O":
            if cur_label is not None:
                entity = Span(doc, label_start, i, label=cur_label)
                entities.append(entity)
            cur_label = None
            continue
        if cur_label is None:
            cur_label = label
            label_start = i
        elif cur_label != label:
            entity = Span(doc, label_start, i, label=cur_label)
            entities.append(entity)
            cur_label = label
            label_start = i
    doc.set_ents(entities)

    return doc


def try_load_predefined_models_tokenizers():
    section_classification_model_dir = os.path.join("models", "section classifier models")
    section_classification_model_filename = "BERT_5Section_CustomWeightsTrue_classifier_model"
    section_classification_tokenizer_filename = "BERT_section_classifier_tokenizer"

    participant_classification_model_dir = os.path.join("models", "participant classifier models")
    participant_classification_model_filename = "BERT_CustomWeightsTrue_participant_classifier_model"
    participant_classification_tokenizer_filename = "BERT_participant_classifier_tokenizer"

    ner_model_dir = os.path.join("models", "NER models")
    ner_model_filename = "RoBERTa_CustomWeightTrue_NER_model"
    ner_tokenizer_filename = "RoBERTa_NER_tokenizer"
    ner_model_type = "RoBERTa"

    section_classification_model_filepath = os.path.join(section_classification_model_dir,
                                                         section_classification_model_filename)
    section_classification_tokenizer_filepath = os.path.join(section_classification_model_dir,
                                                             section_classification_tokenizer_filename)
    section_classification_model, section_classification_tokenizer = load_fine_tuned_classification_model_tokenizer(
        section_classification_model_filepath, section_classification_tokenizer_filepath)

    participant_classification_model_filepath = os.path.join(participant_classification_model_dir,
                                                             participant_classification_model_filename)
    participant_classification_tokenizer_filepath = os.path.join(participant_classification_model_dir,
                                                                 participant_classification_tokenizer_filename)
    participant_classification_model, participant_classification_tokenizer = load_fine_tuned_classification_model_tokenizer(
        participant_classification_model_filepath, participant_classification_tokenizer_filepath)

    ner_model_filepath = os.path.join(ner_model_dir, ner_model_filename)
    ner_tokenizer_filepath = os.path.join(ner_model_dir, ner_tokenizer_filename)
    ner_model, ner_tokenizer = load_fine_tuned_NER_model_tokenizer(ner_model_filepath, ner_tokenizer_filepath,
                                                                   ner_model_type)
    return section_classification_model, section_classification_tokenizer, participant_classification_model, \
        participant_classification_tokenizer, ner_model, ner_tokenizer
