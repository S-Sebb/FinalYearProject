# -*- coding: utf-8 -*-

import spacy
import torch
from spacy.tokens import Span, Doc
from transformers import BertTokenizer


def init_ner_model_tokenizer(ner_model_path, ner_tokenizer_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(ner_model_path).to(device)
    tokenizer = BertTokenizer.from_pretrained(ner_tokenizer_path)
    return model, tokenizer


def init_spacy_parser():
    parser = spacy.load("en_core_web_sm", disable=["tagger", "ner", "entity"])  # just the parser
    return parser


def make_spacy_doc(text, parser):
    doc = parser.make_doc(text)
    return doc


def text_ner(words, ner_model, tokenizer, ids_to_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokens = []
    word_ids = []
    for i, word in enumerate(words):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        word_ids.extend([i] * len(token))
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([input_ids]).to(device)
    with torch.no_grad():
        output = ner_model(input_ids)
    label_ids = torch.argmax(output[0], dim=2)
    labels = []
    for i, label_id in enumerate(label_ids.squeeze()):
        label_id = label_id.item()
        if word_ids[i] >= len(labels):
            if label_id not in ids_to_labels:
                labels.append("O")
            else:
                labels.append(ids_to_labels[label_id])

    return labels


def create_spacy_entities(doc, labels):
    ents = []
    cur_label = None
    label_start = None
    for i, label in enumerate(labels):
        if label == "O":
            if cur_label is not None:
                ent = Span(doc, label_start, i, label=cur_label)
                ents.append(ent)
            cur_label = None
            continue
        if cur_label is None:
            cur_label = label
            label_start = i
        elif cur_label != label:
            ent = Span(doc, label_start, i, label=cur_label)
            ents.append(ent)
            cur_label = label
            label_start = i
    doc.set_ents(ents)

    return doc


def merge_spacy_docs(docs):
    merged_doc = Doc.from_docs(docs)
    return merged_doc


if __name__ == '__main__':
    input_text = """There are limited data from randomized trials regarding whether volume-based, low-dose computed tomographic (CT) screening can reduce lung-cancer mortality among male former and current smokers.

    A total of 13,195 men (primary analysis) and 2594 women (subgroup analyses) between the ages of 50 and 74 were randomly assigned to undergo CT screening at T0 (baseline), year 1, year 3, and year 5.5 or no screening. We obtained data on cancer diagnosis and the date and cause of death through linkages with national registries in the Netherlands and Belgium, and a review committee confirmed lung cancer as the cause of death when possible. A minimum follow-up of 10 years until December 31, 2015, was completed for all participants.

    Among men, the average adherence to CT screening was 90.0%. On average, 9.2% of the screened participants underwent at least one additional CT scan (initially indeterminate). The overall referral rate for suspicious nodules was 2.1%. At 10 years of follow-up, the incidence of lung cancer was 5.58 cases per 1000 person-years in the screening group and 4.91 cases per 1000 person-years in the control group; lung-cancer mortality was 2.50 deaths per 1000 person-years and 3.30 deaths per 1000 person-years, respectively. The cumulative rate ratio for death from lung cancer at 10 years was 0.76 (95% confidence interval [CI], 0.61 to 0.94; P = 0.01) in the screening group as compared with the control group, similar to the values at years 8 and 9. Among women, the rate ratio was 0.67 (95% CI, 0.38 to 1.14) at 10 years of follow-up, with values of 0.41 to 0.52 in years 7 through 9.

    In this trial involving high-risk persons, lung-cancer mortality was significantly lower among those who underwent volume CT screening than among those who underwent no screening. There were low rates of follow-up procedures for results suggestive of lung cancer. (Funded by the Netherlands Organization of Health Research and Development and others; NELSON Netherlands Trial Register number, NL580.).
    """

    ids_to_label = {0: 'INTV', 1: 'MEAS', 2: 'O', 3: 'OC'}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = init_ner_model_tokenizer()
    model.eval()
    model.to(device)

    tokenized_text = tokenizer(
        input_text,
        add_special_tokens=True,
        return_tensors='pt'
    )
    input_ids = tokenized_text['input_ids'].to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    attention_mask = tokenized_text['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    logits = outputs.logits
    logits = logits.squeeze().detach().cpu().numpy()
    label_ids = logits.argmax(axis=1)
    labels = [ids_to_label[label_id] for label_id in label_ids]
    for token, label in zip(tokens, labels):
        print(token, label)
