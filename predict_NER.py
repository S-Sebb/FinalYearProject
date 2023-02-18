# -*- coding: utf-8 -*-

import spacy
import torch
from transformers import BertTokenizer
import numpy as np
from spacy.tokens import Span

def init_ner_model_tokenizer(ner_model_path, ner_tokenizer_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(ner_model_path).to(device)
    tokenizer = BertTokenizer.from_pretrained(ner_tokenizer_path)
    return model, tokenizer


def init_spacy_parser():
    parser = spacy.load("en_core_web_sm", disable=["tagger", "ner"])  # just the parser
    return parser


def text_ner(text, model, tokenizer, parser, ids_to_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    doc = parser.make_doc(text)
    words = [token.text for token in doc]
    tokens = []
    word_ids = []
    for i, word in enumerate(words):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        word_ids.extend([i] * len(token))
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([input_ids]).to(device)
    with torch.no_grad():
        output = model(input_ids)
    label_ids = torch.argmax(output[0], dim=2)
    labels = []
    for i, label_id in enumerate(label_ids.squeeze()):
        label_id = label_id.item()
        if word_ids[i] >= len(labels):
            labels.append(ids_to_labels[label_id])
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
            ent = Span(doc, i, i + 1, label=label)
            ents.append(ent)
    doc.set_ents(ents)

    return doc


if __name__ == '__main__':
    input_text = """Objectives: To compare health-related quality of life (HRQoL) in newly diagnosed, treatment-naive patients with OAG or OHT, treated with two treatment pathways: topical IOP-lowering medication from the outset (Medicine-1st) or primary SLT followed by topical medications as required (Laser-1st). We also compared the clinical effectiveness and cost-effectiveness of the two pathways.

    Design: A 36-month pragmatic, unmasked, multicentre randomised controlled trial.
    
    Settings: Six collaborating specialist glaucoma clinics across the UK.
    
    Participants: Newly diagnosed patients with OAG or OHT in one or both eyes who were aged ≥ 18 years and able to provide informed consent and read and understand English. Patients needed to qualify for treatment, be able to perform a reliable visual field (VF) test and have visual acuity of at least 6 out of 36 in the study eye. Patients with VF loss mean deviation worse than -12 dB in the better eye or -15 dB in the worse eye were excluded. Patients were also excluded if they had congenital, early childhood or secondary glaucoma or ocular comorbidities; if they had any previous ocular surgery except phacoemulsification, at least 1 year prior to recruitment or any active treatment for ophthalmic conditions; if they were pregnant; or if they were unable to use topical medical therapy or had contraindications to SLT.
    
    Interventions: SLT according to a predefined protocol compared with IOP-lowering eyedrops, as per national guidelines.
    
    Main outcome measures: The primary outcome was HRQoL at 3 years [as measured using the EuroQol-5 Dimensions, five-level version (EQ-5D-5L) questionnaire]. Secondary outcomes were cost and cost-effectiveness, disease-specific HRQoL, clinical effectiveness and safety.
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
