# -*- coding: utf-8 -*-
import torch

from ClassificationDataSequence import ClassificationDataSequence
from NERDataSequence import NERDataSequence


def predict_classification(lines, ids_to_labels, model, tokenizer):
    dataset = ClassificationDataSequence(lines, tokenizer)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    pred_label_ids = []
    for batch in dataset:
        input_ids, attention_masks, _ = batch
        input_ids = input_ids.to(device).unsqueeze(0)
        attention_masks = attention_masks.to(device).unsqueeze(0)
        model_output = model(input_ids, attention_mask=attention_masks)
        pred_label_ids.append(torch.argmax(model_output[0].squeeze()).item())
    pred_labels = [ids_to_labels[pred_label_id] for pred_label_id in pred_label_ids]
    return pred_labels


def predict_NER(texts_list, model, tokenizer):
    dataset = NERDataSequence(texts_list, tokenizer)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    pred_label_ids_list = []
    for batch in dataset:
        input_ids, attention_masks, word_ids = batch
        input_ids = input_ids.to(device).unsqueeze(0)
        attention_masks = attention_masks.to(device).unsqueeze(0)
        model_output = model(input_ids, attention_mask=attention_masks)
        pred_unaligned_label_ids = torch.argmax(model_output[0].squeeze(), dim=1)
        last_word_id = -1
        pred_label_ids = []
        for i, word_id in enumerate(word_ids):
            if word_id != last_word_id and word_id != -100:
                pred_label_ids.append(pred_unaligned_label_ids[i].item())
                last_word_id = word_id
        pred_label_ids_list.append(pred_label_ids)
    return pred_label_ids_list
