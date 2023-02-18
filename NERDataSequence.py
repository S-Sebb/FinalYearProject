# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset


def tokenize_and_align_labels(text, labels, tokenizer, label_to_ids):
    tokenized_text = []
    aligned_labels = []
    for word, label in zip(text, labels):
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        tokenized_text.extend(tokenized_word)
        if n_subwords > 0:
            aligned_labels.extend([label] + ["X"] * (n_subwords - 1))
    aligned_label_ids = [label_to_ids[label] for label in aligned_labels]
    tokenized_text = ["[CLS]"] + tokenized_text + ["[SEP]"]
    aligned_label_ids = [-100] + aligned_label_ids + [-100]
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
    return input_ids, aligned_label_ids


def pad_token_label(token_ids, label_ids, max_length):
    token_ids = token_ids + [0] * (max_length - len(token_ids))
    label_ids = label_ids + [-100] * (max_length - len(label_ids))
    return token_ids, label_ids


class NERDataSequence(Dataset):
    def __init__(self, text_list, labels_list, tokenizer, label_to_ids):
        self.input_ids, self.attention_masks, self.aligned_label_ids = [], [], []
        input_id_list, aligned_label_id_list = [], []
        max_length = 0
        for text, labels in zip(text_list, labels_list):
            input_ids, aligned_label_ids = tokenize_and_align_labels(text, labels, tokenizer, label_to_ids)
            input_id_list.append(input_ids)
            aligned_label_id_list.append(aligned_label_ids)
            if len(input_ids) > max_length:
                max_length = len(input_ids)

        for input_ids, aligned_label_ids in zip(input_id_list, aligned_label_id_list):
            input_ids, aligned_label_ids = pad_token_label(input_ids, aligned_label_ids, max_length)
            self.input_ids.append(torch.tensor(input_ids))
            self.attention_masks.append(torch.tensor([1 if token_id != 0 else 0 for token_id in input_ids]))
            self.aligned_label_ids.append(torch.tensor(aligned_label_ids))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.aligned_label_ids[idx]
