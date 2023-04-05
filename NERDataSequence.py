# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset


def tokenize_and_align_labels(text, tokenizer, label_ids=None):
    tokenized_text = []
    aligned_label_ids = []
    if label_ids is None:
        for i, word in enumerate(text):
            tokenized_word = tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)
            tokenized_text.extend(tokenized_word)
            if n_subwords > 0:
                aligned_label_ids.extend([i] * n_subwords)
    else:
        for word, label_id in zip(text, label_ids):
            tokenized_word = tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)
            tokenized_text.extend(tokenized_word)
            if n_subwords > 0:
                aligned_label_ids.extend([label_id] + [-100] * (n_subwords - 1))
    tokenized_text = ["[CLS]"] + tokenized_text + ["[SEP]"]
    aligned_label_ids = [-100] + aligned_label_ids + [-100]
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
    return input_ids, aligned_label_ids


def pad_token_label(token_ids, label_ids, max_length):
    token_ids = token_ids + [0] * (max_length - len(token_ids))
    label_ids = label_ids + [-100] * (max_length - len(label_ids))
    return token_ids, label_ids


class NERDataSequence(Dataset):
    def __init__(self, text_list, tokenizer, label_ids_list=None):
        self.input_ids, self.attention_masks, self.aligned_label_ids = [], [], []
        input_id_list, aligned_label_ids_list = [], []
        max_length = 0
        if label_ids_list is not None:
            for text, label_ids in zip(text_list, label_ids_list):
                input_ids, aligned_label_ids = tokenize_and_align_labels(text, tokenizer, label_ids)
                input_id_list.append(input_ids)
                aligned_label_ids_list.append(aligned_label_ids)
                if len(input_ids) > max_length:
                    max_length = len(input_ids)
        else:
            for text in text_list:
                input_ids, aligned_label_ids = tokenize_and_align_labels(text, tokenizer)
                input_id_list.append(input_ids)
                aligned_label_ids_list.append(aligned_label_ids)
                if len(input_ids) > max_length:
                    max_length = len(input_ids)

        for input_ids, aligned_label_ids in zip(input_id_list, aligned_label_ids_list):
            input_ids, aligned_label_ids = pad_token_label(input_ids, aligned_label_ids, max_length)
            self.input_ids.append(torch.tensor(input_ids))
            self.attention_masks.append(torch.tensor([1 if token_id != 0 else 0 for token_id in input_ids]))
            self.aligned_label_ids.append(torch.tensor(aligned_label_ids))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.aligned_label_ids[idx]
