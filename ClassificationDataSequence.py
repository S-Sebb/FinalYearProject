# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset


def preprocess_text(text, tokenizer):
    processed_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        padding="max_length",
        max_length=256,
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt'
    )
    return processed_text


class ClassificationDataSequence(Dataset):
    def __init__(self, lines, tokenizer, label_ids=None):
        self.input_ids, self.attention_masks, self.label_ids = [], [], []
        for line in lines:
            processed_text_dict = preprocess_text(line, tokenizer)
            self.input_ids.append(processed_text_dict['input_ids'])
            self.attention_masks.append(processed_text_dict['attention_mask'])
        self.input_ids = torch.cat(self.input_ids, dim=0)
        self.attention_masks = torch.cat(self.attention_masks, dim=0)
        if label_ids is not None:
            self.label_ids = torch.tensor(label_ids)
        else:
            self.label_ids = torch.tensor([-1] * len(lines))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.label_ids[idx]
