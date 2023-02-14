# -*- coding: utf-8 -*-
import json
import os

import torch
from livelossplot import PlotLosses
from torch.utils.data import Dataset, DataLoader
from tqdm import trange, tqdm
from transformers import BertTokenizer, BertForTokenClassification, logging

from jsonl_to_conll import jsonl_to_IOB


def tokenize_and_align_labels(text, labels, tokenizer):
    tokenized_text = []
    aligned_labels = []
    for word, label in zip(text, labels):
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        tokenized_text.extend(tokenized_word)
        if n_subwords > 0:
            aligned_labels.extend([label] + ["X"] * (n_subwords - 1))
    aligned_label_ids = [label_to_ids[label] for label in aligned_labels]
    if len(tokenized_text) > max_length:
        tokenized_text = tokenized_text[:max_length]
        aligned_label_ids = aligned_label_ids[:max_length]
    tokenized_text = ["[CLS]"] + tokenized_text + ["[SEP]"]
    aligned_label_ids = [-100] + aligned_label_ids + [-100]
    if len(tokenized_text) < max_length + 2:
        tokenized_text = tokenized_text + ["[PAD]"] * (max_length + 2 - len(tokenized_text))
        aligned_label_ids = aligned_label_ids + [-100] * (max_length + 2 - len(aligned_label_ids))
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
    attention_mask = [1 if input_id != 0 else 0 for input_id in input_ids]
    return input_ids, attention_mask, aligned_label_ids


class DataSequence(Dataset):
    def __init__(self, text_list, labels_list, tokenizer):
        self.input_id_list, self.attention_mask_list, self.aligned_label_id_list = [], [], []
        for text, labels in zip(text_list, labels_list):
            input_ids, attention_mask, aligned_label_ids = tokenize_and_align_labels(text, labels, tokenizer)
            self.input_id_list.append(torch.tensor(input_ids))
            self.attention_mask_list.append(torch.tensor(attention_mask))
            self.aligned_label_id_list.append(torch.tensor(aligned_label_ids))

    def __len__(self):
        return len(self.input_id_list)

    def __getitem__(self, idx):
        return self.input_id_list[idx], self.attention_mask_list[idx], self.aligned_label_id_list[idx]


if __name__ == "__main__":
    input_filepath = os.path.join("annotated_NER", "RCT_ART_NER.jsonl")
    output_filepath = os.path.join("NER.pt")
    max_length = 300
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_list = []
    labels_list = []
    label_counts = {}

    with open(input_filepath, "r", encoding="utf-8") as f:
        for line in f:
            json_data = json.loads(line)
            text = json_data["text"]
            labels = json_data["spans"]
            tokenized_text, iob_tags = jsonl_to_IOB(text, labels)
            for label in iob_tags:
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1
            text_list.append(tokenized_text)
            labels_list.append(iob_tags)

    unique_labels = []

    for labels in labels_list:
        for label in labels:
            if label not in unique_labels:
                unique_labels.append(label)

    unique_labels = sorted(unique_labels)
    print(unique_labels)
    label_to_ids = {label: i for i, label in enumerate(unique_labels)}
    label_to_ids["X"] = -100
    ids_to_label = {i: label for i, label in enumerate(unique_labels)}

    # Suppress warnings from training a pre-trained model
    logging.set_verbosity_error()

    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-cased',
        do_lower_case=False
    )

    dataset = DataSequence(text_list, labels_list, tokenizer)
    train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = BertForTokenClassification.from_pretrained(
        'bert-base-cased',
        num_labels=len(unique_labels),
        output_attentions=False,
        output_hidden_states=False
    )
    model.to(device)

    epochs = 200
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, eps=1e-08, weight_decay=1e-8)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    model.train()

    plt = PlotLosses()

    pbar = trange(epochs, desc="Epoch", leave=False)
    pbar_2 = trange(len(train_dataloader), desc="Batch", leave=False)

    for epoch in range(epochs):
        pbar.update(1)
        total_loss = 0
        with tqdm(total=len(train_dataloader), desc="Training batches", leave=False) as pbar_2:
            for batch in train_dataloader:
                pbar_2.update(1)
                input_ids, attention_mask, aligned_label_ids = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                aligned_label_ids = aligned_label_ids.to(device)
                model.train()
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=None,
                                labels=aligned_label_ids)
                logits = outputs[1]
                loss = loss_fn(logits.view(-1, len(unique_labels)), aligned_label_ids.view(-1))
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
        pbar.set_postfix({"loss": total_loss / len(train_dataloader)})
        plt.update({"loss": total_loss / len(train_dataloader)})

    torch.save(model, output_filepath)
    plt.send()
