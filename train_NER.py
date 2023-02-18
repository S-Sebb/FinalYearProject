# -*- coding: utf-8 -*-
import json
import os

import numpy as np
import torch
from livelossplot import PlotLosses
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from transformers import BertTokenizer, BertForTokenClassification, logging

from NERDataSequence import NERDataSequence

if __name__ == "__main__":
    input_filepath = os.path.join("annotated_NER", "RCT_ART_NER.jsonl")
    model_output_filepath = os.path.join("NER.pt")
    tokenizer_output_filepath = os.path.join("NER_tokenizer.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_list = []
    labels_list = []
    label_counts = {}

    with open(input_filepath, "r", encoding="utf-8") as f:
        for line in f:
            json_data = json.loads(line)
            text = json_data["text"]
            tokens = json_data["tokens"]
            spans = json_data["spans"]

            tokenized_text = [token["text"] for token in tokens]
            labels = ["O"] * len(tokenized_text)

            for span in spans:
                token_start = span["token_start"]
                token_end = span["token_end"]
                label = span["label"]
                for i in range(token_start, token_end + 1):
                    labels[i] = label

            for label in labels:
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1
            text_list.append(tokenized_text)
            labels_list.append(labels)

    unique_labels = []

    for spans in labels_list:
        for label in spans:
            if label not in unique_labels:
                unique_labels.append(label)

    unique_labels = sorted(unique_labels)

    label_to_ids = {label: i for i, label in enumerate(unique_labels)}
    label_to_ids["X"] = -100
    ids_to_label = {i: label for i, label in enumerate(unique_labels)}
    ids_to_label[-100] = "X"

    # Suppress warnings from training a pre-trained model
    logging.set_verbosity_error()

    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-cased',
        do_lower_case=False
    )

    dataset = NERDataSequence(text_list, labels_list, tokenizer, label_to_ids)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8),
                                                                         len(dataset) - int(len(dataset) * 0.8)])
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)

    model = BertForTokenClassification.from_pretrained(
        'bert-base-cased',
        num_labels=len(unique_labels),
        output_attentions=False,
        output_hidden_states=False
    )
    model.to(device)

    epochs = 5
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
            train_accurate_num = 0
            train_total_num = 0

            for batch in train_dataloader:
                pbar_2.update(1)
                input_ids, attention_mask, aligned_label_ids = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                aligned_label_ids = aligned_label_ids.to(device)
                for labels in aligned_label_ids.squeeze():
                    for label in labels:
                        if label != -100:
                            train_total_num += 1
                model.train()
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=None,
                                labels=aligned_label_ids)
                loss = outputs[0]
                logits = outputs[1]
                logits = logits.detach().cpu().numpy()
                true_label_ids = aligned_label_ids.to('cpu').numpy()
                for i in range(len(logits)):
                    for j in range(len(logits[i])):
                        if logits[i][j].argmax() == true_label_ids[i][j] and true_label_ids[i][j] != -100:
                            train_accurate_num += 1
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            test_accurate_num = 0
            test_total_num = 0

            for batch in val_dataloader:
                input_ids, attention_mask, aligned_label_ids = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                aligned_label_ids = aligned_label_ids.to(device)
                for labels in aligned_label_ids.squeeze():
                    for label in labels:
                        if label != -100:
                            test_total_num += 1

                model.eval()
                outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=None,
                                labels=aligned_label_ids)
                logits = outputs[1]
                logits = logits.detach().cpu().numpy()
                true_label_ids = aligned_label_ids.to('cpu').numpy()
                pred_label_ids = np.argmax(logits, axis=2)
                for i in range(len(logits)):
                    for j in range(len(logits[i])):
                        if pred_label_ids[i][j] == true_label_ids[i][j] and true_label_ids[i][j] != -100:
                            test_accurate_num += 1

        pbar.set_postfix({"loss": total_loss / len(train_dataloader),
                          "train_accuracy": (train_accurate_num / train_total_num),
                          "test_accuracy": (test_accurate_num / test_total_num)})
        plt.update({"loss": total_loss / len(train_dataloader),
                    "train_accuracy": (train_accurate_num / train_total_num),
                    "test_accuracy": (test_accurate_num / test_total_num)})

    torch.save(model, model_output_filepath)
    tokenizer.save_pretrained(tokenizer_output_filepath)
    plt.send()
