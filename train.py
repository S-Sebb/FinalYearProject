# -*- coding: utf-8 -*-
import json
import os
import time

import livelossplot
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from transformers import logging

from ClassificationDataSequence import ClassificationDataSequence
from NERDataSequence import NERDataSequence
from utils import *


def train_classifier(output_dir_filepath, output_model_filename, output_tokenizer_filename, classifier_dataset_filepath,
                     labels_to_ids, epochs, learning_rate, weight_decay, custom_weights_flag):
    output_model_filepath = os.path.join(output_dir_filepath, output_model_filename)
    output_tokenizer_filepath = os.path.join(output_dir_filepath, output_tokenizer_filename)
    if not os.path.exists(output_dir_filepath):
        os.makedirs(output_dir_filepath)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv(classifier_dataset_filepath, index_col=0)
    lines = df.line.values
    label_texts = df.label.values
    label_ids = [labels_to_ids[label_text] for label_text in label_texts]

    unique_label_ids = np.unique(label_ids)
    num_labels = len(unique_label_ids)
    label_counts_dict = {}
    for label in label_ids:
        if label not in label_counts_dict:
            label_counts_dict[label] = 0
        label_counts_dict[label] += 1

    # Calculate the class weights for the loss function for balancing the dataset
    total_label_counts = len(label_ids)
    class_weights = []
    for label, count in label_counts_dict.items():
        class_weights.append(1 - label_counts_dict[label] / total_label_counts)

    # Suppress warnings from training a pre-trained model
    logging.set_verbosity_error()

    tokenizer, model = load_pretrained_classifier_tokenizer_model(num_labels)
    model.to(device)

    dataset = ClassificationDataSequence(lines, tokenizer, label_ids)
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42, stratify=label_ids)

    train_dataloader = DataLoader(train_dataset, batch_size=8)
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if custom_weights_flag:
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    pbar_1 = trange(epochs, desc='Epoch', leave=False)

    plt = livelossplot.PlotLosses()

    best_val_acc = 0
    val_plateau_counter = 0

    # Start timing
    start_time = time.time()

    for epoch in pbar_1:
        # Tracking variables
        with tqdm(total=len(train_dataloader), desc="Training batches", leave=False) as pbar_2:
            model.train()
            train_loss_sum = 0
            for batch in train_dataloader:
                pbar_2.update(1)
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, gt_labels = batch
                optimizer.zero_grad()
                # Forward pass
                train_output = model(input_ids, attention_mask=input_mask)
                # Backward pass
                train_loss = loss_fn(train_output.logits.squeeze(), gt_labels)
                train_loss.backward()
                optimizer.step()
                # Update tracking variables
                train_loss_sum += train_loss.item()
            pbar_2.close()

        with tqdm(total=len(val_dataloader), desc="Validation batches", leave=False) as pbar_3:
            model.eval()
            val_label_count = 0
            val_correct_count = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    pbar_3.update(1)
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, gt_labels = batch
                    # Forward pass
                    val_output = model(input_ids, attention_mask=input_mask)
                    val_labels = torch.argmax(val_output.logits, dim=1)
                    val_label_count += len(val_labels)
                    val_correct_count += torch.sum(val_labels == gt_labels).item()

        val_acc = val_correct_count / val_label_count
        if epoch >= 5:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                val_plateau_counter = 0
            else:
                val_plateau_counter += 1
                if val_plateau_counter >= 5:
                    print("\nEarly stopping at epoch " + str(epoch) + "!")
                    break
        pbar_1.set_postfix(
            {"train loss": train_loss_sum / len(train_dataloader), "val accuracy": val_acc,
             "best_val accuracy": best_val_acc})
        plt.update(
            {"train loss": train_loss_sum / len(train_dataloader), "val accuracy": val_acc,
             "best_val accuracy": best_val_acc})

    # End timing
    end_time = time.time()
    print("Training time: {:.3f} minutes".format((end_time - start_time) / 60))

    model.save_pretrained(output_model_filepath)
    tokenizer.save_pretrained(output_tokenizer_filepath)

    plt.send()


def train_NER(output_dir_filepath, output_model_filename, output_tokenizer_filename, NER_dataset_filepath,
                     labels_to_ids, epochs, learning_rate, weight_decay, model_type):
    # Start of the script
    output_model_filepath = os.path.join(output_dir_filepath, output_model_filename)
    output_tokenizer_filepath = os.path.join(output_dir_filepath, output_tokenizer_filename)
    if not os.path.exists(output_dir_filepath):
        os.makedirs(output_dir_filepath)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(NER_dataset_filepath, "r") as f:
        dataset = json.load(f)
    token_texts_list = dataset["token_texts_list"]
    labels_list = dataset["labels_list"]
    label_ids_list = [[labels_to_ids[label] for label in labels] for labels in labels_list]

    # Suppress warnings from training a pre-trained model
    logging.set_verbosity_error()

    tokenizer, model = load_pretrained_NER_tokenizer_model(model_type, len(labels_to_ids.keys()))
    tokenizer_filepath = os.path.join("models", "NER models", "NER_tokenizer")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_filepath, local_files_only=True)
    model.to(device)

    dataset = NERDataSequence(token_texts_list, tokenizer, label_ids_list)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size],
                                                               generator=torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(train_dataset, batch_size=16)
    val_dataloader = DataLoader(val_dataset, batch_size=16)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    plt = livelossplot.PlotLosses()
    pbar_1 = trange(epochs, desc="Epoch", leave=False)
    pbar_2 = trange(len(train_dataloader), desc="Batch", leave=False)
    best_val_acc = 0
    val_plateau_counter = 0

    for epoch in range(epochs):
        pbar_1.update(1)
        with tqdm(total=len(train_dataloader), desc="Training batches", leave=False) as pbar_2:
            model.train()
            train_loss_sum = 0
            for batch in train_dataloader:
                pbar_2.update(1)
                input_ids, attention_mask, gt_label_ids = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                gt_label_ids = gt_label_ids.to(device)
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=None, labels=gt_label_ids)
                loss = outputs[0]
                train_loss_sum += loss.item()
                loss.backward()
                optimizer.step()

        with tqdm(total=len(val_dataloader), desc="Validation batches", leave=False) as pbar_3:
            # Calculate percentage accuracy on the validation set
            model.eval()
            val_label_count = 0
            val_correct_count = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    pbar_3.update(1)
                    input_ids, attention_mask, gt_label_ids = batch
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    gt_label_ids = gt_label_ids.to(device)
                    for labels in gt_label_ids.squeeze():
                        for label in labels:
                            if label != -100:
                                val_label_count += 1
                    outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=None)
                    logits = outputs[0]
                    logits = logits.detach().cpu().numpy()
                    true_label_ids = gt_label_ids.to('cpu').numpy()
                    pred_label_ids = np.argmax(logits, axis=2)
                    for i in range(len(logits)):
                        for j in range(len(logits[i])):
                            if pred_label_ids[i][j] == true_label_ids[i][j] and true_label_ids[i][j] != -100:
                                val_correct_count += 1

        val_acc = val_correct_count / val_label_count
        if epoch >= 5:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                val_plateau_counter = 0
            else:
                val_plateau_counter += 1
                if val_plateau_counter >= 5:
                    print("\nEarly stopping at epoch " + str(epoch) + "!")
                    break
        pbar_1.set_postfix(
            {"train loss": train_loss_sum / len(train_dataloader), "val accuracy": val_acc,
             "best_val accuracy": best_val_acc})
        plt.update(
            {"train loss": train_loss_sum / len(train_dataloader), "val accuracy": val_acc,
             "best_val accuracy": best_val_acc})

    model.save_pretrained(output_model_filepath)
    tokenizer.save_pretrained(output_tokenizer_filepath)
    plt.send()
