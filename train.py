# -*- coding: utf-8 -*-
import json
import os
import time

import livelossplot
import numpy as np
import pandas as pd
import torch
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
    total_label_counts = sum(label_counts_dict.values())
    class_weights = []
    for i in range(num_labels):
        class_weights.append(1 - label_counts_dict[i] / total_label_counts)
    if custom_weights_flag:
        print("Class weights: %s" % class_weights)

    # Suppress warnings from training a pre-trained model
    logging.set_verbosity_error()

    tokenizer, model = load_pretrained_classification_tokenizer_model(num_labels)
    model.to(device)

    train_dataset = ClassificationDataSequence(lines, tokenizer, label_ids)

    train_dataloader = DataLoader(train_dataset, batch_size=8)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if custom_weights_flag:
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    pbar_1 = trange(epochs, desc='Epoch', leave=False)
    plt = livelossplot.PlotLosses()

    plateau_counter = 0
    last_train_loss = 9999

    # Start timing
    start_time = time.time()

    for epoch in pbar_1:
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

        avg_train_loss = train_loss_sum / len(train_dataloader)
        if abs(last_train_loss - avg_train_loss) > 0.001:
            plateau_counter = 0
        else:
            plateau_counter += 1
            if epoch >= 5:  # Train for at least 5 epochs
                if plateau_counter >= 3:
                    print("\nEarly stopping at epoch " + str(epoch))
                    break
        pbar_1.set_postfix(
            {"train loss": avg_train_loss, "plateau counter": plateau_counter})
        plt.update(
            {"train loss": avg_train_loss})
        last_train_loss = avg_train_loss

    # End timing
    end_time = time.time()
    print("Training time: {:.3f} minutes".format((end_time - start_time) / 60))

    model.save_pretrained(output_model_filepath)
    tokenizer.save_pretrained(output_tokenizer_filepath)

    plt.send()


def train_NER(output_dir_filepath, output_model_filename, output_tokenizer_filename, NER_dataset_filepath,
              labels_to_ids, epochs, learning_rate, weight_decay, model_type, custom_weights_flag):
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
    label_counts_dict = {}
    for label_ids in label_ids_list:
        for label_id in label_ids:
            if label_id not in label_counts_dict:
                label_counts_dict[label_id] = 0
            label_counts_dict[label_id] += 1
    print("Label counts: %s" % label_counts_dict)
    class_weights = []
    total_label_counts = sum(label_counts_dict.values())
    class_portions = []
    for i in range(len(labels_to_ids.keys())):
        class_portions.append(label_counts_dict[i] / total_label_counts)
    print("Class portions: %s" % class_portions)
    for i in range(len(labels_to_ids.keys())):
        class_weights.append(1 - label_counts_dict[i] / total_label_counts)
    if custom_weights_flag:
        print("Class weights: %s" % class_weights)

    # Suppress warnings from training a pre-trained model
    logging.set_verbosity_error()

    tokenizer, model = load_pretrained_NER_tokenizer_model(model_type, len(labels_to_ids.keys()))
    model.to(device)

    train_dataset = NERDataSequence(token_texts_list, tokenizer, label_ids_list)
    train_dataloader = DataLoader(train_dataset, batch_size=16)

    if custom_weights_flag:
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    plt = livelossplot.PlotLosses()
    pbar_1 = trange(epochs, desc="Epoch", leave=False)

    last_train_loss = 9999
    plateau_counter = 0

    for epoch in pbar_1:
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
                outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=None)
                logits = outputs.logits
                loss = loss_fn(logits.view(-1, logits.shape[-1]), gt_label_ids.view(-1))
                train_loss_sum += loss.item()
                loss.backward()
                optimizer.step()

        avg_train_loss = train_loss_sum / len(train_dataloader)
        if abs(last_train_loss - avg_train_loss) > 0.001:
            plateau_counter = 0
        else:
            plateau_counter += 1
            if epoch >= 5:  # Train for at least 5 epochs
                if plateau_counter >= 3:
                    print("\nEarly stopping at epoch " + str(epoch))
                    break
        pbar_1.set_postfix(
            {"train loss": avg_train_loss, "plateau counter": plateau_counter})
        plt.update(
            {"train loss": avg_train_loss})
        last_train_loss = avg_train_loss

    model.save_pretrained(output_model_filepath)
    tokenizer.save_pretrained(output_tokenizer_filepath)
    plt.send()
