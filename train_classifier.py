# -*- coding: utf-8 -*-
from copy import deepcopy

import livelossplot
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tqdm import trange, tqdm
from transformers import BertTokenizer, BertForSequenceClassification, logging


def preprocess_text(text, classifier_tokenizer):
    processed_text = classifier_tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        padding="max_length",
        max_length=256,
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt'
    )
    return processed_text


if __name__ == "__main__":
    classifier_dataset_filepath = "classifier_dataset.csv"
    num_labels = 4
    output_model_filepath = "classifier_model.pt"
    output_tokenizer_filepath = "classifier_tokenizer.pt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv(classifier_dataset_filepath, index_col=0)
    lines = df.line.values
    labels = df.label.values

    label_counts = {}
    for label in labels:
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

    # Suppress warnings from training a pre-trained model
    logging.set_verbosity_error()

    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased',
        do_lower_case=True
    )

    token_id = []
    attention_masks = []

    for line in lines:
        processed_text_dict = preprocess_text(line, tokenizer)
        token_id.append(processed_text_dict['input_ids'])
        attention_masks.append(processed_text_dict['attention_mask'])

    token_id = torch.cat(token_id, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    all_set = TensorDataset(token_id, attention_masks, labels)
    train_set, val_set = torch.utils.data.random_split(all_set,
                                                       [int(len(all_set) * 0.8),
                                                        len(all_set) - int(len(all_set) * 0.8)])
    train_dataloader = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=16)
    val_dataloader = DataLoader(val_set, sampler=RandomSampler(val_set), batch_size=16)

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, eps=1e-08, weight_decay=1e-8)

    model.to(device)

    epochs = 20

    pbar_1 = trange(epochs, desc='Epoch', leave=False)
    postfix_dict = {}

    plt = livelossplot.PlotLosses()

    best_eval_accuracy = 0
    best_model = None

    for epoch in pbar_1:
        model.train()

        # Tracking variables
        train_loss = 0
        train_num, train_steps = 0, 0
        with tqdm(total=len(train_dataloader), desc="Training batches", leave=False) as pbar_2:
            for step, batch in enumerate(train_dataloader):
                pbar_2.update(1)
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                optimizer.zero_grad()
                # Forward pass
                train_output = model(b_input_ids,
                                     token_type_ids=None,
                                     attention_mask=b_input_mask,
                                     labels=b_labels)
                # Backward pass
                train_output.loss.backward()
                optimizer.step()
                # Update tracking variables
                train_loss += train_output.loss.item()
                train_num += b_input_ids.size(0)
                train_steps += 1
            pbar_2.close()

        model.eval()
        eval_correct_num = 0
        eval_num = 0
        with tqdm(total=len(val_dataloader), desc="Validation batches", leave=False) as pbar_2:
            for batch in val_dataloader:
                pbar_2.update(1)
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                with torch.no_grad():
                    val_output = model(b_input_ids,
                                       token_type_ids=None,
                                       attention_mask=b_input_mask,
                                       labels=b_labels)
                # Update tracking variables
                eval_correct_num += torch.sum(torch.argmax(val_output.logits, dim=1) == b_labels).item()
                eval_num += b_input_ids.size(0)
            pbar_2.close()
        eval_accuracy = eval_correct_num / eval_num

        pbar_1.set_postfix(postfix_dict)
        if epoch > epochs * 0.5:
            if eval_accuracy > best_eval_accuracy:
                best_eval_accuracy = eval_accuracy
                best_model = deepcopy(model)

        postfix_dict['train_loss'] = train_loss / train_steps
        postfix_dict['eval_accuracy'] = eval_accuracy
        postfix_dict['eval_correct_num'] = "{}/{}".format(eval_correct_num, eval_num)

        plt.update({"train_loss": train_loss / train_steps, "eval_accuracy": eval_accuracy})

    torch.save(best_model, output_model_filepath)
    tokenizer.save_pretrained(output_tokenizer_filepath)

    plt.send()
