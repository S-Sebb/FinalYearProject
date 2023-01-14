# -*- coding: utf-8 -*-
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import trange

abstract_labeling_filepath = "abstract_labeling.csv"
df = pd.read_csv(abstract_labeling_filepath, index_col=0)

lines = df.line.values
labels = df.label.values
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case=True
)


def preprocess_text(text, tokenizer):
    processed_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        padding="max_length",
        max_length=128,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return processed_text


token_id = []
attention_masks = []

for line in lines:
    processed_text_dict = preprocess_text(line, tokenizer)
    token_id.append(processed_text_dict['input_ids'])
    attention_masks.append(processed_text_dict['attention_mask'])

token_id = torch.cat(token_id, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

train_set = TensorDataset(token_id, attention_masks, labels)
train_dataloader = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=8)

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, eps=1e-08)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

epochs = 1

for _ in trange(epochs, desc='Epoch'):
    model.train()

    # Tracking variables
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    for step, batch in enumerate(train_dataloader):
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
        tr_loss += train_output.loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

    print('\n\t - Train loss: {:.4f}'.format(tr_loss / nb_tr_steps))

torch.save(model, 'model.pt')
