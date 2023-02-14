# -*- coding: utf-8 -*-

import torch
from transformers import BertTokenizer


def init_model_tokenizer():
    model = torch.load('METHODS.pt')
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased',
        do_lower_case=True
    )
    return model, tokenizer


if __name__ == '__main__':
    input_text = """
    Crossover, double-masked, randomized clinical trial. Participants recruited from two tertiary care centres.
    Fifty-seven participants, diagnosed and treated for glaucoma.
    Participants received oral placebo or nicotinamide and reviewed six-weekly. Participants commenced 6 weeks of 1.5 g/day then 6 weeks of 3.0 g/day followed by crossover without washout. Visual function measured using electroretinography and perimetry.
    Change in inner retinal function, determined by photopic negative response (PhNR) parameters: saturated PhNR amplitude (Vmax), ratio of PhNR/b-wave amplitude (Vmax ratio)."""

    ids_to_label = {0: 'B-Intervention', 1: 'B-Intervention_Duration', 2: 'B-Outcome_Measures', 3: 'B-Patient',
                    4: 'I-Intervention', 5: 'I-Intervention_Duration', 6: 'I-Outcome_Measures', 7: 'I-Patient', 8: 'O-'}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = init_model_tokenizer()
    model.eval()
    model.to(device)

    tokenized_text = tokenizer(
        input_text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = tokenized_text['input_ids'].to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    attention_mask = tokenized_text['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    logits = outputs[0]
    logits = logits.squeeze().detach().cpu().numpy()
    label_ids = logits.argmax(axis=1)
    labels = [ids_to_label[label_id] for label_id in label_ids]
    for token, label in zip(tokens, labels):
        print(token, label)


