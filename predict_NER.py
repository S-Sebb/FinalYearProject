# -*- coding: utf-8 -*-

import torch
from transformers import BertTokenizer


def init_model_tokenizer():
    model = torch.load('NER.pt')
    tokenizer = BertTokenizer.from_pretrained("NER_tokenizer.pt")
    return model, tokenizer


if __name__ == '__main__':
    input_text = """Objectives: To compare health-related quality of life (HRQoL) in newly diagnosed, treatment-naive patients with OAG or OHT, treated with two treatment pathways: topical IOP-lowering medication from the outset (Medicine-1st) or primary SLT followed by topical medications as required (Laser-1st). We also compared the clinical effectiveness and cost-effectiveness of the two pathways.

    Design: A 36-month pragmatic, unmasked, multicentre randomised controlled trial.
    
    Settings: Six collaborating specialist glaucoma clinics across the UK.
    
    Participants: Newly diagnosed patients with OAG or OHT in one or both eyes who were aged ≥ 18 years and able to provide informed consent and read and understand English. Patients needed to qualify for treatment, be able to perform a reliable visual field (VF) test and have visual acuity of at least 6 out of 36 in the study eye. Patients with VF loss mean deviation worse than -12 dB in the better eye or -15 dB in the worse eye were excluded. Patients were also excluded if they had congenital, early childhood or secondary glaucoma or ocular comorbidities; if they had any previous ocular surgery except phacoemulsification, at least 1 year prior to recruitment or any active treatment for ophthalmic conditions; if they were pregnant; or if they were unable to use topical medical therapy or had contraindications to SLT.
    
    Interventions: SLT according to a predefined protocol compared with IOP-lowering eyedrops, as per national guidelines.
    
    Main outcome measures: The primary outcome was HRQoL at 3 years [as measured using the EuroQol-5 Dimensions, five-level version (EQ-5D-5L) questionnaire]. Secondary outcomes were cost and cost-effectiveness, disease-specific HRQoL, clinical effectiveness and safety.
    """

    ids_to_label = {0: 'INTV', 1: 'MEAS', 2: 'O', 3: 'OC'}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = init_model_tokenizer()
    model.eval()
    model.to(device)

    tokenized_text = tokenizer(
        input_text,
        add_special_tokens=True,
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
    logits = outputs.logits
    logits = logits.squeeze().detach().cpu().numpy()
    label_ids = logits.argmax(axis=1)
    labels = [ids_to_label[label_id] for label_id in label_ids]
    for token, label in zip(tokens, labels):
        print(token, label)
