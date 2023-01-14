# -*- coding: utf-8 -*-
import torch
from transformers import BertTokenizer

input_line = "In a study population of black Africans with advanced glaucoma in Ghana we conducted a prospective study of intraoperative 5-fluorouracil alone. Eyes undergoing trabeculectomy were randomly selected either to receive or not receive a single intraoperative application of 5-fluorouracil (50 mg/ml for five minutes). Fifty-five eyes had a mean follow-up of 282 days (minimum, 92 days). Twenty of 24 eyes (83%) in the 5-fluorouracil group vs 12 of 31 eyes (39%) in the control group had postoperative intraocular pressure of 20 mm Hg or less with or without medical therapy (P = .01). Eleven of 24 eyes (46%) in the 5-fluorouracil group and five of 31 eyes (16%) in the control group had intraocular pressure of 15 mm Hg or less (P = .02). Without medical therapy, 17 of 24 eyes (71%) in the 5-fluorouracil group and ten of 31 eyes (32%) in the control group had intraocular pressure of 20 mm Hg or less (P = .02). The overall complications were similar in the two groups. In this population, intraoperative 5-fluorouracil markedly improved the ability of trabeculectomy to lower intraocular pressure. We recommend that intraoperative 5-fluorouracil be considered in glaucoma surgery with poor prognosis as an alternative to postoperative subconjunctival injections when multiple injections are not feasible."
input_lines = input_line.lower().replace(". ", ".\n").split("\n")

model = torch.load('model.pt')
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


for line in input_lines:
    prediction_input = preprocess_text(line, tokenizer)
    prediction_input_ids = prediction_input['input_ids'].to('cuda')
    prediction_attention_masks = prediction_input['attention_mask'].to('cuda')
    predict = model(prediction_input_ids, prediction_attention_masks)
    print(line)
    print(torch.argmax(predict[0], dim=1).item())
    print("-------------------")
