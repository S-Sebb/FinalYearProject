# -*- coding: utf-8 -*-
import nltk
import torch
from transformers import BertTokenizer

label_convert_dict = {"OBJECTIVE&BACKGROUND": 0, "METHODS": 1, "CONCLUSIONS": 2, "RESULTS": 3}
reverse_label_convert_dict = {v: k for k, v in label_convert_dict.items()}


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


def init_model_tokenizer():
    model = torch.load('model.pt')
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased',
        do_lower_case=True
    )
    return model, tokenizer


def classify_abstract(abstract_text, model, tokenizer):
    input_lines = nltk.sent_tokenize(abstract_text.text)
    input_lines = [line.strip() for line in input_lines if line.strip() != ""]

    line_label_list = []

    for line in input_lines:
        prediction_input = preprocess_text(line, tokenizer)
        prediction_input_ids = prediction_input['input_ids'].to('cuda')
        prediction_attention_masks = prediction_input['attention_mask'].to('cuda')
        predict = model(prediction_input_ids, prediction_attention_masks)
        label_num = torch.argmax(predict[0], dim=1).item()
        line_label_list.append((line, reverse_label_convert_dict[label_num]))
    return line_label_list


input_line = """
The current objective of antiglaucomatous therapy is to reduce intraocular pressure (IOP), and thus to preserve visual function. Many ophthalmologists believe this objective is best achieved by methods that improve ocular blood flow to the optic nerve head. Beta-blockers are effective ocular hypotensive agents, but they can reduce choroidal blood flow. Bimatoprost, a new prostamide analogue, has been shown to have a better IOP-lowering effect compared with the nonselective beta-adrenergic receptor blocker timolol maleate, but little is known about its effects on the vascular bed of the eye.

The aim of this study was to compare the effects of bimatoprost and timolol on IOP and choroidal blood flow (as measured using pulsatile ocular blood flow [pOBF]) in patients with primary open-angle glaucoma (POAG).

This prospective, open-label, randomized, 2-arm, parallel-group study was conducted at the Glaucoma Research Centre, Department of Ophthalmology, University Hospital of Bari, Bari, Italy. Patients with POAG having well-controlled IOP (<16 mm Hg) on monotherapy with timolol 0.5% ophthalmic solution (2 drops per affected eye BID) for ≥12 months but with a progressive decrease in pOBF during the same time period were randomly allocated to 1 of 2 treatment groups. One group continued monotherapy with timolol, 2 drops per affected eye BID. The other group was switched (without washout) to bimatoprost 0.3% ophthalmic solution (2 drops per affected eye QD [9 pm]). Treatment was given for 180 days. IOP and pOBF were assessed at the diagnostic visit (pre-timolol), baseline (day 0), and treatment days 15, 30, 60, 90, and 180. Primary adverse effects (AEs) (ie, conjunctival hyperemia, conjunctival papillae, stinging, burning, foreign body sensation, and pigmentation of periorbital skin) were monitored throughout the study.

Thirty-eight patients were enrolled (22 men, 16 women; mean [SD] age, 51.7 [4.8] years; 19 patients per treatment group; 38 eligible eyes). At 180-day follow-up in the timolol group, the IOP and the pOBF remained unchanged compared with baseline. In the bimatoprost group the IOP remained unchanged and the pOBF was decreased by 38.9% compared with baseline (P < 0.01). All AEs were mild to moderate and included conjunctival hyperemia and ocular itching (5 patients [26.3%] in the bimatoprost group) and pigmentation of periorbital skin (2 patients [40.0%] in the bimatoprost group). The incidence of each AE was higher in the bimatoprost group than in the timolol group (P = 0.008).

In this population of patients with POAG, bimatoprost was associated with increased pOBF, and the reduction in pOBF associated with timolol was corrected after patients were switched to bimatoprost. Bimatoprost was associated with increased choroidal blood flow, beyond the levels recorded before timolol treatment. The decreased IOP level achieved in the timolol group seemed to be improved further by bimatoprost. Considering the potential efficacy of bimatoprost on IOP and pOBF, we suggest that this new drug may represent a clinical advance in the medical treatment of POAG.
"""

if __name__ == "__main__":
    model, tokenizer = init_model_tokenizer()
    line_label_list = classify_abstract(input_line, model, tokenizer)
    for line, label in line_label_list:
        print(f"{label}: {line}")
