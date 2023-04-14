# -*- coding: utf-8 -*-

import numpy as np  # https://numpy.org/doc/stable/
import pandas as pd  # https://pandas.pydata.org/
import spacy_streamlit  # https://spacy.io/universe/project/spacy-streamlit

from predict import *
from utils import *

# set page config
st.set_page_config(
    page_title="RCT Abstract Analyzer"
)

st.title("RCT Abstract Analyzer")

default_text = """Lithium remains an important treatment for mood disorders but is associated with kidney disease. Nephrogenic diabetes insipidus (NDI) is associated with up to 3-fold risk of incident chronic kidney disease among lithium users. There are limited randomized controlled trials (RCT) for treatments of lithium-induced NDI, and existing therapies can be poorly tolerated. Therefore, novel treatments are needed for lithium-induced NDI.

We conducted a 12-week double-blind pilot RCT to assess the feasibility and efficacy of 20 mg/d atorvastatin vs placebo in the treatment of NDI in chronic lithium users. Patients, recruited between September 2017 and October 2018, were aged 18 to 85, currently on a stable dose of lithium, and determined to have NDI.

Urinary osmolality (UOsm) at 12 weeks adjusted for baseline was not statistically different between groups (+39.6 mOsm/kg [95% CI, -35.3, 114.5] in atorvastatin compared to placebo groups). Secondary outcomes of fluid intake and aquaporin-2 excretions at 12 weeks adjusted for baseline were -0.13 L [95% CI, -0.54, 0.28] and 98.68 [95% CI, -190.34, 387.70], respectively. A moderate effect size was observed for improvements in baseline UOsm by ≥100 mOsm/kg at 12 weeks in patients who received atorvastatin compared to placebo (38.45% (10/26) vs 22.58% (7/31); Cohen's d = 0.66).

Among lithium users with NDI, atorvastatin 20 mg/d did not significantly improve urinary osmolality compared to placebo over a 12-week period. Larger confirmatory trials with longer follow-up periods may help to further assess the effects of statins on NDI, especially within patients with more severe NDI."""

input_abstract_text = st.text_area("Paste full RCT paper abstract below", default_text, height=500)

section_labels_to_ids = {"BACKGROUND": 0, "OBJECTIVE": 1, "METHODS": 2, "RESULTS": 3, "CONCLUSIONS": 4}
section_ids_to_labels = {v: k for k, v in section_labels_to_ids.items()}
participant_labels_to_ids = {"NON-PATIENT": 0, "PATIENT": 1}
participant_ids_to_labels = {v: k for k, v in participant_labels_to_ids.items()}
NER_labels_to_ids = {"O": 0, "INTV": 1, "OC": 2, "MEAS": 3}
NER_ids_to_labels = {v: k for k, v in NER_labels_to_ids.items()}

try:
    section_classification_model, section_classification_tokenizer, participant_classification_model, \
        participant_classification_tokenizer, NER_model, NER_tokenizer = try_load_predefined_models_tokenizers()
except Exception as e:
    print(e)
    st.error(
        "Something went wrong. Please check the model files are in the correct location and try again.")
    st.stop()
    exit()

spacy_parser = init_spacy_parser()

paragraphs = abstract_to_paragraphs(input_abstract_text)
pred_section_labels = predict_classification(paragraphs, section_ids_to_labels, section_classification_model,
                                             section_classification_tokenizer)
section_dict = make_section_dict(paragraphs, pred_section_labels)

INTV_entities = []
OC_entities = []
# Processing the RESULTS section
if "RESULTS" in section_dict:
    result_paragraph = section_dict["RESULTS"]
    result_spacy_doc = spacy_parser.make_doc(result_paragraph)
    spacy_tokenized_result_paragraph = [[token.text for token in result_spacy_doc]]
    NER_pred_labels = predict_NER(spacy_tokenized_result_paragraph, NER_ids_to_labels, NER_model, NER_tokenizer)[0]
    result_spacy_doc = align_spacy_doc_entity(result_spacy_doc, NER_pred_labels)
    INTV_entities = np.unique([ent.text.lower() for ent in result_spacy_doc.ents if ent.label_ == "INTV"])
    OC_entities = np.unique([ent.text.lower() for ent in result_spacy_doc.ents if ent.label_ == "OC"])

# Processing the METHODS section
methods_paragraph = section_dict["METHODS"]
methods_sentences = paragraph_to_sentences(methods_paragraph)
pred_participant_labels = predict_classification(methods_sentences, participant_ids_to_labels,
                                                 participant_classification_model,
                                                 participant_classification_tokenizer)
patient_sentences = [sentence for sentence, label in zip(methods_sentences, pred_participant_labels) if
                     label == "PATIENT"]
patient_paragraph = " ".join(patient_sentences)
methods_spacy_doc = spacy_parser.make_doc(methods_paragraph)
spacy_tokenized_methods_paragraph = [token.text for token in methods_spacy_doc]
fake_NER_pred_labels = ["O"] * len(spacy_tokenized_methods_paragraph)
INTV_words = []
if len(INTV_entities) > 0:
    for INTV_entity in INTV_entities:
        INTV_words.extend(INTV_entity.split())
    for i, token in enumerate(spacy_tokenized_methods_paragraph):
        if token.lower() in INTV_words:
            fake_NER_pred_labels[i] = "INTV"
methods_spacy_doc = align_spacy_doc_entity(methods_spacy_doc, fake_NER_pred_labels)

st.markdown("""---""")

st.title("Analysis Result")

for section_label in section_labels_to_ids.keys():
    if section_label not in section_dict.keys():
        continue
    paragraph = section_dict[section_label]
    st.subheader(section_label)
    if section_label == "METHODS":
        spacy_ner_labels = ['INTV']
        spacy_streamlit.visualize_ner(methods_spacy_doc, labels=spacy_ner_labels, show_table=False, title=None,
                                      key=section_label)
    elif section_label == "RESULTS":
        spacy_ner_labels = ['INTV', 'OC', 'MEAS']
        spacy_streamlit.visualize_ner(result_spacy_doc, labels=spacy_ner_labels, show_table=False, title=None,
                                      key=section_label)
    else:
        st.write(paragraph)

st.markdown("""---""")

st.title("Key Information")
if patient_paragraph.strip() != "":
    patient_df = pd.DataFrame({"Patient": [patient_paragraph]})
    st.table(patient_df)
if len(INTV_entities) > 0:
    INTV_df = pd.DataFrame({"Intervention": INTV_entities})
    st.table(INTV_df)
if len(OC_entities) > 0:
    OC_df = pd.DataFrame({"Outcome": OC_entities})
    st.table(OC_df)
