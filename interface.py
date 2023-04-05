# -*- coding: utf-8 -*-
import spacy_streamlit
import streamlit as st

from predict_NER import init_ner_model_tokenizer, init_spacy_parser, text_ner, create_spacy_entities, make_spacy_doc
from predict_classifier import init_classifier_model_tokenizer, classify_abstract, classify_patient, sentence_tokenize

classifier_model_path = "classifier_model.pt"
classifier_tokenizer_path = "classifier_tokenizer.pt"
patient_classifier_model_path = "patient_classifier_model.pt"
patient_classifier_tokenizer_path = "patient_classifier_tokenizer.pt"
classifier_ids_to_labels = {0: "OBJECTIVE&BACKGROUND", 1: "METHODS", 2: "RESULTS", 3: "CONCLUSIONS"}

# set page config
st.set_page_config(
    page_title="RCT Abstract Analyzer"
)

st.title("RCT Abstract Analyzer")

default_text = """Although exercise has been addressed as an adjuvant treatment for anxiety, depression and cancer-related symptoms, limited studies have evaluated the effectiveness of exercise in patients with lung cancer.

We recruited 116 patients from a medical centre in northern Taiwan, and randomly assigned them to either a walking-exercise group (n=58) or a usual-care group (n=58). We conducted a 12-week exercise programme that comprised home-based, moderate-intensity walking for 40 min per day, 3 days per week, and weekly exercise counselling. The outcome measures included the Hospital Anxiety and Depression Scale and the Taiwanese version of the MD Anderson Symptom Inventory.

We analysed the effects of the exercise programme on anxiety, depression and cancer-related symptoms by using a generalised estimating equation method. The exercise group patients exhibited significant improvements in their anxiety levels over time (P=0.009 and 0.006 in the third and sixth months, respectively) and depression (P=0.00006 and 0.004 in the third and sixth months, respectively) than did the usual-care group patients.

The home-based walking exercise programme is a feasible and effective intervention method for managing anxiety and depression in lung cancer survivors and can be considered as an essential component of lung cancer rehabilitation.
"""

text = st.text_area("Paste full RCT paper abstract below", default_text, height=200)

classifier_model, classifier_tokenizer = init_classifier_model_tokenizer(classifier_model_path,
                                                                         classifier_tokenizer_path)
ner_model, ner_tokenizer = init_ner_model_tokenizer("NER_model.pt", "NER_tokenizer.pt")
patient_classifier_model, patient_classifier_tokenizer = init_classifier_model_tokenizer(patient_classifier_model_path,
                                                                                         patient_classifier_tokenizer_path)
spacy_parser = init_spacy_parser()

classify_result = classify_abstract(text, classifier_model, classifier_tokenizer, classifier_ids_to_labels)

st.markdown("""---""")

st.title("Analysis Result")

patient_lines = []
intervention_entities = []
outcome_entities = []

for classifier_label, paragraph_text in classify_result.items():
    st.subheader(classifier_label)
    if paragraph_text.strip() == "":
        st.write("No information found.")
        continue

    if classifier_label == "METHODS":
        sentences = sentence_tokenize(paragraph_text)
        for sentence in sentences:
            patient_flag = classify_patient(sentence, patient_classifier_model, patient_classifier_tokenizer)
            if patient_flag == 1:
                patient_lines.append(sentence)

        ner_labels = ['INTV', 'PATIENTS']
        ner_ids_to_labels = {0: 'INTV', 2: 'O'}
        doc = make_spacy_doc(paragraph_text, spacy_parser)
        words = [token.text for token in doc]
        ner_predict_labels = text_ner(words, ner_model, ner_tokenizer, ner_ids_to_labels)
        doc = create_spacy_entities(doc, ner_predict_labels)
        for ent in doc.ents:
            if ent.label_ == 'INTV':
                intervention_entities.append(ent.text)

        spacy_streamlit.visualize_ner(doc, labels=ner_labels, show_table=False, title=None, key=classifier_label)

    elif classifier_label == "RESULTS":
        ner_labels = ['INTV', 'MEAS', 'OC']
        ner_ids_to_labels = {0: 'INTV', 1: 'MEAS', 2: 'O', 3: 'OC'}
        doc = make_spacy_doc(paragraph_text, spacy_parser)
        words = [token.text for token in doc]
        ner_predict_labels = text_ner(words, ner_model, ner_tokenizer, ner_ids_to_labels)
        doc = create_spacy_entities(doc, ner_predict_labels)
        for ent in doc.ents:
            if ent.label_ == "OC":
                outcome_entities.append(ent.text)

        spacy_streamlit.visualize_ner(doc, labels=ner_labels, show_table=False, title=None, key=classifier_label)

    else:
        st.write(paragraph_text)

st.markdown("""---""")

st.header("Key Information")

st.subheader("Patient Information")
if patient_lines:
    st.write("\n\n".join(patient_lines))
else:
    st.write("No patient information found")

st.subheader("Intervention")
if intervention_entities:
    st.write("\n\n".join(intervention_entities))
else:
    st.write("No intervention information found")

st.subheader("Outcome Measures")
if outcome_entities:
    st.write("\n\n".join(outcome_entities))
else:
    st.write("No outcome measures information found")

# if len(ner_dict["intervention"]) < len(ner_dict["outcome measures"]):
#     ner_dict["intervention"].extend([""] * (len(ner_dict["outcome measures"]) - len(ner_dict["intervention"])))
# elif len(ner_dict["intervention"]) > len(ner_dict["outcome measures"]):
#     ner_dict["outcome measures"].extend([""] * (len(ner_dict["intervention"]) - len(ner_dict["outcome measures"])))
#
# st.title("Key Information Table")
#
# result_df = pd.DataFrame(ner_dict)
# st.dataframe(result_df, width=1000)
