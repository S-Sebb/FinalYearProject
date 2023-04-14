# -*- coding: utf-8 -*-
import numpy as np  # https://numpy.org/doc/stable/
import pandas as pd  # https://pandas.pydata.org/
import tqdm  # https://tqdm.github.io/

from predict import *
from utils import *

input_dirname = "input"
input_filename = "input.csv"

output_dirname = "output"
output_filename = "output.csv"

input_filepath = os.path.join(root_dir, input_dirname, input_filename)
output_filepath = os.path.join(root_dir, output_dirname, output_filename)

with open(input_filepath, "r") as f:
    df = pd.read_csv(f)

section_labels_to_ids = {"BACKGROUND": 0, "OBJECTIVE": 1, "METHODS": 2, "RESULTS": 3, "CONCLUSIONS": 4}
section_ids_to_labels = {v: k for k, v in section_labels_to_ids.items()}
participant_labels_to_ids = {"NON-PATIENT": 0, "PATIENT": 1}
participant_ids_to_labels = {v: k for k, v in participant_labels_to_ids.items()}
NER_labels_to_ids = {"O": 0, "INTV": 1, "OC": 2, "MEAS": 3}
NER_ids_to_labels = {v: k for k, v in NER_labels_to_ids.items()}

section_classification_model, section_classification_tokenizer, participant_classification_model, \
    participant_classification_tokenizer, NER_model, NER_tokenizer = try_load_predefined_models_tokenizers()

spacy_parser = init_spacy_parser()

output_df_dict = {"input_abstract_text": [], "background": [], "objective": [], "methods": [], "results": [],
                  "conclusions": [],
                  "participant": [], "intervention": [], "outcome": []}

input_abstract_texts = df["abstract_text"].values
input_abstract_texts = [input_abstract_text for input_abstract_text in input_abstract_texts if
                        input_abstract_text is not np.nan and input_abstract_text.strip() != ""]

p_bar = tqdm.tqdm(total=len(input_abstract_texts), desc="Analysing and tabulating abstracts", leave=False)

for input_abstract_text in input_abstract_texts:
    p_bar.update(1)
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

    output_df_dict["input_abstract_text"].append(input_abstract_text)
    output_df_dict["background"].append(section_dict.get("BACKGROUND", ""))
    output_df_dict["objective"].append(section_dict.get("OBJECTIVE", ""))
    output_df_dict["methods"].append(section_dict.get("METHODS", ""))
    output_df_dict["results"].append(section_dict.get("RESULTS", ""))
    output_df_dict["conclusions"].append(section_dict.get("CONCLUSIONS", ""))
    output_df_dict["participant"].append(patient_paragraph)
    output_df_dict["intervention"].append(", ".join(INTV_entities))
    output_df_dict["outcome"].append(", ".join(OC_entities))

p_bar.close()

output_df = pd.DataFrame(output_df_dict)
output_df.to_csv(output_filepath)

print("Tabulation complete. Output file saved to '{}'.".format(output_filepath))
