# -*- coding: utf-8 -*-
import os
import time
from xml.dom import minidom
from xml.etree import ElementTree

import pandas as pd
from pymed import PubMed
from sklearn.model_selection import train_test_split

pubmed = PubMed(email="794194678@qq.com")

section_dataset_output_filename = "5Section_classifier_dataset.csv"
participant_dataset_output_filename = "participant_classifier_dataset.csv"
section_dataset_output_dir_filepath = os.path.join("datasets", "section classifier datasets")
participant_dataset_output_dir_filepath = os.path.join("datasets", "participant classifier datasets")

if not os.path.exists(section_dataset_output_dir_filepath):
    os.makedirs(section_dataset_output_dir_filepath)
if not os.path.exists(participant_dataset_output_dir_filepath):
    os.makedirs(participant_dataset_output_dir_filepath)
train_section_dataset_output_filepath = os.path.join(section_dataset_output_dir_filepath,
                                                     "train_" + section_dataset_output_filename)
test_section_dataset_output_filepath = os.path.join(section_dataset_output_dir_filepath,
                                                    "test_" + section_dataset_output_filename)
train_participant_dataset_output_filepath = os.path.join(participant_dataset_output_dir_filepath,
                                                         "train_" + participant_dataset_output_filename)
test_participant_dataset_output_filepath = os.path.join(participant_dataset_output_dir_filepath,
                                                        "test_" + participant_dataset_output_filename)

labels_to_ids = {"BACKGROUND": 0, "OBJECTIVE": 0, "METHODS": 1, "RESULTS": 2, "CONCLUSIONS": 3}
ids_to_labels = {0: "BACKGROUND&OBJECTIVE", 1: "METHODS", 2: "RESULTS", 3: "CONCLUSIONS"}

section_lines = []
section_labels = []
participant_lines = []
participant_labels = []
diseases = ["glaucoma", "lung cancer", "diabetes"]
for disease in diseases:
    print("Collecting data for %s" % disease)
    success = False
    result = []
    while not success:
        try:
            query = "(Randomized Controlled Trial[Publication Type]) AND (%s)" % disease
            result = pubmed.query(query, max_results=5000)
            success = True
        except Exception as e:
            print("Error %s\nRetrying" % e)
            time.sleep(30)

    for i, article in enumerate(result):
        xml_root = article.toDict()["xml"]
        xml_str = ElementTree.tostring(xml_root)
        xml_str = minidom.parseString(xml_str).toprettyxml(indent="   ")
        # print(xml_str)
        medline_citation = xml_root.find("MedlineCitation")
        article = medline_citation.find("Article")
        title = article.find("ArticleTitle").input_abstract_text
        # print(title)
        abstract = article.find("Abstract")
        if abstract is None:
            continue
        intervention = False
        for abstract_text in abstract.findall("AbstractText"):
            if abstract_text.attrib.get("Label") == "INTERVENTION" or abstract_text.attrib.get(
                    "Label") == "INTERVENTIONS":
                intervention = True
                break
        if not intervention:
            continue
        article_with_patient = False
        non_patient_lines = []
        patient_lines = []
        for abstract_text in abstract.findall("AbstractText"):
            if "NlmCategory" not in abstract_text.attrib:
                continue
            if abstract_text.input_abstract_text is None or abstract_text.input_abstract_text.strip() == "":
                continue
            label = abstract_text.attrib["Label"]
            nlm_category = abstract_text.attrib["NlmCategory"]
            if nlm_category == "UNASSIGNED":
                continue
            if nlm_category == "METHODS":
                patient_found = False
                for keyword in ["PATIENTS", "PATIENT", "SUBJECTS", "SUBJECT", "PARTICIPANTS", "PARTICIPANT",
                                "POPULATION"]:
                    if keyword in label.upper():
                        patient_found = True
                        article_with_patient = True
                        patient_lines.append(abstract_text.input_abstract_text)
                        break
                if not patient_found:
                    non_patient_lines.append(abstract_text.input_abstract_text)
            # label_num = labels_to_ids[nlm_category]
            # label_text = ids_to_labels[label_num]
            if nlm_category in labels_to_ids:
                section_lines.append(abstract_text.input_abstract_text)
                section_labels.append(nlm_category)
        if article_with_patient:
            participant_lines.extend(patient_lines)
            participant_labels.extend(["PATIENT"] * len(patient_lines))
            participant_lines.extend(non_patient_lines)
            participant_labels.extend(["NON-PATIENT"] * len(non_patient_lines))

    # Sleep for 20 seconds to avoid frequent requests
    if disease != diseases[-1]:
        time.sleep(30)

section_train_lines, section_test_lines, section_train_labels, section_test_labels = train_test_split(
    section_lines, section_labels, test_size=0.2, random_state=42, stratify=section_labels)
participant_train_lines, participant_test_lines, participant_train_labels, participant_test_labels = train_test_split(
    participant_lines, participant_labels, test_size=0.2, random_state=42, stratify=participant_labels)

section_train_df = pd.DataFrame({"line": section_train_lines, "label": section_train_labels})
section_test_df = pd.DataFrame({"line": section_test_lines, "label": section_test_labels})
participant_train_df = pd.DataFrame({"line": participant_train_lines, "label": participant_train_labels})
participant_test_df = pd.DataFrame({"line": participant_test_lines, "label": participant_test_labels})

section_train_df.to_csv(train_section_dataset_output_filepath)
section_test_df.to_csv(test_section_dataset_output_filepath)
# participant_train_df.to_csv(train_participant_dataset_output_filepath)
# participant_test_df.to_csv(test_participant_dataset_output_filepath)
