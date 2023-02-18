# -*- coding: utf-8 -*-
import json
import os
from xml.dom import minidom
from xml.etree import ElementTree

import nltk
import pandas as pd
from pymed import PubMed

#nltk.download('punkt')

pubmed = PubMed(email="794194678@qq.com")
result = pubmed.query("(Randomized Controlled Trial[Publication Type]) AND (glaucoma)", max_results=10000)
nlm_category_convert_dict = {"BACKGROUND": 0, "OBJECTIVE": 0, "METHODS": 1, "RESULTS": 2, "CONCLUSIONS": 3}
reverse_nlm_category_convert_dict = {0: "OBJECTIVE&BACKGROUND", 1: "METHODS", 2: "RESULTS", 3: "CONCLUSIONS"}
all_lines = []
all_label_texts = []
all_nlm_categories = []
all_label_nums = []

all_label_num_line_dict = {k: [] for k, v in reverse_nlm_category_convert_dict.items()}
all_label_text_line_dict = {v: [] for k, v in reverse_nlm_category_convert_dict.items()}

for i, article in enumerate(result):
    label_text_line_dict = {v: [] for k, v in reverse_nlm_category_convert_dict.items()}

    xml_root = article.toDict()["xml"]
    xml_str = ElementTree.tostring(xml_root)
    xml_str = minidom.parseString(xml_str).toprettyxml(indent="   ")
    # print(xml_str)
    medline_citation = xml_root.find("MedlineCitation")
    article = medline_citation.find("Article")
    title = article.find("ArticleTitle").text
    # print(title)
    abstract = article.find("Abstract")
    if abstract is None:
        continue
    intervention = False
    for abstract_text in abstract.findall("AbstractText"):
        if abstract_text.attrib.get("Label") == "INTERVENTION" or abstract_text.attrib.get("Label") == "INTERVENTIONS":
            intervention = True
            break
    if not intervention:
        continue
    for abstract_text in abstract.findall("AbstractText"):
        if "NlmCategory" not in abstract_text.attrib:
            continue
        if abstract_text.text is None or abstract_text.text.strip() == "":
            continue
        # label = abstract_text.attrib["Label"]
        nlm_category = abstract_text.attrib["NlmCategory"]
        label_num = nlm_category_convert_dict[nlm_category]
        label_text = reverse_nlm_category_convert_dict[label_num]
        if nlm_category in nlm_category_convert_dict:
            all_label_num_line_dict[label_num].append(abstract_text.text)
            label_text_line_dict[label_text].append(abstract_text.text)
    for k, v in label_text_line_dict.items():
        if len(v) > 0:
            all_label_text_line_dict[k].append("\n".join(v))

line_label_num_dict = {}
for k, v in all_label_num_line_dict.items():
    for line in v:
        line_label_num_dict[line] = k

df = pd.DataFrame(line_label_num_dict.items(), columns=["line", "label"])
df.to_csv("classifier_dataset.csv", encoding="utf-8")

# for k, v in all_label_text_line_dict.items():
#     with open(os.path.join("outputs", "%s.jsonl" % k), "w", encoding="utf-8") as f:
#         for line in v:
#             f.write(json.dumps({"text": line, "label": []}) + "\n")
