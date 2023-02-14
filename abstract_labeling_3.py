# -*- coding: utf-8 -*-
import json
import os
import pickle
import re

from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from pymed import PubMed

from dependency_parsing import dependency_parsing
from predict_classifier import init_model_tokenizer, classify_abstract


def remove_stopwords_symbols(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = " ".join([word for word in text.split() if word not in stopwords.words("english")])
    return text


mesh_json_filename = "descriptor_dict_num.json"
with open(mesh_json_filename, "r") as f:
    mesh_num_descriptor_dict = json.load(f)

chemical_drug_descriptors = []

for key, value in mesh_num_descriptor_dict.items():
    if key.startswith("D"):
        chemical_drug_descriptors.append(value.lower())

output_folder_path = "outputs"
table_analysis_result = os.path.join(output_folder_path, "table_analysis_result.pkl")
table_dict_list = pickle.load(open(table_analysis_result, "rb"))
references_path = os.path.join("references.txt")
pubmed = PubMed(email="794194678@qq.com")

with open(references_path, "r") as f:
    references = f.readlines()
c = 1
reference_list = []
for line in references:
    if line.startswith("%s." % c):
        reference = line.split("%s.\t" % c, maxsplit=2)[1]
        reference_list.append(reference)
        c += 1
    else:
        if line.strip() != "" and (not line.startswith("©")):
            reference_list[-1] += line.strip()

model, tokenizer = init_model_tokenizer()

for table_dict in table_dict_list:
    study_details = "\n".join(table_dict["Study details"])
    reference_num = re.findall(r"\^\{([0-9]+)}", study_details)
    if len(reference_num) > 1 or len(reference_num) == 0:
        continue
    reference_num = int(reference_num[0])
    reference = reference_list[reference_num - 1]
    result = pubmed.query(reference, max_results=1)
    article_dict = {}
    for i, article in enumerate(result):
        if i == 0:
            article_dict = article.toDict()

    if article_dict == {}:
        continue

    abstract = article_dict["abstract"]

    if abstract == "":
        continue

    line_label_list = classify_abstract(abstract, model, tokenizer)
    methods_lines = [pair[0] for pair in line_label_list if pair[1] == "METHODS"]

    interventions = "\n".join(table_dict["Interventions"])
    outcome_measures = table_dict["Outcome measures"]
    group_intervention_dict = {}
    intervention_lines = interventions.split("\n")
    cur_group = ""
    cur_group_lines = ""
    for line in intervention_lines:
        line = line.strip()
        if line.startswith("Group"):
            if cur_group != "":
                group_intervention_dict[cur_group] = cur_group_lines.strip().replace("\n", " ")
            cur_group_lines = ""
            group = line.split("Group", maxsplit=1)[1].strip()
            cur_group = group
        else:
            if line.startswith("Examination methods") or line.startswith("All"):
                if cur_group != "":
                    group_intervention_dict[cur_group] = cur_group_lines.strip()
                break
            cur_group_lines += line + " "

    intervention_missed = False

    if group_intervention_dict == {}:
        print("group_intervention_dict is empty")

    for group, intervention in group_intervention_dict.items():
        # intervention = remove_stopwords_symbols(intervention)
        intervention = intervention.lower()
        key_substances = []
        for descriptor in chemical_drug_descriptors:
            ratio = fuzz.partial_ratio(descriptor, intervention)
            if ratio > 95:
                key_substances.append(descriptor)
        print("group: %s" % group)
        if key_substances == "":
            intervention_missed = True
            print("Can't find key substance:\n%s" % intervention)
            print("------------------")
            continue
        key_substances = list(set(key_substances))
        print("intervention: %s" % intervention)
        print("key_substance: %s" % key_substances)
        found_lines = []
        for line in methods_lines:
            substance_hit = False
            for substance in key_substances:
                if substance in line:
                    substance_hit = True
                else:
                    substance_hit = False
                    break
            if substance_hit:
                found_lines.append(line)
        if found_lines == []:
            print("Can't find key substance in methods")
            print("Methods:\n%s" % "\n".join(methods_lines))
            print("------------------")
            continue
        for line in found_lines:
            print(line)
            parsing_result = dependency_parsing(line)
            # print(parsing_result)
            # important_words = []
            # for word in parsing_result:
            #     for substance in key_substances:
            #         if word.text.lower() in substance:
            #             important_words.append(word)
            # for word in important_words:
            #     for word_2 in parsing_result:
            #         if word.head == word_2.id:
            #             important_words.append(word_2)
            # queue = important_words.copy()
            # while len(queue) > 0:
            #     cur_word = queue.pop(0)
            #     for word in parsing_result:
            #         if word.head == cur_word.id:
            #             if word not in important_words:
            #                     queue.append(word)
            #                     important_words.append(word)
            # important_words = sorted(important_words, key=lambda x: x.id)
            # important_words = [word.text for word in important_words]
            # print("important_words: %s" % " ".join(important_words))

    print("------------------")
