# -*- coding: utf-8 -*-
import os
import pickle
import re

import pandas as pd
import yake
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from pymed import PubMed


def remove_stopwords_symbols(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = " ".join([word for word in text.split() if word not in stopwords.words("english")])
    return text


output_folder_path = "outputs"
table_analysis_result = os.path.join(output_folder_path, "table_analysis_result.pkl")
table_dict_list = pickle.load(open(table_analysis_result, "rb"))
references_path = os.path.join("references.txt")
pubmed = PubMed(email="794194678@qq.com")
language = "en"
max_ngram_size = 3
deduplication_threshold = 0
numOfKeywords = 2
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold,
                                            top=numOfKeywords, features=None)
all_lines = []
all_labels = []
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

    abstract_lines = abstract.lower().replace(". ", ".\n").split("\n")
    abstract_lines = [line.strip() for line in abstract_lines]

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

    line_labels = [0 for _ in range(len(abstract_lines))]

    for group, intervention in group_intervention_dict.items():
        # print(intervention)
        # keywords = custom_kw_extractor.extract_keywords(intervention)
        # for kw in keywords:
        #     print(kw)
        # for line in abstract_lines:
        #     total_ratio = 0
        #     intervention_words = remove_stopwords_symbols(intervention).split()
        #     for intervention_word in intervention_words:
        #         line_words = remove_stopwords_symbols(line).split()
        #         for line_word in line_words:
        #             ratio = fuzz.ratio(intervention_word, line_word)
        #             if ratio > 80:
        #                 total_ratio += ratio
        #     if total_ratio / 100 / len(intervention_words) > 0.3:
        #         print(intervention)
        #         print(line)
        #         print(total_ratio / 100 / len(intervention_words))
        #         print("-----------------")
        intervention = remove_stopwords_symbols(intervention)
        keywords = custom_kw_extractor.extract_keywords(intervention)
        for i, line in enumerate(abstract_lines):
            line = remove_stopwords_symbols(line)
            partial_ratio = fuzz.partial_ratio(intervention, line)
            keyword_hit = True
            for keyword in keywords:
                if fuzz.partial_ratio(keyword[0], line) > 90:
                    keyword_hit = True
                else:
                    keyword_hit = False
                    break
            if partial_ratio > 50 or keyword_hit:
                line_labels[i] = 1
    # for i, line in enumerate(abstract_lines):
    #     print(line)
    #     if line_labels[i] == 1:
    #         print("Hit: intervention")
    #     else:
    #         print("No hit!")
    #     print("-----------------")
    all_lines += abstract_lines
    all_labels += line_labels

line_label_dict = {}
for line, line_label in zip(all_lines, all_labels):
    line_label_dict[line] = line_label
df = pd.DataFrame(line_label_dict.items(), columns=["line", "label"])
df.to_csv("abstract_labeling.csv")

    # for outcome_measure in outcome_measures:
    #     outcome_measure = remove_stopwords_symbols(outcome_measure)
    #     keywords = custom_kw_extractor.extract_keywords(outcome_measure)
    #     for line in abstract_lines:
    #         line = remove_stopwords_symbols(line)
    #         print(line)
    #         partial_ratio = fuzz.partial_ratio(outcome_measure, line)
    #         keyword_hit = False
    #         for keyword in keywords:
    #             if fuzz.partial_ratio(keyword[0], line) > 90:
    #                 keyword_hit = True
    #                 break
    #         if partial_ratio > 50 or keyword_hit:
    #             print("Hit!")
    #             print(outcome_measure)
    #             print(partial_ratio)
    #         else:
    #             print("Not hit")
    #         print("-----------------")
    # intervention_lines = []
    # for group, intervention in group_intervention_dict.items():
    #     intervention_simplified = remove_stopwords_symbols(intervention)
    #     print(intervention_simplified)
    #     intervention_words = intervention_simplified.split()
    #     for line in abstract_lines:
    #         intervention_score = 0
    #         line_words = remove_stopwords_symbols(line).split()
    #         hit_words = []
    #         for word in intervention_words:
    #             if word in line_words:
    #                 hit_words.append(word)
    #                 intervention_score += 1
    #         if intervention_score / len(intervention_words) > 0.5:
    #             print(intervention_words)
    #             print(hit_words)
    #             print(line)
    #             print("-----------------")

    # print(intervention_lines)
    # break
