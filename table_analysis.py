# -*- coding: utf-8 -*-
import os
import pandas as pd
import re
from pymed import PubMed
import difflib
import nltk
from nltk.corpus import stopwords
import pickle

pubmed = PubMed(email="794194678@qq.com")

output_folder_path = "outputs"
extracted_table_path = os.path.join(output_folder_path, "Book1.xlsx")
table_analysis_result = os.path.join(output_folder_path, "table_analysis_result.pkl")
references_path = os.path.join("references.txt")

extracted_tables_dict = pd.read_excel(extracted_table_path, sheet_name=None)
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

table_dict_list = []

for sheet_name, table in extracted_tables_dict.items():
    if "Column1" not in table.columns:
        continue
    if "Study\ndetails" != table["Column1"][0]:
        if len(table["Column1"]) < 2 or "Study" != table["Column1"][0] or "details" != table["Column1"][1]:
            continue
    col_num = len(table.columns)
    table = table.values
    table_dict = {}
    for i in range(col_num):
        column = table[:, i]
        column = [str(cell).strip() for cell in column]
        column = [cell for cell in column if cell != "nan"]
        if column[0] == "Study" and column[1] == "details":
            column[0] = "Study details"
            column[1] = ""
        column[0] = column[0].replace("\n", " ")
        if "\uf0b7" == column[0]:
            continue
        table_dict[column[0]] = column[1:]
    table_dict_list.append(table_dict)

pickle.dump(table_dict_list, open(table_analysis_result, "wb"))



    # study_details = table["study details"][0]
    # participants = table["participants"][0]
    # interventions = table["interventions"][0]
    # outcome_measures = table["outcome measures"][0]
    # effect_size = table["effect size"][0]
    # quote_num = re.search(r'\d+', study_details).group()[4:]
    # quote_num = int(quote_num)
    # reference = reference_list[quote_num - 1]
    # result = pubmed.query(reference, max_results=1)
    # article_dict = {}
    # for i, article in enumerate(result):
    #     if i == 0:
    #         article_dict = article.toDict()
    #
    # if article_dict == {}:
    #     continue
    #
    # abstract = article_dict["abstract"]
    # print(abstract)
    # print("=====================================")
    # abstract_lines = abstract.lower().replace(". ", ".\n").split("\n")
    # abstract_lines = [line.strip() for line in abstract_lines]
    # for category_text in [participants, interventions, outcome_measures, effect_size]:
    #     category_text = category_text.replace("\n", " ").lower()
    #     # most_similar_line = difflib.get_close_matches(category, abstract_lines, n=1, cutoff=0)
    #     category_text_words = category_text.split()
    #     category_text_words = [word for word in category_text_words if word not in stopwords.words("english")]
    #     similar_lines = []
    #     for line in abstract_lines:
    #         line_words = line.split()
    #         line_words = [word for word in line_words if word not in stopwords.words("english")]
    #         score = 0
    #         for line_word in line_words:
    #             if line_word in category_text_words:
    #                 score += 1
    #         score /= len(line_words)
    #         if score > 0.3:
    #             similar_lines.append(line)
    #     print("Category text:\n", category_text)
    #     print("Most similar line:\n", similar_lines)
    #     print()
    #
    # break

    # group_intervention_dict = {}
    # intervention_lines = interventions.split("\n")
    # cur_group = ""
    # cur_group_lines = ""
    # for line in intervention_lines:
    #     line = line.strip()
    #     if line.startswith("Group"):
    #         if cur_group != "":
    #             group_intervention_dict[cur_group] = cur_group_lines.strip().replace("\n", " ")
    #         cur_group_lines = ""
    #         group = line.split("Group", maxsplit=1)[1].strip()
    #         cur_group = group
    #     else:
    #         if line.startswith("Examination methods") or line.startswith("All"):
    #             if cur_group != "":
    #                 group_intervention_dict[cur_group] = cur_group_lines.strip()
    #             break
    #         cur_group_lines += line + " "
    # abstract_lines = abstract.replace(". ", ".\n").replace(", ", ",\n").split("\n")
    # for group, intervention in group_intervention_dict.items():
    #     most_similar_line = difflib.get_close_matches(intervention, abstract_lines, n=10, cutoff=0.3)
    #     print(intervention)
    #     print(most_similar_line)
