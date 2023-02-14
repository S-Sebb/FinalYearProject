# -*- coding: utf-8 -*-
import json
import os

import pandas as pd
import spacy

input_filepath = os.path.join("annotated_NER", "METHODS.jsonl")
output_filepath = os.path.join("annotated_NER", "METHODS.csv")

nlp = spacy.blank("en")

text_list = []
labels_list = []


def jsonl_to_IOB(text, annotations):
    """
    Convert a text and its annotations to IOB format.
    """
    doc = nlp.make_doc(text)
    doc.ents = [doc.char_span(annotation[0], annotation[1], annotation[2]) for annotation in annotations]
    tags = []
    tokens = []
    for t in doc:
        tokens.append(t.text)
        tags.append(f"{t.ent_iob_}-{t.ent_type_}" if t.ent_iob_ else "O")
    return tokens, tags


# with open(input_filepath, "r", encoding="utf-8") as f:
#     for line in f:
#         json_data = json.loads(line)
#         text = json_data["text"]
#         label = json_data["label"]
#         tokenized_text, iob_tags = jsonl_to_IOB(text, label)
#         text_list.append(text)
#         labels_list.append(iob_tags)
#
# df = pd.DataFrame({"text": text_list, "labels": labels_list})
# df.to_csv(output_filepath, encoding="utf-8", index=False)
