# -*- coding: utf-8 -*-
import json
import os

import simplejson  # https://simplejson.readthedocs.io/en/latest/
from sklearn.model_selection import \
    train_test_split  # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

from utils import root_dir

if __name__ == "__main__":
    input_filepath = os.path.join(root_dir, "datasets", "NER datasets", "RCT_ART_NER.jsonl")
    train_dataset_output_filepath = os.path.join(root_dir, "datasets", "NER datasets", "train_NER_dataset.json")
    test_dataset_output_filepath = os.path.join(root_dir, "datasets", "NER datasets", "test_NER_dataset.json")

    token_texts_list = []
    labels_list = []
    label_counts = {}

    with open(input_filepath, "r", encoding="utf-8") as f:
        for line in f:
            json_data = json.loads(line)
            text = json_data["text"]
            tokens = json_data["tokens"]
            spans = json_data["spans"]

            token_texts = [token["text"] for token in tokens]
            labels = ["O"] * len(token_texts)

            for span in spans:
                token_start = span["token_start"]
                token_end = span["token_end"]
                label = span["label"]
                for i in range(token_start, token_end + 1):
                    labels[i] = label

            for label in labels:
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1
            token_texts_list.append(token_texts)
            labels_list.append(labels)

    train_token_texts_list, test_token_texts_list, train_labels_list, test_labels_list = train_test_split(
        token_texts_list, labels_list, test_size=0.2, random_state=42)

    # Pretty save the datasets
    with open(train_dataset_output_filepath, "w", encoding="utf-8") as f:
        f.write(
            simplejson.dumps({"token_texts_list": train_token_texts_list, "labels_list": train_labels_list}, indent=4))

    with open(test_dataset_output_filepath, "w", encoding="utf-8") as f:
        f.write(
            simplejson.dumps({"token_texts_list": test_token_texts_list, "labels_list": test_labels_list}, indent=4))
