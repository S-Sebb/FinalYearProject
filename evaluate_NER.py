# -*- coding: utf-8 -*-
import os

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from predict import predict_classification, predict_NER
from utils import *
import json

if __name__ == "__main__":
    # Custom parameters
    evaluation_dataset_path = os.path.join("datasets", "NER datasets",
                                           "test_NER_dataset.json")
    model_type = "BERT"
    model_filename = "BERT_NER_model"
    tokenizer_filename = "BERT_NER_tokenizer"
    model_dir_filepath = os.path.join("models", "NER models")
    labels_to_ids = {"O": 0, "INTV": 1, "OC": 2, "MEAS": 3}
    ids_to_labels = {v: k for k, v in labels_to_ids.items()}

    # Start of the script
    model_filepath = os.path.join(model_dir_filepath, model_filename)
    tokenizer_filepath = os.path.join(model_dir_filepath, tokenizer_filename)

    with open(evaluation_dataset_path, "r") as f:
        dataset = json.load(f)
    token_texts_list = dataset["token_texts_list"]
    labels_list = dataset["labels_list"]
    label_ids_list = [[labels_to_ids[label] for label in labels] for labels in labels_list]
    gt_label_ids = [label_id for label_ids in label_ids_list for label_id in label_ids]
    gt_labels = [ids_to_labels[label_id] for label_id in gt_label_ids]

    # load the model and tokenizer
    model, tokenizer = load_fine_tuned_NER_model_tokenizer(model_filepath, tokenizer_filepath, model_type)

    pred_label_ids = predict_NER(token_texts_list, model, tokenizer)
    pred_label_ids = [label_id for label_ids in pred_label_ids for label_id in label_ids]
    pred_labels = [ids_to_labels[label_id] for label_id in pred_label_ids]

    report = classification_report(gt_labels, pred_labels)
    confusion_matrix = confusion_matrix(gt_labels, pred_labels)
    print(report)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels_to_ids.keys())
    disp.plot()
    plt.show()
