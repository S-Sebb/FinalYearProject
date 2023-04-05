# -*- coding: utf-8 -*-
import os

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from predict import predict_classification
from utils import *

if __name__ == "__main__":
    # Custom parameters
    evaluation_dataset_path = os.path.join("datasets", "section classifier datasets",
                                           "test_4Section_classifier_dataset.csv")
    model_filename = "BERT_4Section_CustomWeightsTrue_classifier_model"
    tokenizer_filename = "BERT_section_classifier_tokenizer"
    model_dir_filepath = os.path.join("models", "section classifier models")
    labels_to_ids = {"BACKGROUND&OBJECTIVE": 0, "METHODS": 1, "RESULTS": 2, "CONCLUSIONS": 3}
    labels_to_ids = {"BACKGROUND": 0, "OBJECTIVE": 1, "METHODS": 2, "RESULTS": 3, "CONCLUSIONS": 4}
    labels_to_ids = {"NON-PATIENT": 0, "PATIENT": 1}
    ids_to_labels = {v: k for k, v in labels_to_ids.items()}

    # Start of the script
    model_filepath = os.path.join(model_dir_filepath, model_filename)
    tokenizer_filepath = os.path.join(model_dir_filepath, tokenizer_filename)

    df = pd.read_csv(evaluation_dataset_path, index_col=0)
    lines = df.line.values
    gt_labels = df.label.values

    # load the model and tokenizer
    model, tokenizer = load_fine_tuned_classification_model_tokenizer(model_filepath, tokenizer_filepath)

    pred_labels = predict_classification(lines, ids_to_labels, model, tokenizer)

    report = classification_report(gt_labels, pred_labels)
    confusion_matrix = confusion_matrix(gt_labels, pred_labels)
    print(report)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels_to_ids.keys())
    disp.plot()
    plt.show()
