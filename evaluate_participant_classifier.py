# -*- coding: utf-8 -*-
import os

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from predict import predict_classification
from utils import *

if __name__ == "__main__":
    # Custom parameters
    evaluation_dataset_path = os.path.join("datasets", "participant classifier datasets",
                                           "test_participant_classifier_dataset.csv")
    custom_weights_flag = True
    model_filename = "BERT_CustomWeights%s_participant_classifier_model" % custom_weights_flag
    tokenizer_filename = "BERT_participant_classifier_tokenizer"
    model_dir_filepath = os.path.join("models", "participant classifier models")
    labels_to_ids = {"NON-PATIENT": 0, "PATIENT": 1}
    ids_to_labels = {v: k for k, v in labels_to_ids.items()}

    # Start of the script
    model_filepath = os.path.join(model_dir_filepath, model_filename)
    tokenizer_filepath = os.path.join(model_dir_filepath, tokenizer_filename)

    df = pd.read_csv(evaluation_dataset_path, index_col=0)
    lines = df.line.values
    gt_labels = df.label.values

    # Calculate the proportion of each label in the dataset
    label_counts = df.label.value_counts()
    label_counts = label_counts / label_counts.sum()
    print(label_counts)

    # load the model and tokenizer
    model, tokenizer = load_fine_tuned_classification_model_tokenizer(model_filepath, tokenizer_filepath)

    pred_labels = predict_classification(lines, ids_to_labels, model, tokenizer)

    report = classification_report(gt_labels, pred_labels)
    confusion_matrix = confusion_matrix(gt_labels, pred_labels)
    print(model_filename)
    print(report)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                  display_labels=["PARTICIPANT", "NON-PARTICIPANT"])
    disp.plot()
    plt.show()
