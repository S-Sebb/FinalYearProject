# -*- coding: utf-8 -*-
import pandas as pd  # https://pandas.pydata.org/
from matplotlib import pyplot as plt  # https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.html
from sklearn.metrics import classification_report, confusion_matrix, \
    ConfusionMatrixDisplay  # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

from predict import predict_classification
from utils import *

if __name__ == "__main__":
    # Custom parameters
    section_number = 4
    evaluation_dataset_path = os.path.join("datasets", "section classifier datasets",
                                           "test_%sSection_classifier_dataset.csv" % section_number)
    model_filename = "BERT_%sSection_CustomWeightsTrue_classifier_model" % section_number
    tokenizer_filename = "BERT_section_classifier_tokenizer"
    model_dir_filepath = os.path.join("models", "section classifier models")
    if section_number == 4:
        labels_to_ids = {"BACKGROUND&OBJECTIVE": 0, "METHODS": 1, "RESULTS": 2, "CONCLUSIONS": 3}
    else:
        labels_to_ids = {"BACKGROUND": 0, "OBJECTIVE": 1, "METHODS": 2, "RESULTS": 3, "CONCLUSIONS": 4}
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
    if section_number == 4:
        display_labels = ["BACKGROUND&OBJECTIVE", "CONCLUSIONS", "RESULTS", "METHODS"]
    else:
        display_labels = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                  display_labels=display_labels)
    disp.plot()
    plt.show()
