# -*- coding: utf-8 -*-
import json
import os

from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from predict import predict_NER
from utils import *

if __name__ == "__main__":
    # Custom parameters
    evaluation_dataset_path = os.path.join("datasets", "NER datasets",
                                           "test_NER_dataset.json")
    custom_weights_flag = True
    model_type = "RoBERTa"
    model_filename = "%s_CustomWeight%s_NER_model" % (model_type, custom_weights_flag)
    tokenizer_filename = "%s_NER_tokenizer" % model_type
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
    # Flatten the list of lists
    gt_label_ids = [label_id for label_ids in label_ids_list for label_id in label_ids]
    # Convert the label ids to labels
    gt_labels = [ids_to_labels[label_id] for label_id in gt_label_ids]

    # load the model and tokenizer
    model, tokenizer = load_fine_tuned_NER_model_tokenizer(model_filepath, tokenizer_filepath, model_type)

    pred_labels = predict_NER(token_texts_list, ids_to_labels, model, tokenizer)
    pred_labels_flattened = [label for labels in pred_labels for label in labels]

    print(model_filename)
    report = classification_report(gt_labels, pred_labels_flattened)
    confusion_matrix = confusion_matrix(gt_labels, pred_labels_flattened)
    print(report)
    display_labels = ["INTV", "MEAS", "O", "OC"]
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=display_labels)
    disp.plot()
    plt.show()
