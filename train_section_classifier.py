# -*- coding: utf-8 -*-
from utils import *
import os
from train import train_classifier

if __name__ == "__main__":
    # Custom parameters
    section_number = 4
    epochs = 50
    learning_rate = 5e-5
    weight_decay = 1e-8
    custom_weights_flag = True

    if section_number == 4:
        labels_to_ids = {"BACKGROUND&OBJECTIVE": 0, "METHODS": 1, "RESULTS": 2, "CONCLUSIONS": 3}
    else:
        labels_to_ids = {"BACKGROUND": 0, "OBJECTIVE": 1, "METHODS": 2, "RESULTS": 3, "CONCLUSIONS": 4}

    dataset_filepath = os.path.join("datasets", "section classifier datasets",
                                    "train_%sSection_classifier_dataset.csv" % section_number)
    output_model_filename = "BERT_%sSection_CustomWeights%s_classifier_model" % (section_number, custom_weights_flag)
    output_tokenizer_filename = "BERT_section_classifier_tokenizer"
    output_dir_filepath = os.path.join("models", "section classifier models")

    train_classifier(output_dir_filepath, output_model_filename, output_tokenizer_filename, dataset_filepath,
                     labels_to_ids, epochs, learning_rate, weight_decay, custom_weights_flag)
