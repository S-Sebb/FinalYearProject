# -*- coding: utf-8 -*-
from utils import *
import os
from train import train_classifier

if __name__ == "__main__":
    # Custom parameters
    labels_to_ids = {"BACKGROUND": 0, "OBJECTIVE": 1, "METHODS": 2, "RESULTS": 3, "CONCLUSIONS": 4}
    epochs = 30
    learning_rate = 5e-5
    weight_decay = 1e-8
    custom_weights_flag = False

    dataset_filepath = os.path.join("datasets", "section classifier datasets",
                                    "train_5Section_classifier_dataset.csv")
    output_model_filename = "BERT_5Section_CustomWeights%s_classifier_model" % custom_weights_flag
    output_tokenizer_filename = "BERT_section_classifier_tokenizer"
    output_dir_filepath = os.path.join("models", "section classifier models")

    train_classifier(output_dir_filepath, output_model_filename, output_tokenizer_filename, dataset_filepath,
                     labels_to_ids, epochs, learning_rate, weight_decay, custom_weights_flag)
