# -*- coding: utf-8 -*-
import os

from train import train_classifier
from utils import root_dir

if __name__ == "__main__":
    # Custom parameters
    labels_to_ids = {"NON-PATIENT": 0, "PATIENT": 1}
    epochs = 30
    learning_rate = 5e-5
    weight_decay = 1e-8
    custom_weights_flag = True

    dataset_filepath = os.path.join(root_dir, "datasets", "participant classifier datasets",
                                    "train_participant_classifier_dataset.csv")
    output_model_filename = "BERT_CustomWeights%s_participant_classifier_model" % custom_weights_flag
    output_tokenizer_filename = "BERT_participant_classifier_tokenizer"
    output_dir_filepath = os.path.join(root_dir, "models", "participant classifier models")

    train_classifier(output_dir_filepath, output_model_filename, output_tokenizer_filename, dataset_filepath,
                     labels_to_ids, epochs, learning_rate, weight_decay, custom_weights_flag)
