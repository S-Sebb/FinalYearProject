# -*- coding: utf-8 -*-
import os

from train import train_NER
from utils import root_dir

if __name__ == "__main__":
    # Custom parameters
    model_type = "RoBERTa"  # "BERT", "RoBERTa" or "BioBERT"
    labels_to_ids = {"O": 0, "INTV": 1, "OC": 2, "MEAS": 3}
    epochs = 50
    learning_rate = 5e-5
    weight_decay = 1e-8
    custom_weights_flag = True

    NER_dataset_filepath = os.path.join(root_dir, "datasets", "NER datasets", "train_NER_dataset.json")
    output_model_filename = "%s_CustomWeight%s_NER_model" % (model_type, custom_weights_flag)
    output_tokenizer_filename = "%s_NER_tokenizer" % model_type
    output_dir_filepath = os.path.join(root_dir, "models", "NER models")

    train_NER(output_dir_filepath, output_model_filename, output_tokenizer_filename, NER_dataset_filepath,
              labels_to_ids, epochs, learning_rate, weight_decay, model_type, custom_weights_flag)
