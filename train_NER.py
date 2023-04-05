# -*- coding: utf-8 -*-
import os

from train import train_NER

if __name__ == "__main__":
    # Custom parameters
    model_type = "BERT"  # "BERT", "RoBERTa" or "BioBERT"
    labels_to_ids = {"O": 0, "INTV": 1, "OC": 2, "MEAS": 3}
    epochs = 30
    learning_rate = 5e-5
    weight_decay = 1e-8

    NER_dataset_filepath = os.path.join("datasets", "NER datasets", "train_NER_dataset.json")
    output_model_filename = "%s_NER_model" % model_type
    output_tokenizer_filename = "%s_NER_tokenizer" % model_type
    output_dir_filepath = os.path.join("models", "NER models")

    train_NER(output_dir_filepath, output_model_filename, output_tokenizer_filename, NER_dataset_filepath,
              labels_to_ids, epochs, learning_rate, weight_decay, model_type)
