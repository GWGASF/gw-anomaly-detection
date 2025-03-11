#!/bin/python
import yaml
import os
import multiprocessing
import argparse
import copy
from gw_anomaly_detection.event_preprocessing.dataset_preprocess import training_create_dataset
from gw_anomaly_detection.event_preprocessing.dataset_preprocess import testing_create_dataset
from gw_anomaly_detection.Training_pipeline.filtering_chain import Series_training
from gw_anomaly_detection.Training_pipeline.filtering_chain import Series_passing
from gw_anomaly_detection.Training_pipeline.weakly_supervised_classifier import trainSeriesSupC_struct

def main(
    config: dict, 
):
    # print(device)
    dataset_for_training = training_create_dataset(config['Training_set_config'])
    # dataset_for_testing = testing_create_dataset(config['Testing_set_config'])

    # Start for training
    # dataset_for_training[len(dataset_for_training)] = dataset_for_testing
    print(dataset_for_training.keys())
    node_list = Series_training(training_set_ae=dataset_for_training, config_dict=config['Filtering_Chain'])
    passed_dataset = Series_passing(node_list, config_dict=config['Filtering_Chain'], config_dict_for_passing=config['Testing_set_config'])

    # Remember to make copy for the training dataset
    
    return 0
    
if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    os.chdir(script_directory)

    # Loading data_config.yaml
    with open("./pipeline_config.yaml", "r") as file:
        full_config = yaml.safe_load(file)
        
    # Define the device
    device = full_config['Full_pipeline']['Training_scheme']['device']
        
    main(full_config)
    