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

import pynvml

def get_gpu_utilization(gpu_index=0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    pynvml.nvmlShutdown()
    return utilization.gpu  # 返回 GPU 利用率百分比


def main(
    config: dict, 
):
    # print(device)
    dataset_for_training = training_create_dataset(config['Training_set_config'])
    # dataset_for_testing = testing_create_dataset(config['Testing_set_config'])

    # Start for training
    # dataset_for_training[len(dataset_for_training)] = dataset_for_testing
    print(dataset_for_training.keys())
    node_list = Series_training(training_set_ae=copy.deepcopy(dataset_for_training), config_dict=config['Filtering_Chain'])
    passed_dataset = Series_passing(node_list, config_dict=config['Filtering_Chain'], config_dict_for_passing=config['Testing_set_config'])
    print(passed_dataset.shape)

    dataset_for_training[len(dataset_for_training)] = passed_dataset
    for key in dataset_for_training.keys():
        print(key)
        print(dataset_for_training[key].shape)
        
    trainSeriesSupC_struct(dataset_for_training, config_dict=config['Weakly_Supervised'])

    
    return 0
    
if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    os.chdir(script_directory)

    # Loading data_config.yaml
    with open("./pipeline_config.yaml", "r") as file:
        full_config = yaml.safe_load(file)
        
    # Define the device
    # device = full_config['Full_pipeline']['Training_scheme']['device']
    # full_config['Filtering_Chain']['Output_dir'] = full_config['Full_pipeline']['Training_scheme']['Output_dir']
    # full_config['Weakly_Supervised']['Training_scheme']['Output_dir'] = full_config['Full_pipeline']['Training_scheme']['Output_dir']
    # main(full_config)

    max_models = 10
    trained_models = 0
    processes = []

    while trained_models < max_models:
        gpu_utilization = get_gpu_utilization()

        if gpu_utilization < 80:
            config_cached = copy.deepcopy(full_config)
            config_cached['Filtering_Chain']['Output_dir'] = config_cached['Full_pipeline']['Training_scheme']['Output_dir'] + f'/output_{trained_models}'
            config_cached['Weakly_Supervised']['Training_scheme']['Output_dir'] = config_cached['Full_pipeline']['Training_scheme']['Output_dir'] + f'/output_{trained_models}'
            p = multiprocessing.Process(target=main, args=(config_cached,))
            p.start()
            processes.append(p)
            trained_models += 1
            print(f"Started training model {trained_models}. GPU utilization: {gpu_utilization}%")

        time.sleep(30)

    for p in processes:
        p.join()

    print("All models have been trained.")
    