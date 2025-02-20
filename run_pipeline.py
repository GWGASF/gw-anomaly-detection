#!/bin/python

# if len(sys.argv) < 4 or len(sys.argv) > 5:
#     print("Usage: python script.py <device> <cnt_min> <cnt_max> <(suffix_for_output_file)>")
#     sys.exit(1)
# elif len(sys.argv) == 5:
#     suffix = str(sys.argv[4])
#     print("Suffix set to be {}".format(suffix))
    
# device_pick = int(sys.argv[1])
# cnt_min = int(sys.argv[2])
# cnt_max = int(sys.argv[3])


# # create logger
# logger = logging.getLogger('simple_example')
# logging.basicConfig(filename='../Log/cutscan_cntrange_{}-{}{}.log'.format(cnt_min, cnt_max, suffix), encoding='utf-8')
# logger.setLevel(logging.INFO)

# # create console handler and set level to debug
# ch = logging.StreamHandler()
# ch.setLevel(logging.INFO)

# # create formatter
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# # add formatter to ch
# ch.setFormatter(formatter)

# # add ch to logger
# logger.addHandler(ch)

# logger.info("Start to train the model with cnt in range {}-{}".format(cnt_min, cnt_max))


# # torch.cuda.set_device(0)

# device = torch.device("cuda:{}".format(int(device_pick-1)) if device_pick else "cpu")
# logger.info(f"Using device: {device}")


# dataDir = "../Data"
# outputDir = "../Output"
# modelDir = "../Model"

import yaml
import os
import multiprocessing
import argparse
import copy
from gw_anomaly_detection.event_preprocessing.dataset_preprocess import training_create_dataset
from gw_anomaly_detection.event_preprocessing.dataset_preprocess import testing_create_dataset
from gw_anomaly_detection.Training_pipeline.filtering_chain import Series_training
from gw_anomaly_detection.Training_pipeline.weakly_supervised_classifier import trainSeriesSupC_struct

def main(
    config: dict, 
):
    # print(device)
    dataset_for_training = training_create_dataset(config['Training_set_config'])
    dataset_for_testing = testing_create_dataset(config['Testing_set_config'])

    # Start for training
    dataset_for_training[len(dataset_for_training)] = dataset_for_testing
    print(dataset_for_training.keys())
    dataset_for_training[len(dataset_for_training)-1] = Series_training(training_set_ae=dataset_for_training, config_dict=config['Filtering_Chain'])
    model = trainSeriesSupC_struct(dataset_for_training, config_dict=config['Weakly_Supervised'])

    return model
    
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
    