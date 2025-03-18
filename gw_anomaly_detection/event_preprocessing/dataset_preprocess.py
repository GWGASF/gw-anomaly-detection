import numpy as np
import os
import torch
import h5py
from copy import deepcopy
import yaml


def read_from_files(file_path, detectors):
    injected_waveforms = np.concatenate([np.array(h5py.File(file_path, 'r')[detector])[:, np.newaxis, :] for detector in detectors], axis = 1)
    # raw_waveforms = np.concatenate((np.array(h5py.File(file_path, 'r')['waveform_H1'])[:, np.newaxis, :], np.array(h5py.File(file_path, 'r')['waveform_L1'])[:, np.newaxis, :]), axis = 1)
    # inject_params = np.array(h5py.File(file_path, 'r')['waveform_parameters'])
    
    # return injected_waveforms, raw_waveforms, inject_params
    return injected_waveforms

def read_from_directory(directory_path, detectors):
    all_injected_waveforms = []
    # all_raw_waveforms = []
    # all_inject_params = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".hdf5"): 
            file_path = os.path.join(directory_path, filename)
            print(f"Processing file: {file_path}")

            # injected_waveforms, raw_waveforms, inject_params = read_from_files(file_path)
            injected_waveforms= read_from_files(file_path, detectors)

            all_injected_waveforms.append(injected_waveforms)
            # all_raw_waveforms.append(raw_waveforms)
            # all_inject_params.append(inject_params)
    
    all_injected_waveforms = np.concatenate(all_injected_waveforms, axis=0)
    # all_raw_waveforms = np.concatenate(all_raw_waveforms, axis=0)
    # all_inject_params = np.concatenate(all_inject_params, axis=0)

    return all_injected_waveforms, 0,0


def cut_events_from_waveforms(cutting_config):
    
    EVENT_PER_SEGMENT = cutting_config['General_config']['Event_per_segment']
    segment_freq = cutting_config['General_config']['Segment_frequency']
    segment_length = cutting_config['General_config']['Segment_length']
    file_path = cutting_config['Dataset_path']
    cutting_type = cutting_config['Cutting_type']
    detectors = cutting_config['Detectors']
    
    waveforms, _, _ = read_from_directory(file_path, detectors)
    # I have no ideas of how to use the SNR right now
    
    if cutting_type == 'window_cut':
        cutting_window_left = cutting_config['Cutting_window'][0]
        cutting_window_right = cutting_config['Cutting_window'][1]
    
        cutted_events = np.empty((len(waveforms),EVENT_PER_SEGMENT,len(detectors),200))
        midp = (waveforms[0].shape)[-1]//2
        
        
        for i in range(len(waveforms)):
        # for i in range(1):
            for j in range(EVENT_PER_SEGMENT):
                starting_time = int(np.random.uniform(cutting_window_left * segment_freq, cutting_window_right * segment_freq - 200))
                ending_time = starting_time + 200
                
                cache = waveforms[i][:,midp + starting_time:midp + ending_time]
            
                cutted_events[i,j] = cache.copy()
    
    elif cutting_type == 'full_scan':
        cutted_events = np.empty((len(waveforms),segment_length-199,len(detectors),200))
        # midp = (waveforms[0].shape)[-1]//2
        
        
        for i in range(len(waveforms)):
        # for i in range(1):
            for j in range(segment_length-199):
                starting_time = j
                ending_time = starting_time + 200
                
                cache = waveforms[i][:,starting_time:ending_time]
            
                cutted_events[i,j] = cache.copy()
    
    return cutted_events.reshape(-1,len(detectors),200)

    
def make_training_separated_and_normalized_datasets(config_dict):

    # dataDir = "../../../../Data_cached"
    # list_dataset = ['glitch_L', 'glitch_H', 'noise_L', 'noise_H']
    dataset = {}
    dataset_fft = {}

    # dataset['glitch_L'] = np.load(dataDir+"/real_glitches_snrlt5_60132_4000Hz_25ms.npz")["strain_time_data"][:12500];
    # dataset['glitch_H'] = np.load(dataDir+"/real_glitches_H_snrlt5_59732_4000Hz_25ms.npz")["strain_time_data"][:12500];
    # dataset['noise_L'] = np.concatenate((np.load(dataDir+'/Noise_processing/Processed_noise_sets/noise_sets_v1.npy'), np.load('E://GWNMMAD_data/Tw_dataset/Datasets/background.npz')['data']), axis = 0)[:187500,1,:]
    # dataset['noise_H'] = np.concatenate((np.load(dataDir+'/Noise_processing/Processed_noise_sets/noise_sets_v1.npy'), np.load('E://GWNMMAD_data/Tw_dataset/Datasets/background.npz')['data']), axis = 0)[:187500,0,:]
    for dtype in config_dict['Dataset_type']:
        config_dic_cached = deepcopy(config_dict)
        config_dic_cached['Dataset_path'] = os.path.join(config_dic_cached['Dataset_dir'], dtype)
        config_dic_cached['Cutting_window'] = config_dict['Cutting_window'][dtype]
        dataset[dtype] = cut_events_from_waveforms(config_dic_cached)
        
    list_dataset = dataset.keys()
    
    # for ds in dataset.keys():
    #     np.random.shuffle(dataset[ds])

    # dataset['noise_L'] = dataset['noise_L'][:410000]
    # dataset['noise_H'] = dataset['noise_H'][:410000]

    # listSNR = ['5-12','12-24','24-48','48-96']
    # for dt in ['BBH', 'SGLF', 'SGHF']:
    #     for snr in listSNR:
    #         foo = np.load(dataDir+f'/Noise_processing/For_WSC_pipeline/{dt}_events_v1_snr_{snr_dict[snr]}.npy')[:22500]
    #         dataset[dt+'_L_'+snr] = foo[:, 1]
    #         dataset[dt+'_H_'+snr] = foo[:, 0]
    #         list_dataset += [dt+'_L_'+snr, dt+'_H_'+snr]
            # np.random.shuffle(dataset[dt+'_L_'+snr])
            # np.random.shuffle(dataset[dt+'_H_'+snr])

    for ds in list_dataset:
        dataset[ds] /= np.linalg.norm(dataset[ds], axis=-1)[:,:,np.newaxis]
        dataset_fft[ds] = abs(np.fft.rfft(dataset[ds]))
        dataset_fft[ds] /= np.linalg.norm(dataset_fft[ds], axis=-1)[:,:,np.newaxis]
        
    dataset_final = dataset_fft
    
    return dataset_final




def training_create_dataset(config_dict):
    
    dataset = {}
    full_dataset_raw = make_training_separated_and_normalized_datasets(config_dict)
    cache_path = config_dict['Cache_path']
    
    dtype_list = list(full_dataset_raw.keys())
    idxflag = 0
    
    if 'glitch' in dtype_list:
        # if glitch is presented, set it to the first one, and separated it to be glitch H and glitch L.
        
        if False:
            assert 'noise' in dtype_list
            # Still need to do something here, to make sure it follows the sequence of detectors
            num_of_glitches = len(full_dataset_raw['glitch'])
            dataset[idxflag] = np.concatenate((full_dataset_raw['glitch'][:,[0],:], full_dataset_raw['noise'][:num_of_glitches,[1],:]), axis = 1).reshape(-1,202)
            dataset[idxflag+1] = np.concatenate((full_dataset_raw['noise'][:num_of_glitches,[0],:], full_dataset_raw['glitch'][:,[1],:]), axis = 1).reshape(-1,202)
            
            full_dataset_raw['noise'] = full_dataset_raw['noise'][num_of_glitches:]
            dtype_list.remove('glitch')
            idxflag += 2
        
    for dtype in dtype_list:
        dataset[idxflag] = full_dataset_raw[dtype].reshape(len(full_dataset_raw[dtype], -1))
        idxflag += 1
    
    torch.save(dataset, cache_path)
    
    return dataset

def testing_create_dataset(config_dict):
    # Returnning just the full test dataset
    # Bring severe memory burden. New method introduced in the filtering_chain.py
    
    # scanning_type = config_dict['Scan_type']
    # smoothing_window = config_dict['Smoothing_window']
    
    # if scanning_type == 'full_scan':
    
    dataset = make_training_separated_and_normalized_datasets(config_dict)[config_dict['Dataset_type'][0]]
    cache_path = config_dict['Cache_path']
    
    np.save(cache_path, dataset)
    
    return dataset

if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'event_window_config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
        
    # dataset_trial = torch.load('/home/app/test_data/trial.json', weights_only=False)
    
    
    # print(dataset_trial.keys())
    
    # config['Weakly_Supervised']['Training_scheme']['Output_dir'] = config['Full_pipeline']['Training_scheme']['Output_dir']
    
    # trainSeriesSupC_struct(dataset_trial, config_dict=config['Weakly_Supervised'])
    
    training_create_dataset(config['Training_set_config'])