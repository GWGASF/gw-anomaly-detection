import numpy as np
import os
import torch
import h5py

def read_from_files(file_path):
    injected_waveforms = np.concatenate((np.array(h5py.File(file_path, 'r')['H1'])[:, np.newaxis, :], np.array(h5py.File(file_path, 'r')['L1'])[:, np.newaxis, :]), axis = 1)
    raw_waveforms = np.concatenate((np.array(h5py.File(file_path, 'r')['waveform_H1'])[:, np.newaxis, :], np.array(h5py.File(file_path, 'r')['waveform_L1'])[:, np.newaxis, :]), axis = 1)
    inject_params = np.array(h5py.File(file_path, 'r')['waveform_parameters'])
    
    return injected_waveforms, raw_waveforms, inject_params


def cut_events_from_waveforms(cutting_config):
    
    EVENT_PER_SEGMENT = 
    segment_freq = 
    segment_length = 
    file_path = 
    
    waveforms = read_from_files(file_path)
    
    cutted_events = np.empty((len(waveforms),EVENT_PER_SEGMENT,2,200))

    for i in range(len(waveforms)):
    # for i in range(1):
        for j in range(EVENT_PER_SEGMENT):
            starting_time = np.random.choice(segment_freq * (segment_length-2) - 200, 1, replace = False)
            ending_time = starting_time + 200
            
            cache = waveforms[i][:,starting_time[0]:ending_time[0]]
        
            cutted_events[i,j] = cache.copy()
    
    return cutted_events

    
def make_training_separated_and_normalized_datasets(config_dict):

    # dataDir = "../../../../Data_cached"
    # list_dataset = ['glitch_L', 'glitch_H', 'noise_L', 'noise_H']
    dataset = {}
    dataset_fft = {}

    # dataset['glitch_L'] = np.load(dataDir+"/real_glitches_snrlt5_60132_4000Hz_25ms.npz")["strain_time_data"][:12500];
    # dataset['glitch_H'] = np.load(dataDir+"/real_glitches_H_snrlt5_59732_4000Hz_25ms.npz")["strain_time_data"][:12500];
    # dataset['noise_L'] = np.concatenate((np.load(dataDir+'/Noise_processing/Processed_noise_sets/noise_sets_v1.npy'), np.load('E://GWNMMAD_data/Tw_dataset/Datasets/background.npz')['data']), axis = 0)[:187500,1,:]
    # dataset['noise_H'] = np.concatenate((np.load(dataDir+'/Noise_processing/Processed_noise_sets/noise_sets_v1.npy'), np.load('E://GWNMMAD_data/Tw_dataset/Datasets/background.npz')['data']), axis = 0)[:187500,0,:]
    
    dataset = cut_events_from_waveforms(config_dict_)
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
        dataset[ds] /= np.linalg.norm([dataset[ds]], axis=2).T
        dataset_fft[ds] = abs(np.fft.rfft(dataset[ds]))
        dataset_fft[ds] /= np.linalg.norm([dataset_fft[ds]], axis=2).T
        
    dataset_final = dataset_fft
    
    return dataset_final


def training_create_dataset(config_dict):
    
    return 

def testing_create_dataset():
    
    return