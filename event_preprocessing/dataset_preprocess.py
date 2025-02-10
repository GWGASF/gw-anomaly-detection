import numpy as np
import os
import torch




dataDir = "../../../../Data_cached"
list_dataset = ['glitch_L', 'glitch_H', 'noise_L', 'noise_H']
dataset = {};
dataset_fft = {};

dataset['glitch_L'] = np.load(dataDir+"/real_glitches_snrlt5_60132_4000Hz_25ms.npz")["strain_time_data"][:12500];
dataset['glitch_H'] = np.load(dataDir+"/real_glitches_H_snrlt5_59732_4000Hz_25ms.npz")["strain_time_data"][:12500];
dataset['noise_L'] = np.concatenate((np.load(dataDir+'/Noise_processing/Processed_noise_sets/noise_sets_v1.npy'), np.load('E://GWNMMAD_data/Tw_dataset/Datasets/background.npz')['data']), axis = 0)[:187500,1,:]
dataset['noise_H'] = np.concatenate((np.load(dataDir+'/Noise_processing/Processed_noise_sets/noise_sets_v1.npy'), np.load('E://GWNMMAD_data/Tw_dataset/Datasets/background.npz')['data']), axis = 0)[:187500,0,:]

for ds in list_dataset:
    np.random.shuffle(dataset[ds])

# dataset['noise_L'] = dataset['noise_L'][:410000]
# dataset['noise_H'] = dataset['noise_H'][:410000]

listSNR = ['5-12','12-24','24-48','48-96']
for dt in ['BBH', 'SGLF', 'SGHF']:
    for snr in listSNR:
        foo = np.load(dataDir+f'/Noise_processing/For_WSC_pipeline/{dt}_events_v1_snr_{snr_dict[snr]}.npy')[:22500]
        dataset[dt+'_L_'+snr] = foo[:, 1]
        dataset[dt+'_H_'+snr] = foo[:, 0]
        list_dataset += [dt+'_L_'+snr, dt+'_H_'+snr]
        # np.random.shuffle(dataset[dt+'_L_'+snr])
        # np.random.shuffle(dataset[dt+'_H_'+snr])

for ds in list_dataset:
    dataset[ds] /= np.linalg.norm([dataset[ds]], axis=2).T
    dataset_fft[ds] = abs(np.fft.rfft(dataset[ds]))
    dataset_fft[ds] /= np.linalg.norm([dataset_fft[ds]], axis=2).T
    
dataset_final = dataset_fft