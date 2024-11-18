#!/bin/bash
import os
import glob
import h5py as h5
import numpy as np

from inj_utils import get_source
from inj_utils import get_bg_ts
from inj_utils import select_asd
from inj_utils import sg_params
from inj_utils import sg_waveform
from inj_utils import bbh_params
from inj_utils import bbh_waveform
from inj_utils import ccsn_params
from inj_utils import ccsn_waveform
from inj_utils import estimate_snr
from inj_utils import rescale_snr
from inj_utils import process

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-w", "--waveform", default=None, type=str, help="The type of the waveforms to inject, bbh, lfsg or hfsg.")
parser.add_argument("-QM", "--qm-file", default=None, type=str, help="The h5 file of the simulated quadrupole moments of ccsn waveforms.")
parser.add_argument("-h1", "--h1-background", default=None, type=str, help="The hdf5 file of the background t0 of the detector H1")
parser.add_argument("-l1", "--l1-background", default=None, type=str, help="The hdf5 file of the background t0 of the detector L1")
parser.add_argument("-s", "--idstart", default=None, type=int, help="The start id of the background noise.")
parser.add_argument("-e", "--idend", default=None, type=int, help="The end id of the background noise.")
parser.add_argument("-L", "--snr-low", default=None, type=float, help="The lower bound of the SNR of the injected events.")
parser.add_argument("-H", "--snr-high", default=None, type=float, help="The lower bound of the SNR of the injected events.")
parser.add_argument("-afH", "--asd-folder-H1", default=None, type=str, help="The folder where the H1 asd.txt files are stored.")
parser.add_argument("-afL", "--asd-folder-L1", default=None, type=str, help="The folder where the H1 asd.txt files are stored.")
parser.add_argument("-o", "--output-directory", default=None, type=str, help="The path of the directory to output hdf5s.")
args = parser.parse_args()

waveform = args.waveform
qm_file = args.qm_file
h1_background = args.h1_background
l1_background = args.l1_background
idstart = args.idstart
idend = args.idend
snr_low = args.snr_low
snr_high = args.snr_high
output_directory = args.output_directory
asd_folder_H1 = args.asd_folder_H1
asd_folder_L1 = args.asd_folder_L1

def main():
    ifos = ["H1", "L1"]
    bg_files = [h1_background, l1_background]
    duration = 8
    sampling_frequency = 16384
    # Load bgseg_t_list and asd_list
    bgseg_t_list = dict.fromkeys(ifos)
    asd_folder = dict.fromkeys(ifos)
    asd_folder["H1"] = asd_folder_H1
    asd_folder["L1"] = asd_folder_L1
    asd_list = dict.fromkeys(ifos)
    for bg_file, ifo in zip(bg_files, ifos):
        if bg_file.split('/')[-1].split('_')[0] != ifo:
            raise RuntimeError(f"The background file of {ifo} is not correct. Please check again.")
        bgseg_t_list_fname = bg_file
    
        with h5.File(bgseg_t_list_fname, 'r') as f:
            bgseg_t_list[ifo] = f['bgseg_t_list'][idstart:idend]
            print(f"Number of time in bgseg_t_list, {ifo}: {len(bgseg_t_list[ifo])}")
    
        asd_list[ifo] = glob.glob(f"{asd_folder[ifo]}/*.txt")
        asd_list[ifo] = sorted(asd_list[ifo])

    inject_parameter_list = []
    sig_list = []
    whitened_inj_list = []
    # LFSG, HFSG, BBH
    if waveform == 'lfsg':
        mode = 'low'
        param_generator = sg_params
        waveform_generator = sg_waveform
    elif waveform == 'hfsg':
        mode = 'high'
        param_generator = sg_params
        waveform_generator = sg_waveform
    elif waveform == 'bbh':
        mode = None
        param_generator = bbh_params
        waveform_generator = bbh_waveform
    elif waveform == 'ccsn':
        mode = None
        param_generator = ccsn_params
        waveform_generator = ccsn_waveform
    
    for i in range(idend-idstart):
        try:
            if waveform == 'ccsn':
                inject_parameters = param_generator(mode=mode, qm_file=qm_file)
            else:
                inject_parameters = param_generator(mode=mode)
            target_snr = np.random.uniform(snr_low, snr_high)
            inject_parameters[f'network_snr'] = target_snr
            signal_ts = dict()
            asd = dict()
            bg_ts = dict()
            inj_ts = dict()
            whitened_inj_ts = dict()
            for ifo in ifos:
                t0 = bgseg_t_list[ifo][i]
                print(f"Background noise starts from {t0}...")
                bg_ts[ifo] = get_bg_ts(ifo, t0)
                print("Waveform...")
                if waveform == 'ccsn':
                    signal_ts[ifo] = waveform_generator(ifo, inject_parameters, qm_file, bg_ts[ifo].t0.value)
                else:
                    signal_ts[ifo] = waveform_generator(ifo, inject_parameters, bg_ts[ifo].t0.value)
                print("ASD...")
                asd[ifo] = select_asd(t0, asd_list[ifo])
                inject_parameters[f't0_{ifo}'] = t0 

            # Rescale SNR, inject signal and process the injected timeseries
            print("Rescaling...")
            re_sigs, re_snrs = rescale_snr(target_snr, signal_ts, asd)
            print("Injecting...")
            for ifo in ifos:
                inject_parameters[f'snr_{ifo}'] = re_snrs[ifo]
                inj_ts[ifo] = bg_ts[ifo].inject(re_sigs[ifo])
                whitened_inj_ts[ifo] = process(inj_ts[ifo], asd[ifo]).to_value()
                signal_ts[ifo] = re_sigs[ifo].resample(4096).crop(signal_ts[ifo].t0.value + 1, signal_ts[ifo].t0.value + signal_ts[ifo].duration.value - 1).to_value()

            if len(signal_ts['H1']) == len(whitened_inj_ts['H1']) and len(whitened_inj_ts['H1']) == 6*4096 and len(signal_ts['L1']) == len(whitened_inj_ts['L1']) and len(whitened_inj_ts['L1']) == 6*4096:
                inject_parameter_list.append(inject_parameters)
                sig_list.append(signal_ts)
                whitened_inj_list.append(whitened_inj_ts)
            print(f'{len(inject_parameter_list)} injections produced.')
        except Exception as e:
            print(f"Problem with injection {i}")
            print(str(e))
            continue

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    if len(inject_parameter_list) != 0:
        output = f"{output_directory}/inj_{waveform}-{idstart}-{idend}.hdf5"
        with h5.File(output, 'w') as w:
            names = list(inject_parameter_list[0].keys())
            if waveform == 'lfsg' or waveform == 'hfsg':
                dt = np.dtype({'names': names, 'formats': ['f8']*len(inject_parameter_list[0])})
                data = np.array([tuple(inject_parameter.values()) for inject_parameter in inject_parameter_list], dtype=dt)
                w.create_dataset(
                    'inject_parameters',
                    shape=data.shape,
                    dtype=dt,
                    data=data,
                )
            elif waveform == 'bbh':
                dt = np.dtype({'names': names, 'formats': ['f8']*(len(inject_parameter_list[0]) - 6) + [h5.string_dtype(encoding='ascii')] + ['f8']*5})
                data = np.array([tuple(inject_parameter.values()) for inject_parameter in inject_parameter_list], dtype=dt)
                w.create_dataset(
                    'inject_parameters',
                    shape=data.shape,
                    dtype=dt,
                    data=data,
                )
            elif waveform == 'ccsn':
                dt = np.dtype({'names': names, 'formats': ['f8']*(len(inject_parameter_list[0]) - 6) + [h5.string_dtype(encoding='ascii')] + ['f8']*5})
                data = np.array([tuple(inject_parameter.values()) for inject_parameter in inject_parameter_list], dtype=dt)
                w.create_dataset(
                    'inject_parameters',
                    shape=data.shape,
                    dtype=dt,
                    data=data,
                )

            for ifo in ifos:
                data = np.stack([ts[ifo] for ts in whitened_inj_list])
                w.create_dataset(
                    ifo,
                    shape=data.shape,
                    dtype=data.dtype,
                    data=data
                )
                data = np.stack([ts[ifo] for ts in sig_list])
                w.create_dataset(
                    f'{waveform}_{ifo}',
                    shape=data.shape,
                    dtype=data.dtype,
                    data=data
                )
    else:
        print('No injection produced.')

if __name__ == "__main__":
    main()