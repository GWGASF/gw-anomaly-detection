#!/bin/python
import glob
import os
import h5py
import numpy as np

from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries

import bilby
from bilby.core.prior.analytical import Uniform, Cosine, Sine, PowerLaw, Gamma, Logistic
from bilby.gw.conversion import bilby_to_lalsimulation_spins

import lal
from pycbc.waveform import get_sgburst_waveform
from pycbc.waveform import get_td_waveform
from pycbc.types.timeseries import TimeSeries as pycbcts
from pycbc.detector import Detector

channels = {
    "H1": "H1:DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01",
    "L1": "L1:DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01",
}

def get_source(ifo, start, end):
    folder_prefix = f'/ceph/mirror/frames/O3/hoft_C01_clean_sub60Hz/{ifo}/{ifo[0]}-{ifo}_HOFT_CLEAN_SUB60HZ_C01'
    label0 = int(start/1e5)
    labelp = label0 - 1
    folderp = f"{folder_prefix}-{labelp}"
    if os.path.exists(folderp):
        labels = [labelp, label0]
    else:
        labels = [label0]
    if end >= (label0+1)*1e5:
        labels.append(label0+1)

    gwf_folders = [f'{folder_prefix}-{label}' for label in labels] 
    source = []
    for gwf_folder in gwf_folders:
        source.extend(glob.glob(f'{gwf_folder}/*.gwf'))

    source = sorted(source)
    return source

def get_bg_ts(ifo, t0, window_length=8):
    bg_start, bg_end = t0, t0 + window_length
    source = get_source(ifo, bg_start, bg_end)
    ts = TimeSeries.read(
        source=source,
        channel=channels[ifo],
        start=bg_start,
        end=bg_end,
        gap='ignore',
    )
    return ts

def select_asd(trigtime, asd_list):
    asd_times = [(int(file.split('/')[-1].split('-')[1]), int(file.split('/')[-1].split('-')[-1].split('.')[0])) for file in asd_list]
    for i in range(len(asd_times)):
        asd_sted = [asd_times[i][0], asd_times[i][0] + asd_times[i][1]]
        if trigtime >= asd_sted[0] and trigtime < asd_sted[1]:
            asd = FrequencySeries.read(asd_list[i])

    return asd

def sg_params(mode):
    # Making sg_parameters from bilby
    sg_priors = dict()
    sg_priors['Q'] = Uniform(25, 75, name='Q', latex_label='q')
    if mode == 'low':
        sg_priors['frequency'] = Uniform(64, 512, name='frequency', latex_label='frequency')
    elif mode == 'high':
        sg_priors['frequency'] = Uniform(512, 1024, name='frequency', latex_label='frequency')
    sg_priors['hrss'] = Uniform(1e-24, 1e-22, name='hrss', latex_label='hrss')
    sg_priors['phase'] = Uniform(0, 2*np.pi, name='phase', latex_label='phi')
    sg_priors['ra'] = Uniform(0, 2*np.pi, name='ra', latex_label='ra')
    sg_priors['dec'] = Cosine(-np.pi/2, np.pi/2, name='dec', latex_label='dec')
    sg_priors['eccentricity'] = Uniform(0, 0.01, name='eccentricity', latex_label='ecc')
    sg_priors['psi'] = Uniform(0, 2*np.pi, name='psi', latex_label='psi')
    # Sampling
    waveform_parameters = {key:sg_priors[key].sample(1)[0] for key in sg_priors}
    return waveform_parameters

def sg_waveform(ifo, waveform_parameters, t0, length=8, sampling_frequency=16384):
    sghp, sghc = get_sgburst_waveform(
        q=waveform_parameters['Q'],
        frequency=waveform_parameters['frequency'],
        delta_t=1/sampling_frequency,
        hrss=waveform_parameters['hrss'],
    )
    signal = Detector(ifo).project_wave(
        hp=sghp,
        hc=sghc,
        ra=waveform_parameters['ra'],
        dec=waveform_parameters['dec'],
        polarization=waveform_parameters['psi'],
    )
    p_pad_length = int((length/2 - abs(signal.sample_times.min()))*sampling_frequency)
    n_pad_length = int((length/2 - signal.sample_times.max())*sampling_frequency)
    if (int(p_pad_length) + len(signal) + int(n_pad_length)) == length*sampling_frequency:
        padded_signal = np.pad(signal.numpy(), (p_pad_length, n_pad_length), constant_values=(0, 0))
    else:
        padded_signal = np.pad(signal.numpy(), (p_pad_length, n_pad_length - 1), constant_values=(0, 0))

    signal_ts = TimeSeries(
        padded_signal,
        name=f'{ifo}:SG_SIG',
        channel=f'{ifo}:SG_SIG',
        sample_rate=sampling_frequency,
        t0=t0,
    )
    return signal_ts

def bbh_params(mode=None):
    # Making bbh_parameters from bilby
    bbh_priors = dict()
    bbh_priors['Mc'] = Uniform(25, 100, name='Chirp Mass', latex_label='Mc')
    bbh_priors['q'] = Uniform(0.125, 1, name='mass ratio', latex_label='q')
    bbh_priors['a1'] = Uniform(0, 0.99, name='a1')
    bbh_priors['a2'] = Uniform(0, 0.99, name='a2')
    bbh_priors['tilt1'] = Sine(0, np.pi, name='tilt1')
    bbh_priors['tilt2'] = Sine(0, np.pi, name='tilt2')
    bbh_priors['phi12'] = Uniform(0, 2*np.pi, name='phi12')
    bbh_priors['phijl'] = Uniform(0, 2*np.pi, name='phijl')
    bbh_priors['thetajn'] = Sine(0, np.pi, name='thetajn')
    # bbh_priors['distance'] = PowerLaw(alpha=2, minimum=20, maximum=8000, unit='Mpc', name='distance')
    # bbh_priors['distance'] = Gamma(k=3, theta=1200, unit='Mpc', name='distance')
    bbh_priors['distance'] = Uniform(50, 5000, unit='Mpc', name='distance')
    bbh_priors['phase'] = Uniform(0, 2*np.pi, name='phase', latex_label='phi')
    bbh_priors['ra'] = Uniform(0, 2*np.pi, name='ra', latex_label='ra')
    bbh_priors['dec'] = Cosine(-np.pi/2, np.pi/2, name='dec', latex_label='dec')
    bbh_priors['psi'] = Uniform(0, 2*np.pi, name='psi', latex_label='psi')

    waveform_parameters = {key:bbh_priors[key].sample(1)[0] for key in bbh_priors}
    Mc, q = waveform_parameters['Mc'], waveform_parameters['q'] 
    m1 = Mc*(1+q)**(1/5)*q**(-3/5)
    m2 = Mc*(1+q)**(1/5)*q**(2/5)
    while(m1 < 5 or m1 > 100 or m2 < 5 or m2 > 100):
        inject_parameters = {key:bbh_priors[key].sample(1)[0] for key in bbh_priors}
        Mc, q = inject_parameters['Mc'], inject_parameters['q'] 
        m1 = Mc*(1+q)**(1/5)*q**(-3/5)
        m2 = Mc*(1+q)**(1/5)*q**(2/5)

    waveform_parameters['mass1'] = Mc*(1+q)**(1/5)*q**(-3/5)
    waveform_parameters['mass2'] = Mc*(1+q)**(1/5)*q**(2/5)
    waveform_parameters['f_ref'] = 10
    lal_spins = bilby_to_lalsimulation_spins(
        theta_jn=waveform_parameters['thetajn'],
        phi_jl=waveform_parameters['phijl'],
        tilt_1=waveform_parameters['tilt1'],
        tilt_2=waveform_parameters['tilt2'],
        phi_12=waveform_parameters['phi12'],
        a_1=waveform_parameters['a1'],
        a_2=waveform_parameters['a2'],
        mass_1=waveform_parameters['mass1']*lal.MSUN_SI,
        mass_2=waveform_parameters['mass2']*lal.MSUN_SI,
        reference_frequency=waveform_parameters['f_ref'],
        phase=waveform_parameters['phase'],
    )
    waveform_parameters['inclination'] = lal_spins[0]
    waveform_parameters['spin1x'] = lal_spins[1]
    waveform_parameters['spin1y'] = lal_spins[2]
    waveform_parameters['spin1z'] = lal_spins[3]
    waveform_parameters['spin2x'] = lal_spins[4]
    waveform_parameters['spin2y'] = lal_spins[5]
    waveform_parameters['spin2z'] = lal_spins[6]
    waveform_parameters['approximant'] = "IMRPhenomPv2"
    return waveform_parameters

def bbh_waveform(ifo, waveform_parameters, t0, length=8, sampling_frequency=16384):
    hp, hc = get_td_waveform(
        approximant=waveform_parameters['approximant'],
        delta_t=1/sampling_frequency,
        f_lower=30,
        f_ref=waveform_parameters['f_ref'],
        mass1=waveform_parameters['mass1'],
        mass2=waveform_parameters['mass2'],
        spin1x=waveform_parameters['spin1x'],
        spin1y=waveform_parameters['spin1y'],
        spin1z=waveform_parameters['spin1z'],
        spin2x=waveform_parameters['spin2x'],
        spin2y=waveform_parameters['spin2y'],
        spin2z=waveform_parameters['spin2z'],
        distance=waveform_parameters['distance'],
        coa_phase=waveform_parameters['phase'],
        inclination=waveform_parameters['inclination'],
    )
    signal = Detector(ifo).project_wave(
        hp=hp,
        hc=hc,
        ra=waveform_parameters['ra'],
        dec=waveform_parameters['dec'],
        polarization=waveform_parameters['psi'],
    )
    st = signal.sample_times[0] - signal.sample_times[0]%(1/sampling_frequency)
    ed = signal.sample_times[-1] - signal.sample_times[-1]%(1/sampling_frequency)
    plength = int((st - -length/2)*sampling_frequency)
    nlength = int((ed - length/2)*sampling_frequency)
    if plength >= 0:
        pad_signal = np.pad(signal.numpy(), (plength, 0), constant_values=(0, 0))
    if plength < 0:
        pad_signal = signal.numpy()
        pad_signal = pad_signal[-plength:]
    if nlength <= 0:
        pad_signal = np.pad(pad_signal, (0, -nlength), constant_values=(0, 0))
    if nlength > 0:
        pad_signal = pad_signal[:length*sampling_frequency]
    signal_ts = TimeSeries(
        pad_signal,
        name=f'{ifo}:BBH_SIG',
        channel=f'{ifo}:BBH_SIG',
        sample_rate=sampling_frequency,
        t0=t0,
    )
    return signal_ts

def ccsn_params(mode=None, qm_file=None):
    if qm_file == None:
        raise RuntimeError("Please provide the h5 file of the simulated quadrupole moments.")
    # Making ccsn_parameters from bilby
    ccsn_priors = dict()
    ccsn_priors['phi'] = Uniform(0, 2*np.pi, name='phijl')
    ccsn_priors['theta'] = Sine(0, np.pi, name='thetajn')
    ccsn_priors['distance'] = Uniform(50, 5000, unit='Mpc', name='distance')
    ccsn_priors['phase'] = Uniform(0, 2*np.pi, name='phase', latex_label='phi')
    ccsn_priors['ra'] = Uniform(0, 2*np.pi, name='ra', latex_label='ra')
    ccsn_priors['dec'] = Cosine(-np.pi/2, np.pi/2, name='dec', latex_label='dec')
    ccsn_priors['psi'] = Uniform(0, 2*np.pi, name='psi', latex_label='psi')

    waveform_parameters = {key:ccsn_priors[key].sample(1)[0] for key in ccsn_priors}
    family = qm_file.split('/')[-2]
    simulation = f"{family}/{qm_file.split('/')[-1].split('.')[0]}"
    print(f"CCSN: {simulation}")
    waveform_parameters['simulation'] = simulation
    return waveform_parameters

def ccsn_waveform(ifo, waveform_parameters, qm_file, t0, length=8, sampling_frequency=16384):
    theta = waveform_parameters['theta']
    phi = waveform_parameters['phi']
    print(qm_file)
    with h5py.File(qm_file, 'r') as f:
        times = f['time'][:]
        qm = f['quad_moment'][:]
    hp =\
        qm[:,0,0]*(np.cos(theta)**2*np.cos(phi)**2 - np.sin(phi)**2).reshape(-1, 1)\
        + qm[:,1,1]*(np.cos(theta)**2*np.sin(phi)**2 - np.cos(phi)**2).reshape(-1, 1)\
        + qm[:,2,2]*(np.sin(theta)**2).reshape(-1, 1)\
        + qm[:,0,1]*(np.cos(theta)**2*np.sin(2*phi) - np.sin(2*phi)).reshape(-1, 1)\
        - qm[:,1,2]*(np.sin(2*theta)*np.sin(phi)).reshape(-1, 1)\
        - qm[:,2,0]*(np.sin(2*theta)*np.cos(phi)).reshape(-1, 1)
    hp = pycbcts(hp[0], delta_t=1/sampling_frequency, epoch=times[0])
    hc = 2*(
        - qm[:,0,0]*(np.cos(theta)*np.sin(phi)*np.cos(phi)).reshape(-1, 1)
        + qm[:,1,1]*(np.cos(theta)*np.sin(phi)*np.cos(phi)).reshape(-1, 1)
        + qm[:,0,1]*(np.cos(theta)*np.cos(2*phi)).reshape(-1, 1)
        - qm[:,1,2]*(np.sin(theta)*np.cos(phi)).reshape(-1, 1)
        + qm[:,2,0]*(np.sin(theta)*np.sin(phi)).reshape(-1, 1)
    )
    hc = pycbcts(hc[0], delta_t=1/sampling_frequency, epoch=times[0])
    signal = Detector(ifo).project_wave(
        hp=hp,
        hc=hc,
        ra=waveform_parameters['ra'],
        dec=waveform_parameters['dec'],
        polarization=waveform_parameters['psi'],
    )
    st = signal.sample_times[0] - signal.sample_times[0]%(1/sampling_frequency)
    ed = signal.sample_times[-1] - signal.sample_times[-1]%(1/sampling_frequency)
    plength = int((st - -length/2)*sampling_frequency)
    nlength = int((ed - length/2)*sampling_frequency)
    if plength >= 0:
        pad_signal = np.pad(signal.numpy(), (plength, 0), constant_values=(0, 0))
    if plength < 0:
        pad_signal = signal.numpy()
        pad_signal = pad_signal[-plength:]
    if nlength <= 0:
        pad_signal = np.pad(pad_signal, (0, -nlength), constant_values=(0, 0))
    if nlength > 0:
        pad_signal = pad_signal[:length*sampling_frequency]
    signal_ts = TimeSeries(
        pad_signal,
        name=f'{ifo}:BBH_SIG',
        channel=f'{ifo}:BBH_SIG',
        sample_rate=sampling_frequency,
        t0=t0,
    )
    return signal_ts

def estimate_snr(signal, asd, flow=30, fhigh=1500):
    sigf = signal.fft()
    sigf = sigf/sigf.df
    snrf = sigf.conj()*sigf/asd**2
    snr = 4*snrf[int(flow/snrf.df.value):int(fhigh/snrf.df.value)].real.sum()*snrf.df
    return snr.to_value()**0.5

def network_snr(snrs):
    return np.sqrt(np.square(np.array(snrs)).sum())

def rescale_snr(target_snr, signals, asds, flow=30, fhigh=1500):
    ifos = list(signals.keys())
    ifo_snrs = dict.fromkeys(ifos) 
    re_sigs = dict.fromkeys(ifos) 
    re_ifo_snrs = dict.fromkeys(ifos) 
    for ifo in ifos:
        ifo_snrs[ifo] = estimate_snr(signals[ifo], asds[ifo], flow, fhigh)

    current_snr = network_snr(list(ifo_snrs.values()))
    scale_factor = target_snr/current_snr
    for ifo in ifos:
        re_sigs[ifo] = scale_factor * signals[ifo]
        re_ifo_snrs[ifo] = scale_factor * ifo_snrs[ifo]
    return re_sigs, re_ifo_snrs

def process(
        ts,
        asd,
        flow=30,
        fhigh=1500,
        resample=4096,
        crop_length=1,
):
    ts = ts.whiten(asd=asd)
    ts = ts.bandpass(flow, fhigh)
    ts = ts.crop(ts.t0.value + crop_length, ts.t0.value + ts.duration.value - crop_length)
    ts = ts.resample(resample)
    return ts
