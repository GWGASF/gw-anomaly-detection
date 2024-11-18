#!/bin/python
import h5py
import numpy as np
import random

from gwpy.timeseries import TimeSeries

from bilby.core.prior.analytical import Uniform, Cosine, Sine
from bilby.gw.conversion import bilby_to_lalsimulation_spins

import lal
from pycbc.waveform import get_sgburst_waveform
from pycbc.waveform import get_td_waveform
from pycbc.types.timeseries import TimeSeries as pycbcts
from pycbc.detector import Detector

class Waveforms():
    def __init__(
            self,
            ifos: list,
            sampling_frequency: float=16384,
            length: float=4,
        ):
        self.ifos = ifos
        self.sampling_frequency = sampling_frequency
        self.length = length

    def sg_params(
            self,
            mode: str,
            **kwargs,
        ):
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

    def sg_waveform(
            self,
            ifo: str,
            waveform_parameters: dict,
            t0: float=0,
        ):
        sghp, sghc = get_sgburst_waveform(
            q=waveform_parameters['Q'],
            frequency=waveform_parameters['frequency'],
            delta_t=1/self.sampling_frequency,
            hrss=waveform_parameters['hrss'],
        )
        signal = Detector(ifo).project_wave(
            hp=sghp,
            hc=sghc,
            ra=waveform_parameters['ra'],
            dec=waveform_parameters['dec'],
            polarization=waveform_parameters['psi'],
        )
        p_pad_length = int((self.length/2 - abs(signal.sample_times.min()))*self.sampling_frequency)
        n_pad_length = int((self.length/2 - signal.sample_times.max())*self.sampling_frequency)
        if (int(p_pad_length) + len(signal) + int(n_pad_length)) == self.length*self.sampling_frequency:
            padded_signal = np.pad(signal.numpy(), (p_pad_length, n_pad_length), constant_values=(0, 0))
        else:
            padded_signal = np.pad(signal.numpy(), (p_pad_length, n_pad_length - 1), constant_values=(0, 0))

        signal_ts = TimeSeries(
            padded_signal,
            name=f'{ifo}:SG_SIG',
            channel=f'{ifo}:SG_SIG',
            sample_rate=self.sampling_frequency,
            t0=t0,
        )
        return signal_ts

    def bbh_params(
            self,
            **kwargs,
        ):
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

    def bbh_waveform(
            self,
            ifo: str,
            waveform_parameters: dict,
            t0: float=0,
        ):
        hp, hc = get_td_waveform(
            approximant=waveform_parameters['approximant'],
            delta_t=1/self.sampling_frequency,
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
        st = signal.sample_times[0] - signal.sample_times[0]%(1/self.sampling_frequency)
        ed = signal.sample_times[-1] - signal.sample_times[-1]%(1/self.sampling_frequency)
        plength = int((st - -self.length/2)*self.sampling_frequency)
        nlength = int((ed - self.length/2)*self.sampling_frequency)
        if plength >= 0:
            pad_signal = np.pad(signal.numpy(), (plength, 0), constant_values=(0, 0))
        if plength < 0:
            pad_signal = signal.numpy()
            pad_signal = pad_signal[-plength:]
        if nlength <= 0:
            pad_signal = np.pad(pad_signal, (0, -nlength), constant_values=(0, 0))
        if nlength > 0:
            pad_signal = pad_signal[:self.length*self.sampling_frequency]
        signal_ts = TimeSeries(
            pad_signal,
            name=f'{ifo}:BBH_SIG',
            channel=f'{ifo}:BBH_SIG',
            sample_rate=self.sampling_frequency,
            t0=t0,
        )
        return signal_ts

    def ccsn_params(
            self,
            qm_file: str=None,
            **kwargs,
        ):
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
        waveform_parameters['simulation'] = simulation
        waveform_parameters['qm_file'] = qm_file
        return waveform_parameters

    def ccsn_waveform(
            self,
            ifo: str,
            waveform_parameters: dict,
            t0: float=0,
        ):
        theta = waveform_parameters['theta']
        phi = waveform_parameters['phi']
        qm_file = waveform_parameters['qm_file']
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
        hp = pycbcts(hp[0], delta_t=1/self.sampling_frequency, epoch=times[0])
        hc = 2*(
            - qm[:,0,0]*(np.cos(theta)*np.sin(phi)*np.cos(phi)).reshape(-1, 1)
            + qm[:,1,1]*(np.cos(theta)*np.sin(phi)*np.cos(phi)).reshape(-1, 1)
            + qm[:,0,1]*(np.cos(theta)*np.cos(2*phi)).reshape(-1, 1)
            - qm[:,1,2]*(np.sin(theta)*np.cos(phi)).reshape(-1, 1)
            + qm[:,2,0]*(np.sin(theta)*np.sin(phi)).reshape(-1, 1)
        )
        hc = pycbcts(hc[0], delta_t=1/self.sampling_frequency, epoch=times[0])
        signal = Detector(ifo).project_wave(
            hp=hp,
            hc=hc,
            ra=waveform_parameters['ra'],
            dec=waveform_parameters['dec'],
            polarization=waveform_parameters['psi'],
        )
        st = signal.sample_times[0] - signal.sample_times[0]%(1/self.sampling_frequency)
        ed = signal.sample_times[-1] - signal.sample_times[-1]%(1/self.sampling_frequency)
        plength = int((st - -self.length/2)*self.sampling_frequency)
        nlength = int((ed - self.length/2)*self.sampling_frequency)
        if plength >= 0:
            pad_signal = np.pad(signal.numpy(), (plength, 0), constant_values=(0, 0))
        if plength < 0:
            pad_signal = signal.numpy()
            pad_signal = pad_signal[-plength:]
        if nlength <= 0:
            pad_signal = np.pad(pad_signal, (0, -nlength), constant_values=(0, 0))
        if nlength > 0:
            pad_signal = pad_signal[:self.length*self.sampling_frequency]
        signal_ts = TimeSeries(
            pad_signal,
            name=f'{ifo}:CCSN_SIG',
            channel=f'{ifo}:CCSN_SIG',
            sample_rate=self.sampling_frequency,
            t0=t0,
        )
        return signal_ts

    def parse_waveforms(
            self,
            waveform: str,
    ):
        if waveform == 'lfsg':
            param_generator = self.sg_params
            waveform_generator = self.sg_waveform
        elif waveform == 'hfsg':
            param_generator = self.sg_params
            waveform_generator = self.sg_waveform
        elif waveform == 'bbh':
            param_generator = self.bbh_params
            waveform_generator = self.bbh_waveform
        elif waveform == 'ccsn':
            param_generator = self.ccsn_params
            waveform_generator = self.ccsn_waveform

        return param_generator, waveform_generator

    def generate_waveforms(
            self,
            waveform: str,
            number: int,
            mode: str=None,
            qm_file: str=None,
    ):
        if (waveform == 'lfsg'):
            mode = 'low'
        if ( waveform == 'hfsg'):
            mode = 'high'
        if (waveform == 'ccsn') and (qm_file == None):
            raise RuntimeError("Please provide the h5 file of the simulated quadrupole moments to generate CCSN waveforms.")
        param_generator, waveform_generator = self.parse_waveforms(waveform)
        params = [
            param_generator(mode=mode, qm_file=qm_file)
            for i in range(number)
        ]
        waveforms = []
        for param in params:
            wav = dict.fromkeys(self.ifos)
            for ifo in self.ifos:
                wav[ifo] = waveform_generator(ifo, param)
            
            waveforms.append(wav)
        
        return waveforms, params