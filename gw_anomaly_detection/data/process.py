#!/bin/python
import os
import glob
import re
import h5py
import numpy as np
import time
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries

class Process():
    def __init__(
        self,
        ifos: list,
        data_cache: str,
        asd_cache: str,
        flow: float=30,
        fhigh: float=1500,
        resample: float=4096,
        crop_length: float=1,
    ):
        self.ifos = ifos
        self.data_cache = data_cache
        self.asd_cache = asd_cache
        self.flow = flow
        self.fhigh = fhigh
        self.resample = resample
        self.crop_length = crop_length

    def get_ts(
            self,
            ifo: str,
            segment: tuple,
            format: str="hdf5",
    ):
        source = glob.glob(f"{self.data_cache}/{ifo}/*.{format}")
        source.sort()
        start = segment.start
        end = segment.end
        if format == "hdf5":
            format = "hdf5.gwosc"
        ts = TimeSeries.read(
            source=source,
            start=start,
            end=end
            # format=format
        )
        return ts

    def get_asd(
            self,
            ifo: str,
            segment: tuple,
            format: str="txt",
    ):
        start = segment.start
        parse_time = re.compile(r"-[0-9]*-[0-9]*")
        asds = glob.glob(f"{self.asd_cache}/{ifo}/*.{format}")
        asds.sort()
        for file in asds:
            asd_start = int(parse_time.findall(file)[0].split('-')[1])
            duration = int(parse_time.findall(file)[0].split('-')[2])
            asd_end = asd_start + duration
            if (start >= asd_start) and (start < asd_end):
                asd_file = file
                break

        asd = FrequencySeries.read(asd_file)
        return asd

    def process(
            self,
            ts,
            asd,
            flow: float=30,
            fhigh: float=1500,
            resample: float=4096,
            crop_length: float=1,
    ):
        asd = asd.interpolate(1/ts.duration.value)
        start = ts.t0.value
        end = ts.t0.value + ts.duration.value
        ts = ts.whiten(asd=asd)
        ts = ts.bandpass(flow, fhigh)
        ts = ts.resample(resample)
        ts = ts.crop(start + crop_length, end - crop_length)
        return ts

    def estimate_snr(
            self,
            signal,
            asd,
            flow: float=30,
            fhigh: float=1500,
        ):
        asd = asd.interpolate(1/signal.duration.value)
        sigf = signal.fft()
        sigf = sigf/sigf.df
        snrf = sigf.conj()*sigf/asd**2
        snr = 4*snrf[int(flow/snrf.df.value):int(fhigh/snrf.df.value)].real.sum()*snrf.df
        return snr.to_value()**0.5

    def network_snr(
            self,
            snrs: dict,
        ):
        snrs_list = list(snrs.values())
        return np.sqrt(np.square(np.array(snrs_list)).sum())

    def rescale(
            self,
            waveforms: list,
            target_snr_low: float,
            target_snr_high: float,
            background_segments: dict,
            flow: float=30,
            fhigh: float=1500,
        ):
        rescaled_waveforms = []
        rescaled_snrs = []
        for i, waveform in enumerate(waveforms):
            target_snr = np.random.uniform(target_snr_low, target_snr_high)
            sigs = dict.fromkeys(self.ifos)
            asds = dict.fromkeys(self.ifos)
            snrs = dict.fromkeys([f"snr_{ifo}" for ifo in self.ifos])
            for ifo in self.ifos:
                segment = background_segments[ifo][i]
                sigs[ifo] = waveform[ifo]
                asds[ifo] = self.get_asd(ifo, segment)
                snrs[f"snr_{ifo}"] = self.estimate_snr(sigs[ifo], asds[ifo], flow, fhigh)

            current_snr = self.network_snr(snrs)
            scale_factor = target_snr/current_snr
            # print(f"waveform_id: {i}, snrs: {snrs.items()}, network_snr: {current_snr}, target_snr: {target_snr}")
            for ifo in self.ifos:
                sigs[ifo] = scale_factor * sigs[ifo]
                snrs[f"snr_{ifo}"] = self.estimate_snr(sigs[ifo], asds[ifo], flow, fhigh)

            network_snr = self.network_snr(snrs)
            snrs["network_snr"] = network_snr
            rescaled_waveforms.append(sigs)
            rescaled_snrs.append(snrs)

        return rescaled_waveforms, rescaled_snrs
            
    def inject(
            self,
            waveforms: list,
            background_segments: dict,
            start_id: int,
            end_id: int,
            flow: float=30,
            fhigh: float=1500,
            target_snr_low: float=None,
            target_snr_high: float=None,
    ):
        if (target_snr_low == None) and (target_snr_high == None):
            target_snr_low = 1
            target_snr_high = 1

        selected_segments = dict.fromkeys(self.ifos)
        for ifo in self.ifos:
            selected_segments[ifo] = background_segments[ifo][start_id:end_id]

        rescaled_waveforms, rescaled_snrs = self.rescale(
            waveforms=waveforms,
            target_snr_low=target_snr_low,
            target_snr_high=target_snr_high,
            background_segments=selected_segments,
            flow=flow,
            fhigh=fhigh,
        )
        
        injected_ts = []
        for i, waveform in enumerate(rescaled_waveforms):
            inj_ts = dict.fromkeys(self.ifos)
            for ifo in self.ifos:
                segment = selected_segments[ifo][i]
                ts = self.get_ts(ifo, segment)
                waveform[ifo].t0 = ts.t0
                inj_ts[ifo] = ts.inject(waveform[ifo])

            injected_ts.append(inj_ts)

        return injected_ts, rescaled_waveforms, rescaled_snrs

    def get_proccessed_injection(
            self,
            timeseries: list,
            background_segments: dict,
            start_id: int,
            end_id: int,
    ):
        selected_segments = dict.fromkeys(self.ifos)
        for ifo in self.ifos:
            selected_segments[ifo] = background_segments[ifo][start_id:end_id]

        processed_ts = []
        for i, ts in enumerate(timeseries):
            input_ts = dict.fromkeys(self.ifos)
            proc_ts = dict.fromkeys(self.ifos)
            for ifo in self.ifos:
                input_ts[ifo] = ts[ifo].copy()
                segment = selected_segments[ifo][i]
                asd = self.get_asd(ifo, segment)
                proc_ts[ifo] = self.process(
                    ts=input_ts[ifo],
                    asd=asd,
                    flow=self.flow,
                    fhigh=self.fhigh,
                    resample=self.resample,
                    crop_length=self.crop_length,
                )

            processed_ts.append(proc_ts)

        return processed_ts

    def write_injection_data(
        self,
        output_file: str,
        waveform_parameters: list,
        snrs: list,
        processed_data: list,
        processed_waveforms: list=None,
    ):
        print(f"Writing injection data to {output_file}...")
        with h5py.File(output_file, 'w') as w:
            # Waveform Parameters
            param_names = list(waveform_parameters[0].keys())
            param_formats = []
            for name in param_names:
                if isinstance(waveform_parameters[0][name], float) or isinstance(waveform_parameters[0][name], int):
                    param_formats.append('f8')
                if isinstance(waveform_parameters[0][name], str):
                    param_formats.append(h5py.string_dtype(encoding="ascii"))

            snr_names = list(snrs[0].keys())
            snr_formats = ['f8' for i in range(len(snr_names))]
            names = param_names + snr_names
            formats = param_formats + snr_formats
            param_dtype = np.dtype({'names': names, 'formats': formats})
            param_data = np.array(
                [tuple(param.values()) + tuple(snr.values())
                    for param, snr in zip(waveform_parameters, snrs)],
                    dtype=param_dtype,
            )
            w.create_dataset(
                'waveform_parameters',
                shape=param_data.shape,
                dtype=param_dtype,
                data=param_data,
            )

            # Time Series Data
            ifos = list(processed_data[0].keys())
            proc_data = dict.fromkeys(ifos)
            t0_data = dict.fromkeys(ifos)
            for ifo in ifos:
                proc_data[ifo] = np.array([data[ifo] for data in processed_data])
                t0_data[ifo] = np.array([data[ifo].t0.value for data in processed_data])
                sample_rate = processed_data[0][ifo].sample_rate.value
                proc_dset = w.create_dataset(
                    ifo,
                    shape=proc_data[ifo].shape,
                    dtype=proc_data[ifo].dtype,
                    data=proc_data[ifo],
                )
                proc_dset.attrs['sample_rate'] = sample_rate
                w.create_dataset(
                    f"t0_{ifo}",
                    shape=t0_data[ifo].shape,
                    dtype=t0_data[ifo].dtype,
                    data=t0_data[ifo],
                )

            if processed_waveforms != None:
                ifos = list(processed_waveforms[0].keys())
                waveform_data = dict.fromkeys(ifos)
                for ifo in ifos:
                    waveform_data[ifo] = np.array([waveform[ifo] for waveform in processed_waveforms])
                    channel = str(processed_waveforms[0][ifo].channel)
                    waveform_dset = w.create_dataset(
                        f"waveform_{ifo}",
                        shape=waveform_data[ifo].shape,
                        dtype=waveform_data[ifo].dtype,
                        data=waveform_data[ifo],
                    )
                    waveform_dset.attrs['channel'] = channel

        return

    def get_processed_background(
            self,
            background_segments: dict,
            start_id: int=None,
            end_id: int=None,
    ):
        processed_background = dict.fromkeys(self.ifos)
        for ifo in self.ifos:
            seg_length = len(background_segments[ifo])
            if start_id == None:
                st = 0
            if end_id > seg_length:
                ed = seg_length
            else:
                st = start_id
                ed = end_id
            # Process background.
            proc_bg = []
            for segment in background_segments[ifo][st:ed]:
                ts = self.get_ts(ifo, segment)
                asd = self.get_asd(ifo, segment)
                proc_ts = self.process(
                    ts=ts,
                    asd=asd,
                    flow=self.flow,
                    fhigh=self.fhigh,
                    resample=self.resample,
                    crop_length=self.crop_length,
                )
                proc_bg.append(proc_ts)
            
            processed_background[ifo] = proc_bg

        return processed_background

    def write_background_data(
            self,
            output_file: str,
            processed_background: dict,
    ):
        ifos = list(processed_background.keys())
        print(f"Writing background data to {output_file}...")
        with h5py.File(output_file, 'w') as w:
            for ifo in ifos:
                # Time Series Data
                background_data = np.stack([data.value for data in processed_background[ifo]])
                t0_data = np.stack([data.t0.value for data in processed_background[ifo]])
                sample_rate = 1/processed_background[ifo][0].dt.value
                glitch_dset = w.create_dataset(
                    ifo,
                    shape=background_data.shape,
                    dtype=background_data.dtype,
                    data=background_data,
                )
                glitch_dset.attrs['sample_rate'] = sample_rate
                glitch_dset.attrs['channel'] = f"{ifo}:BACKGROUND_NOISE"
                w.create_dataset(
                    f"t0_{ifo}",
                    shape=t0_data.shape,
                    dtype=t0_data.dtype,
                    data=t0_data,
                )

        return

    def get_processed_glitch(
            self,
            glitch_segments: dict,
            start_id: int=None,
            end_id: int=None,
            glitch_info_files: dict=None,
            glitch_window_length: float=4,
    ):
        glitch_infos = dict.fromkeys(self.ifos)
        processed_glitch = dict.fromkeys(self.ifos)
        for ifo in self.ifos:
            seg_length = len(glitch_segments[ifo])
            if start_id == None:
                st = 0
            if end_id > seg_length:
                st = start_id
                ed = seg_length
            else:
                st = start_id
                ed = end_id
            # Process glitch.
            proc_glitch = []
            for segment in glitch_segments[ifo][st:ed]:
                ts = self.get_ts(ifo, segment)
                asd = self.get_asd(ifo, segment)
                proc_ts = self.process(
                    ts=ts,
                    asd=asd,
                    flow=self.flow,
                    fhigh=self.fhigh,
                    resample=self.resample,
                    crop_length=self.crop_length,
                )
                proc_glitch.append(proc_ts)
            
            processed_glitch[ifo] = proc_glitch
            # Get glitch info.
            starts = [seg[0] for seg in glitch_segments[ifo][st:ed]]
            ends = [seg[1] for seg in glitch_segments[ifo][st:ed]]
            interval = (min(starts), max(ends))
            select_infos = []
            for file in glitch_info_files[ifo]:
                with h5py.File(file, 'r') as f:
                    infos = f['glitch_info']
                    for info in infos:
                        trig_time = info['time']
                        after_start =  trig_time - glitch_window_length/2 >= interval[0]
                        before_end =  trig_time + glitch_window_length/2 <= interval[1]
                        if after_start and before_end:
                            select_infos.append(info)

            glitch_infos[ifo] = select_infos

        return processed_glitch, glitch_infos

    def write_glitch_data(
            self,
            output_file: str,
            processed_glitch: dict,
            glitch_infos: dict,
    ):
        ifos = list(processed_glitch.keys())
        print(f"Writing glitch data to {output_file}...")
        with h5py.File(output_file, 'w') as w:
            for ifo in ifos:
                # Glitch Info
                info_names = list(glitch_infos[ifo][0].dtype.names)
                info_formats = []
                for name in info_names:
                    if isinstance(glitch_infos[ifo][0][name], float) or isinstance(glitch_infos[ifo][0][name], int):
                            info_formats.append('f8')
                    if isinstance(glitch_infos[ifo][0][name], bytes):
                        info_formats.append(h5py.string_dtype(encoding="ascii"))
            
                info_dtype = np.dtype({'names': info_names, 'formats': info_formats})
                info_data = np.stack(glitch_infos[ifo])
                w.create_dataset(
                    f"{ifo}_glitch_info",
                    shape=info_data.shape,
                    dtype=info_data.dtype,
                    data=info_data,
                )
                # Time Series Data
                glitch_data = np.stack([data.value for data in processed_glitch[ifo]])
                t0_data = np.stack([data.t0.value for data in processed_glitch[ifo]])
                sample_rate = 1/processed_glitch[ifo][0].dt.value
                glitch_dset = w.create_dataset(
                    ifo,
                    shape=glitch_data.shape,
                    dtype=glitch_data.dtype,
                    data=glitch_data,
                )
                glitch_dset.attrs['sample_rate'] = sample_rate
                glitch_dset.attrs['channel'] = f"{ifo}:GLITCH"
                w.create_dataset(
                    f"t0_{ifo}",
                    shape=t0_data.shape,
                    dtype=t0_data.dtype,
                    data=t0_data,
                )

        return
