#!/bin/python
import os
import glob
import re
import h5py
import numpy as np
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries

class Process():
    def __init__(
        self,
        ifos: list,
        data_cache: str,
        asd_cache: str,
    ):
        self.ifos = ifos
        self.data_cache = data_cache
        self.asd_cache = asd_cache

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
            end=end,
            format=format,
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
            flow: float=30,
            fhigh: float=1500,
            target_snr_low: float=None,
            target_snr_high: float=None,
    ):
        if (target_snr_low == None) and (target_snr_high == None):
            target_snr_low = 1
            target_snr_high = 1

        rescaled_waveforms, rescaled_snrs = self.rescale(
            waveforms=waveforms,
            target_snr_low=target_snr_low,
            target_snr_high=target_snr_high,
            background_segments=background_segments,
            flow=flow,
            fhigh=fhigh,
        )
        
        injected_ts = []
        for i, waveform in enumerate(rescaled_waveforms):
            inj_ts = dict.fromkeys(self.ifos)
            for ifo in self.ifos:
                segment = background_segments[ifo][i]
                ts = self.get_ts(ifo, segment)
                waveform[ifo].t0 = ts.t0
                inj_ts[ifo] = ts.inject(waveform[ifo])

            injected_ts.append(inj_ts)

        return injected_ts, rescaled_waveforms, rescaled_snrs

    def get_proccessed_data(
            self,
            timeseries: list,
            background_segments: dict,
            flow: float=30,
            fhigh: float=1500,
            resample: float=4096,
            crop_length: float=1,
    ):
        processed_ts = []
        for i, ts in enumerate(timeseries):
            input_ts = dict.fromkeys(self.ifos)
            proc_ts = dict.fromkeys(self.ifos)
            for ifo in self.ifos:
                input_ts[ifo] = ts[ifo].copy()
                segment = background_segments[ifo][i]
                asd = self.get_asd(ifo, segment)
                asd = asd.interpolate(1/input_ts[ifo].duration.value)
                input_ts[ifo] = input_ts[ifo].whiten(asd=asd)
                input_ts[ifo] = input_ts[ifo].bandpass(flow, fhigh)
                input_ts[ifo] = input_ts[ifo].resample(resample)
                input_ts[ifo] = input_ts[ifo].crop(segment.start + crop_length, segment.end - crop_length)
                proc_ts[ifo] = input_ts[ifo]

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
        if not os.path.exists(output_file):
            os.makedirs(output_file)

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

    def get_processed_noise(
            self,
            noise_segments: dict,
            start_id: int=None,
            end_id: int=None,
            flow: float=30,
            fhigh: float=1500,
            resample: float=4096,
            crop_length: float=1,
    ):
        noise_ts = dict.fromkeys(self.ifos)
        for ifo in self.ifos:
            seg_length = len(noise_segments[ifo])
            if start_id == None:
                st = 0
            if end_id > seg_length:
                ed = seg_length
            else:
                st = start_id
                ed = end_id

            proc_ts = []
            for segment in noise_segments[ifo][st:ed]:
                ts = self.get_ts(ifo, segment)
                asd = self.get_asd(ifo, segment)
                asd = asd.interpolate(1/ts.duration.value)
                ts = ts.whiten(asd=asd)
                ts = ts.bandpass(flow, fhigh)
                ts = ts.resample(resample)
                ts = ts.crop(segment.start + crop_length, segment.end - crop_length)
                proc_ts.append(ts)
            
            noise_ts[ifo] = proc_ts

        return noise_ts

    # def write_noise_data(
    #         self,
    #         kind: str,
    #         output_file: str,
    #         processed_noise: dict,
    #         glitch_info_files: dict=None,
    # ):
    #     if kind == "glitch":
    #         ifos = list(glitch_info_files.keys())
    #         output_infos = dict.fromkeys(ifos)
    #         for ifo in ifos:
    #             starts = [ts.t0.value for ts in processed_noise[ifo]]
    #             ends = [ts.t0.value + ts.duration.value for ts in processed_noise[ifo]]

    #             glitch_infos = []
    #             for file in glitch_info_files[ifo]:
    #                 with h5py.File(file, 'r') as f:
    #                     for i, st, ed in enumerate(zip(starts, ends)):
    #                         if f['glitch_info']['time']

    #             output_infos[ifo] = glitch_infos

    #     if kind == "background":
    #         print("background")

    #     return