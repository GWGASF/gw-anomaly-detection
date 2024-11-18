#!/bin/python
import glob
import re
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
            snrs = dict.fromkeys(self.ifos)
            for ifo in self.ifos:
                segment = background_segments[ifo][i]
                sigs[ifo] = waveform[ifo]
                asds[ifo] = self.get_asd(ifo, segment)
                snrs[ifo] = self.estimate_snr(sigs[ifo], asds[ifo], flow, fhigh)

            current_snr = self.network_snr(snrs)
            scale_factor = target_snr/current_snr
            # print(f"waveform_id: {i}, snrs: {snrs.items()}, network_snr: {current_snr}, target_snr: {target_snr}")
            for ifo in self.ifos:
                sigs[ifo] = scale_factor * sigs[ifo]
                snrs[ifo] = self.estimate_snr(sigs[ifo], asds[ifo], flow, fhigh)

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

        return injected_ts, rescaled_snrs

    def get_proccessed_data(
            self,
            timeseries: list,
            background_segments: dict,
            flow: float=30,
            fhigh: float=1500,
            resample: float=4096,
            crop_length: float=1,
            method: int=1,
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
                if method == 1:
                    input_ts[ifo] = input_ts[ifo].resample(resample)
                    input_ts[ifo] = input_ts[ifo].crop(segment.start + crop_length, segment.end - crop_length)
                if method == 2:
                    input_ts[ifo] = input_ts[ifo].crop(segment.start + crop_length, segment.end - crop_length)
                    input_ts[ifo] = input_ts[ifo].resample(resample)
                proc_ts[ifo] = input_ts[ifo]

            processed_ts.append(proc_ts)

        return processed_ts
