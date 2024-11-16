#!/bin/python
import glob
import numpy as np
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries

class Process():
    def __init__(
        self,
        ifo: str,
    ):
        self.ifo = ifo

    def get_ts(
            self,
            data_cache: str,
            segment: tuple,
            format: str="hdf5",
    ):
        source = glob.glob(f"{data_cache}/{self.ifo}/*.{format}")
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
            segment: tuple,
            asd_cache: str,
    ):
        asd = 'asd'
        return asd

    def estimate_snr(
            self,
            signal,
            asd,
            flow=30,
            fhigh=1500
        ):
        sigf = signal.fft()
        sigf = sigf/sigf.df
        snrf = sigf.conj()*sigf/asd**2
        snr = 4*snrf[int(flow/snrf.df.value):int(fhigh/snrf.df.value)].real.sum()*snrf.df
        return snr.to_value()**0.5

    def network_snr(
            self,
            snrs,
        ):
        return np.sqrt(np.square(np.array(snrs)).sum())

    def rescale_snr(
            self,
            target_snr,
            signals,
            asds,
            flow=30,
            fhigh=1500
        ):
        ifos = list(signals.keys())
        ifo_snrs = dict.fromkeys(ifos) 
        re_sigs = dict.fromkeys(ifos) 
        re_ifo_snrs = dict.fromkeys(ifos) 
        for ifo in ifos:
            ifo_snrs[ifo] = self.estimate_snr(signals[ifo], asds[ifo], flow, fhigh)

        current_snr = self.network_snr(list(ifo_snrs.values()))
        scale_factor = target_snr/current_snr
        for ifo in ifos:
            re_sigs[ifo] = scale_factor * signals[ifo]
            re_ifo_snrs[ifo] = scale_factor * ifo_snrs[ifo]
        return re_sigs, re_ifo_snrs

    def inject(
            self,
    ):

        return

    def get_proccessed_data(
            self,
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
