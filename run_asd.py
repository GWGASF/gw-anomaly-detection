#!/bin/python
import os
import glob
import yaml
from lalframe.utils import frtools
from gwpy.timeseries import TimeSeries
from gw_anomaly_detection.data.fetch_data import fetch_data
from gw_anomaly_detection.data.s3_utils import S3_session

def main():
    # Read config
    with open("./asd_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    ifo = "H1"
    run = "O3a"
    seg_start_id = config[ifo]['seg_start_id']
    seg_end_id = config[ifo]['seg_end_id']
    sample_rate = config['sample_rate']
    data_cache = config['data_cache']
    asd_cache = config['asd_cache']
    min_asd_length = config['asd']['min_asd_length']
    max_asd_length = config['asd']['max_asd_length']
    fftlength = config['asd']['fftlength']
    overlap = config['asd']['overlap']

    # Download segments
    if not os.path.exists(data_cache):
        os.mkdir(data_cache)
    if not os.path.exists(asd_cache):
        os.mkdir(asd_cache)

    s3 = S3_session(config['s3'])
    bucket = config['s3']['bucket']
    download_file = f"Segments/{ifo}-{run}_segments.txt"
    targe_file = f"{data_cache}/{download_file.split('/')[-1]}"
    s3.download(
        file=download_file,
        target_file=targe_file,
    )


    with open(f"{data_cache}/{ifo}-{run}_segments.txt", 'r') as file:
        lines = file.readlines()
        segs = []
        for line in lines:
            info = line.strip().split(' ')
            start = float(info[0])
            end = float(info[1])
            segs.append((start, end))

    # Parse segments
    asd_segs = []
    for seg in segs[seg_start_id:seg_end_id]:
        duration = seg[1] - seg[0]
        if duration >= min_asd_length and duration <= max_asd_length:
            asd_segs.append(seg)
        if duration > max_asd_length:
            for i in range(0, int(duration), max_asd_length):
                asd_seg_start = seg[0]+i
                asd_seg_end = asd_seg_start + max_asd_length
                asd_segs.append((asd_seg_start, asd_seg_end))

            # If the length of the last asd_seg is smaller than min_asd_length, merge it to the previous asd_seg
            if duration%max_asd_length < min_asd_length:
                asd_segs.pop(-1)
            asd_segs[-1] = (asd_segs[-1][0], seg[1])

    # Fetch open data and estimate asds
    for seg in asd_segs:
        fetch_data(
            ifo=ifo,
            start=seg[0],
            end=seg[1],
            sample_rate=sample_rate,
            data_cache=data_cache,
        )

        print(f"Calculating ASD from {seg[0]} to {seg[1]}...")
        gwf_file = glob.glob(f"{data_cache}/{ifo}/*-{seg[0]}-{seg[1]-seg[0]}.gwf")[0]
        channel = frtools.get_channels(gwf_file)[0]
        ts = TimeSeries.read(
            source=gwf_file,
            channel=channel,
        )
        asd = ts.asd(
            fftlength=fftlength,
            overlap=overlap,
            method="median",
        )

        st = int(ts.t0.value)
        du = int(ts.duration.value)
        output_dir = f"{asd_cache}/{ifo}"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_file = f"{output_dir}/{ifo}_asd-{st}-{du}.txt"
        asd.write(
            output_file,
            format="txt",
        )
        print(f"ASD written to {output_file}.")
        os.remove(gwf_file)

    # Upload ASDs to S3 bucket
    asds = glob.glob(f"{output_dir}/*.txt")
    s3 = S3_session(config['s3'])
    upload_dir = f"{config['s3']['upload_dir']}/{run}/{ifo}"
    for file_name in asds:
        s3.upload(
            file_name=file_name,
            upload_dir=upload_dir,
        )

if __name__ == "__main__":
    main()