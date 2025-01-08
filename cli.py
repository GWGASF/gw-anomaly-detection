#!/bin/python
import yaml
import os
import multiprocessing
import argparse
from gw_anomaly_detection.data.segments import read_segment_files
from gw_anomaly_detection.data.segments import whole_segment
from gw_anomaly_detection.data.s3_utils import S3_session
from gw_anomaly_detection.data.waveforms import Waveforms
from gw_anomaly_detection.data.process import Process


def full_process(config, s3):
    ifos = config['data']['ifos']
    kind = config['data']['kind']
    total_interval = config['data']['total_interval']
    data_cache = config['data']['data_cache']
    asd_cache = config['data']['asd_cache']
    flow = config['data']['processing']['flow']
    fhigh = config['data']['processing']['fhigh']
    resample = config['data']['processing']['resample']
    crop_length = config['data']['processing']['crop_length']

    # parser.add_argument('--flow', type=int, default = config['data']['processing']['flow'])
    # parser.add_argument('--fhigh', type=int, default = config['data']['processing']['fhigh'])

    # args = parser.parse_args()
    
    # flow = args.flow
    # fhigh = args.fhigh

    # Processing glitch.
    if kind == "glitch":
        print(f"Glitch.")
        ifos = config['data']['glitch']['ifos']
        segment_files = config['data']['glitch']['segment_files']
        glitch_segments = dict.fromkeys(ifos)
        for ifo in ifos:
            glitch_segments[ifo] = read_segment_files(segment_files[ifo])
            interval = whole_segment(segment_file=segment_files[ifo])
            print(f"Number of glitch segments from {ifo}: {len(glitch_segments[ifo])}, interval: {interval}.")

        start_id = config['data']['glitch']['start_id']
        end_id = config['data']['glitch']['end_id']
        glitch_info_files = config['data']['glitch']['glitch_info_files']
        glitch_window_length = config['data']['glitch']['glitch_window_length']
        output_file = config['data']['glitch']['output_file']
        # Processing data.
        proc = Process(
            ifos=ifos,
            data_cache=data_cache,
            asd_cache=asd_cache,
            flow=flow,
            fhigh=fhigh,
            resample=resample,
            crop_length=crop_length,
        )
        processed_glitch, glitch_infos = proc.get_processed_glitch(
            glitch_segments=glitch_segments,
            start_id=start_id,
            end_id=end_id,
            glitch_info_files=glitch_info_files,
            glitch_window_length=glitch_window_length,
        )
        # Write data to hdf5 files.
        proc.write_glitch_data(
            output_file=output_file,
            processed_glitch=processed_glitch,
            glitch_infos=glitch_infos,
        )

    # Processing background noise
    if kind == "background":
        print("Background.")
        ifos = config['data']['background']['ifos']
        segment_files = config['data']['background']['segment_files']
        background_segments = dict.fromkeys(ifos)
        for ifo in ifos:
            background_segments[ifo] = read_segment_files(segment_files[ifo])
            interval = whole_segment(segment_file=segment_files[ifo])
            print(f"Number of background segments from {ifo}: {len(background_segments[ifo])}, interval: {interval}.")

        start_id = config['data']['glitch']['start_id']
        end_id = config['data']['glitch']['end_id']

        output_file_suffix = "_ids_{}-{}.hdf5".format(str(start_id), str(end_id))

        window_length = config['data']['background']['window_length']
        output_file = os.path.join(config['data']['background']['output_file_path'], config['data']['background']['output_file']+output_file_suffix)

        # Processing data.
        proc = Process(
            ifos=ifos,
            data_cache=data_cache,
            asd_cache=asd_cache,
            flow=flow,
            fhigh=fhigh,
            resample=resample,
            crop_length=crop_length,
        )
        processed_background = proc.get_processed_background(
            background_segments=background_segments,
            start_id=start_id,
            end_id=end_id,
        )
        # Write data to hdf5 files.
        proc.write_background_data(
            output_file=output_file,
            processed_background=processed_background,
        )

    # Processing injection data.
    if kind == "injection":
        print(f"Injection.")
        ifos = config['data']['injection']['ifos']
        segment_files = config['data']['injection']['segment_files']
        background_segments = dict.fromkeys(ifos)
        for ifo in ifos:
            background_segments[ifo] = read_segment_files(segment_files[ifo])
            interval = whole_segment(segment_file=segment_files[ifo])
            print(f"Number of background segments from {ifo}: {len(background_segments[ifo])}, interval: {interval}.")

        start_id = config['data']['injection']['start_id']
        end_id = config['data']['injection']['end_id']
        waveform = config['data']['injection']['waveform']
        qm_file = config['data']['injection']['qm_file']
        sampling_frequency = config['data']['injection']['sampling_frequency']
        target_snr_low = config['data']['injection']['target_snr_low']
        target_snr_high = config['data']['injection']['target_snr_high']
        keep_waveform = config['data']['injection']['keep_waveform']
        output_file = config['data']['injection']['output_file']
        window_length = config['data']['injection']['window_length']

        # Generating Waveform.
        number = end_id - start_id
        wav = Waveforms(
            ifos=ifos,
            window_length=window_length,
            sampling_frequency=sampling_frequency,
        )
        waveforms, params = wav.generate_waveforms(
            waveform,
            number,
            qm_file=qm_file,
        )
        # Rescaling and Injection.
        proc = Process(
            ifos=ifos,
            data_cache=data_cache,
            asd_cache=asd_cache,
            flow=flow,
            fhigh=fhigh,
            resample=resample,
            crop_length=crop_length,
        )
        injected_ts, rescaled_waveforms, snrs = proc.inject(
            waveforms=waveforms,
            target_snr_low=target_snr_low,
            target_snr_high=target_snr_high,
            background_segments=background_segments,
            start_id=start_id,
            end_id=end_id,
        )
        # Processing.
        timeseries = injected_ts
        processed_data = proc.get_proccessed_injection(
            timeseries=timeseries,
            background_segments=background_segments,
            start_id=start_id,
            end_id=end_id,
        )
        # Write data to hdf5 files.
        if keep_waveform:
            timeseries = rescaled_waveforms
            processed_waveforms = proc.get_proccessed_injection(
                timeseries=timeseries,
                background_segments=background_segments,
                start_id=start_id,
                end_id=end_id,
            )
        else:
            processed_waveforms = None
        proc.write_injection_data(
            output_file=output_file,
            waveform_parameters=params,
            snrs=snrs,
            processed_data=processed_data,
            processed_waveforms=processed_waveforms,
        )

    # Uploading processed data to s3 buckets.
    file_name = output_file
    upload_dir = config['s3']['upload_dir']
    s3.upload(
        file_name=file_name,
        upload_dir=upload_dir,
    )

    # Gathering data on s3 buckets.

if __name__ == "__main__":
    
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    os.chdir(script_directory)

    # Loading data_config.yaml
    with open("./data_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # start_id = config['data']['background']['start_id']
    # end_id = config['data']['background']['end_id']

    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', type=int, default = config['data']['background']['start_id'])
    parser.add_argument('--eid', type=int, default = config['data']['background']['end_id'])

    args = parser.parse_args()
    config['data']['background']['start_id'] = args.sid
    config['data']['background']['end_id'] = args.eid

    print(config['data']['background']['end_id'])

    # Loading strain and asd into data cache.
    background_interval = config['data']['background']['sample_interval']
    data_cache = config['data']['data_cache']
    ifos = config['data']['ifos']
    print("Downloading data from s3 bucket...")
    s3 = S3_session(config['s3'])
    for ifo in ifos:
        s3.fetch_data(
            ifo=ifo,
            start=background_interval[ifo]['start'],
            end=background_interval[ifo]['end'],
            data_cache=data_cache,
        )

   
    processes = []
    num_processes = config['Process_num']

    total_id_num = config['data']['background']['end_id'] - config['data']['background']['start_id']
    interval = total_id_num // num_processes
    # remainder = total_id_num % num_processes
    # For simplicity, assume remainder equals to 0
    assert total_id_num % num_processes == 0


    for i in range(num_processes):
        config_cached = config.copy()
        config_cached['data']['background']['start_id'] = config['data']['background']['start_id'] + i * interval
        config_cached['data']['background']['end_id'] = config['data']['background']['start_id'] + (i + 1) * interval
        p = multiprocessing.Process(target=full_process, args = (config_cached, s3))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
