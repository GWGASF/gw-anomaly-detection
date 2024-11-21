#!/bin/python
import yaml
from gw_anomaly_detection.data.segments import SegmentInfo

def generate_samples(
        kind: str,
        config: dict,
        segment_files: dict,
):
    ifos = config['data'][kind]['ifos']
    sample_interval = config['data'][kind]['sample_interval']
    number_of_samples = config['data'][kind]['number_of_samples']
    window_length = config['data']['processing']['window_length']
    for ifo in ifos:
        start = sample_interval[ifo]['start']
        end = sample_interval[ifo]['end']
        seg_file_st = config['data']['total_interval'][ifo]['start']
        seg_file_ed = config['data']['total_interval'][ifo]['end']
        segment_files = [
            f"segments/O3a/{ifo}-background_segs-{int(seg_file_st)}-{int(seg_file_ed-seg_file_st)}.segwizard"
        ]
        output_file = f"segments/O3a/{ifo}-{kind}_samples_test-{int(start)}-{int(end-start)}.segwizard"

        SegInfo = SegmentInfo()
        SegInfo.get_background_samples(
            number_of_samples=number_of_samples,
            start=start,
            end=end,
            segment_files=segment_files,
            output_file=output_file,
            output_file_format='segwizard',
            window_length=window_length,
        )


def main():
    # Loading data_config.yaml
    with open("./data_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    ifos = config['data']['ifos']
    segment_files = config['data']['segment_files']
    glitch_info_files = config['data']['glitch']['glitch_info_files']
    total_interval = config['data']['total_interval']
    glitch_window_length = config['data']['glitch']['glitch_window_length']
    min_window_length = config['data']['background']['min_window_length']
    window_length = min_window_length

    SegInfo = SegmentInfo()
    for ifo in ifos:
        start = total_interval[ifo]['start']
        end = total_interval[ifo]['end']
        kinds = ['glitch', 'background']
        # Create segment files of the glitches and background noise.
        SegInfo.load_segment_info(
            segment_files=segment_files[ifo],
            glitch_info_files=glitch_info_files[ifo],
            start=start,
            end=end,
            glitch_window_length=glitch_window_length,
        )
        for kind in kinds:
            output_file = f"segments/O3a/{ifo}-{kind}_segs-{int(start)}-{int(end-start)}.segwizard"
            segs = SegInfo.get_segments(
                kind=kind,
                output_file=output_file,
                output_file_format='segwizard',
                glitch_window_length=glitch_window_length,
                min_window_length=min_window_length,
            )

    # Get glitch samples.
    kind = "glitch"
    sample_interval = config['data']['glitch']['sample_interval']
    for ifo in ifos:
        start = sample_interval[ifo]['start']
        end = sample_interval[ifo]['end']
        seg_file_st = total_interval[ifo]['start']
        seg_file_ed = total_interval[ifo]['end']
        segment_files = [
            f"segments/O3a/{ifo}-{kind}_segs-{seg_file_st}-{seg_file_ed-seg_file_st}.segwizard",
        ]
        output_file = f"segments/O3a/{ifo}-{kind}_samples-{start}-{end-start}.segwizard"

        sample_segs = SegInfo.get_glitch_samples(
            start=start,
            end=end,
            segment_files=segment_files,
            output_file=output_file,
            output_file_format='segwizard',
        )

    # Get background samples.
    generate_samples(
        kind="background",
        config=config,
        segment_files=segment_files,
    )

    # Get injection samples.
    generate_samples(
        kind="injection",
        config=config,
        segment_files=segment_files,
    )

if __name__ == "__main__":
    main()