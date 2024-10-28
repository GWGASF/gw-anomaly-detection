#!/bin/python
from gw_anomaly_detection.data.segments import segment_info

def main():
    ifos = ['H1', 'L1']
    ifo = ifos[1]
    segment_files = [
        f"/home/chiajui/gw-anomaly-detection/segments/O3a/{ifo}-science_segs.segwizard",
    ]
    glitch_info_file = f"/home/chiajui/gw-anomaly-detection/glitch_info/O3a/{ifo}-glitch_info.hdf5"
    data_cache = f"O3_data{ifo}"
    interval = {
        'H1': (1238166018, 1238856499),
        'L1': (1238166018, 1238772909),
    }
    start = interval[ifo][0]
    end = interval[ifo][1]

    SegInfo = segment_info(ifo=ifo)
    
    kinds = ['glitch', 'background']
    # Create segment files of the glitches and background noise.
    # SegInfo.load_segment_info(
    #     segment_files=segment_files,
    #     glitch_info_file=glitch_info_file,
    #     start=start,
    #     end=end,
    # )
    # for kind in kinds:
    #     output_file = f"segments/O3a/{ifo}-{kind}_segs-{start}-{int(end-start)}.segwizard"
    #     segs = SegInfo.get_segments(
    #         kind=kind,
    #         output_file=output_file,
    #         output_file_format='segwizard',
    #         glitch_window_length=4,
    #         min_window_length=4,
    #     )

    # Get samples.
    number_of_samples = 60000
    # start, end = 1238172987, 1238196793 # H1
    start, end = 1238205073, 1238228885 # L1
    kind = kinds[1]
    seg_file_st = interval[ifo][0]
    seg_file_ed = interval[ifo][1]
    segment_file = f"segments/O3a/{ifo}-{kind}_segs-{seg_file_st}-{seg_file_ed-seg_file_st}.segwizard"
    output_file = f"segments/O3a/{ifo}-{kind}_samples-{start}-{end-start}.segwizard"
    
    sample_segs = SegInfo.get_samples(
        number_of_samples=number_of_samples,
        start=start,
        end=end,
        kind=kind,
        segment_file=segment_file,
        output_file=output_file,
        output_file_format='segwizard',
        window_length=4,
    )

if __name__ == "__main__":
    main()