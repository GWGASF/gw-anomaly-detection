#!/bin/python
from gw_anomaly_detection.data.segments import get_background_segments

def main():
    ifos = ['H1', 'L1']
    ifo = ifos[0]
    segment_files = [
        f"/home/chiajui/gw-anomaly-detection/segments/O3a/{ifo}_science_segs.segwizard",
    ]
    glitch_info_file = f"/home/chiajui/gw-anomaly-detection/glitch_info/O3a/{ifo}_glitch_info.hdf5"
    data_cache = f"O3_data{ifo}"
    interval = {
        'H1': (1238166018, 1238856499),
        'L1': (1238166018, 1238772909),
    }
    start = interval[ifo][0]
    end = interval[ifo][1]

    segs = get_background_segments(
        ifo=ifo,
        segment_files=segment_files,
        glitch_info_file=glitch_info_file,
        data_cache=data_cache,
        start=start,
        end=end,
        glitch_window_length=4,
    )
    print(len(segs))
    print(segs[0])

if __name__ == "__main__":
    main()