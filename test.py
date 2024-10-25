#!/bin/python
from gw_anomaly_detection.data.segments import get_background_segments

def main():
    ifo = 'L1'
    segment_file = f"/home/chiajui/gw-anomaly-detection/segments/{ifo}_science_segs.segwizard"
    glitch_info_file = f"/home/chiajui/gw-anomaly-detection/glitch_info/O3a/{ifo}_glitch_info.hdf5"
    segs = get_background_segments(
        ifo=ifo,
        segment_file=segment_file,
        glitch_info_file=glitch_info_file,
    )
    print(segs.shape)

if __name__ == "__main__":
    main()