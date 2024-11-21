#!/bin/python
from gw_anomaly_detection.data.segments import SegmentInfo

def main():
    ifos = ['H1', 'L1']
    for ifo in ifos:
        segment_files = [
            f"/home/chiajui/gw-anomaly-detection/segments/O3a/{ifo}-science_segs.segwizard",
        ]
        glitch_info_file = f"/home/chiajui/gw-anomaly-detection/glitch_info/O3a/{ifo}-glitch_info.hdf5"
        interval = {
            'H1': (1238166018, 1238856499),
            'L1': (1238166018, 1238772909),
        }
        start, end = interval[ifo]

        SegInfo = SegmentInfo()
        kinds = ['glitch', 'background']
        # Create segment files of the glitches and background noise.
        SegInfo.load_segment_info(
            segment_files=segment_files,
            glitch_info_file=glitch_info_file,
            start=start,
            end=end,
        )
        for kind in kinds:
            output_file = f"segments/O3a/{ifo}-{kind}_segs-{start}-{int(end-start)}.segwizard"
            segs = SegInfo.get_segments(
                kind=kind,
                output_file=output_file,
                output_file_format='segwizard',
                glitch_window_length=4,
                min_window_length=4,
            )

    # Get glitch samples.
    kind = "glitch"
    sample_intervals = {
        'H1': (1238172987, 1238196793),
        'L1': (1238205073, 1238228885),
    }
    for ifo in ifos:
        start, end = sample_intervals[ifo]
        seg_file_st, seg_file_ed = interval[ifo]
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
    kind = "background"
    number_of_samples = 60000
    sample_intervals = {
        'H1': (1238172987, 1238196793),
        'L1': (1238205073, 1238228885),
    }
    for ifo in ifos:
        start, end = sample_intervals[ifo]
        seg_file_st, seg_file_ed = interval[ifo]
        segment_files = [
            f"segments/O3a/{ifo}-{kind}_segs-{seg_file_st}-{seg_file_ed-seg_file_st}.segwizard",
        ]
        output_file = f"segments/O3a/{ifo}-{kind}_samples-{start}-{end-start}.segwizard"

        sample_segs = SegInfo.get_background_samples(
            number_of_samples=number_of_samples,
            start=start,
            end=end,
            segment_files=segment_files,
            output_file=output_file,
            output_file_format='segwizard',
            window_length=4,
        )

if __name__ == "__main__":
    main()