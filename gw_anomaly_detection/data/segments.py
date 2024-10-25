#!/bin/python
import h5py
from gwpy.segments import Segment
from gwpy.segments import SegmentList

def get_background_segments(
        ifo: str,
        segment_files: list,
        glitch_info_file: str,
        data_cache: str,
        start: float,
        end: float,
        glitch_window_length: float=4,
        min_window_length: float=4,
):
    # Get science segments with start and end time.
    loaded_segs = SegmentList([])
    for file in segment_files:
        loaded_segs.extend(SegmentList.read(file, format='segwizard'))

    target_seg = Segment(start, end)
    selected_segs = SegmentList([])
    for seg in loaded_segs:
        if target_seg.intersects(seg):
            selected_segs.append(target_seg & seg)

    selected_segs.sort()

    # Get the trigger times of the glitches.
    with h5py.File(glitch_info_file, 'r') as f:
        glitch_times = f['glitch_info']['time'][:]

    selected_times = []
    for time in glitch_times:
        for seg in selected_segs:
            if time >= seg.start and time <= seg.end:
                selected_times.append(time)

    selected_times.sort()

    # Get background segments without the glitches.
    bg_segs = SegmentList([])
    seg_id = 0
    to_new_seg = True
    while len(selected_times) != 0:
        seg_st = selected_segs[seg_id].start
        seg_ed = selected_segs[seg_id].end
        if to_new_seg:
            previous = seg_st
        time = selected_times.pop(0)
        if (time - seg_st >= 0 and time - seg_ed <=0):
            bg_segs.append(Segment(previous, time))
            previous = time
            to_new_seg = False
        if time - seg_ed > 0:
            bg_segs.append(Segment(previous, seg_ed))
            selected_times.insert(0, time)
            to_new_seg = True
            seg_id += 1

    output_segs = SegmentList([])
    for seg in bg_segs:
        if (seg.end - seg.start) >= (min_window_length + glitch_window_length):
            cropped_seg = Segment(seg.start+2, seg.end-2)
            output_segs.append(cropped_seg)

    output_segs.write("./bg_segs.segwizard", format="segwizard")

    return output_segs