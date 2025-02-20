#!/bin/python
import h5py
import numpy as np
from gwpy.segments import Segment
from gwpy.segments import SegmentList

def read_segment_files(
        segment_files: list,
):
    output_segs = []
    for file in segment_files:
        with open(file, 'r') as f:
            for line in f.readlines()[1:]:
                start = float(line.split("\t")[1])
                end = float(line.split("\t")[2])
                seg = Segment(start, end)
                output_segs.append(seg)

    return SegmentList(output_segs)

def whole_segment(
        segment_list: list=None,
        segment_file: list=None,
):
    if segment_list == None and segment_file == None:
        raise RuntimeError("Please provide a segment list or a segment file.")
    if segment_list != None and segment_file != None:
        raise RuntimeError("Please don't use segment list and segment file at the same time.")
    if segment_list != None:
        seglist = segment_list
    if segment_file != None:
        seglist = read_segment_files(segment_file)

    starts = [seg[0] for seg in seglist]
    ends = [seg[1] for seg in seglist]
    interval = (min(starts), max(ends))
    return interval

class SegmentInfo():
    def __init__(
            self,
    ):
        """ The information about the science segments and glitch triggers from Omicron.
        """
        self.get_segs = None

    def load_segment_info(
            self,
            segment_files: list,
            glitch_info_files: list,
            start: float,
            end: float,
            glitch_window_length: float=4,
        ) -> None:
        # Get science segments with start and end time.
        self.loaded_segs = read_segment_files(segment_files)

        self.target_seg = Segment(start, end)
        self.selected_segs = SegmentList([])
        for seg in self.loaded_segs:
            if self.target_seg.intersects(seg):
                self.selected_segs.append(self.target_seg & seg)

        # Get the trigger times of the glitches.
        self.glitch_times = np.array([])
        for file in glitch_info_files:
            with h5py.File(file, 'r') as f:
                glitch_times = f['glitch_info']['time'][:]
                self.glitch_times = np.append(self.glitch_times, glitch_times)

        self.selected_times = []
        for time in self.glitch_times:
            for seg in self.selected_segs:
                if time - seg.start >= glitch_window_length/2 and time - seg.end <= glitch_window_length/2:
                    self.selected_times.append(time)

        return

    def get_segments(
            self,
            kind: str,
            output_file: str=None,
            output_file_format: str=None,
            glitch_window_length: float=4,
            window_length: float=4,
    ):
        """ Get the segments containing the glitches or the segments without glitch.
        """
        output_segs = SegmentList([])
        if kind == "glitch":
            for time in self.selected_times:
                seg = Segment(
                    time - glitch_window_length/2,
                    time + glitch_window_length/2,
                )
                if (seg.end - seg.start) == glitch_window_length:
                    output_segs.append(seg)

        if kind == "background":
            bg_segs = SegmentList([])
            seg_id = 0
            to_new_seg = True
            while len(self.selected_times) != 0:
                seg_st = self.selected_segs[seg_id].start
                seg_ed = self.selected_segs[seg_id].end
                if to_new_seg:
                    previous = seg_st
                time = self.selected_times.pop(0)
                if (time - seg_st >= 0 and time - seg_ed <= 0):
                    bg_segs.append(Segment(previous, time))
                    previous = time
                    to_new_seg = False
                if time - seg_ed > 0:
                    bg_segs.append(Segment(previous, seg_ed))
                    self.selected_times.insert(0, time)
                    to_new_seg = True
                    seg_id += 1

            for seg in bg_segs:
                if (seg.end - seg.start) >= (window_length + glitch_window_length):
                    cropped_seg = Segment(seg.start + glitch_window_length/2, seg.end - glitch_window_length/2)
                    output_segs.append(cropped_seg)

        self.get_segs = output_segs

        # Write the segment file.
        if (output_file != None) and (output_file_format != None):
            try:
                output_segs.write(output_file, format=output_file_format)
                print(f"Segment file written to {output_file}")
            except Exception as e:
                print(str(e))

        return

    def get_glitch_samples(
            self,
            start: float,
            end: float,
            segment_files: list=None,
            output_file: str=None,
            output_file_format: str=None,
    ):
        if (segment_files == None) and (self.get_segs == None):
            raise RuntimeError("Please get segments first or provide a segment file in order to sample from those segments.")
        if segment_files != None:
            try:
                try_segs = read_segment_files(segment_files)
                self.get_segs = try_segs
            except Exception as e:
                print(str(e))

        # Get the segments with the given interval for sampling.
        sample_interval_seg = Segment(start, end)
        filtered_segs = SegmentList([])
        for seg in self.get_segs:
            if sample_interval_seg.intersects(seg):
                filtered_segs.append(sample_interval_seg & seg)

        # Get the sample segments.
        sample_segs = SegmentList([])
        for seg in filtered_segs:
            try:
                sample_st = seg.start
                sample_ed = seg.end
                seg = Segment(sample_st, sample_ed)
                sample_segs.append(seg)
            except Exception as e:
                print(str(e))
                continue

        self.sample_segs = sample_segs

        # Write the segment file.
        if (output_file != None) and (output_file_format != None):
            try:
                sample_segs.write(output_file, format=output_file_format)
                print(f"Sample segments written to {output_file}")
            except Exception as e:
                print(str(e))

        return

    def get_background_samples(
            self,
            number_of_samples: int,
            start: float,
            end: float,
            segment_files: list=None,
            output_file: str=None,
            output_file_format: str=None,
            window_length: float=4,
    ):
        if (segment_files == None) and (self.get_segs == None):
            raise RuntimeError("Please get segments first or provide a segment file in order to sample from those segments.")
        if segment_files != None:
            try:
                try_segs = read_segment_files(segment_files)
                self.get_segs = try_segs
            except Exception as e:
                print(str(e))

        # Get the segments with the given interval for sampling.
        sample_interval_seg = Segment(start, end)
        filtered_segs = SegmentList([])
        for seg in self.get_segs:
            if sample_interval_seg.intersects(seg):
                filtered_segs.append(sample_interval_seg & seg)

        # Get the sample segments.
        sample_segs = SegmentList([])
        seg_ids = np.random.randint(0, len(filtered_segs), number_of_samples)
        for id in seg_ids:
            try:
                sample_st = filtered_segs[id].start
                sample_ed = filtered_segs[id].end - window_length
                sample_t0 = np.random.uniform(sample_st, sample_ed)
                seg = Segment(sample_t0, sample_t0 + window_length)
                sample_segs.append(seg)
            except Exception as e:
                print(str(e))
                continue

        self.sample_segs = sample_segs

        # Write the segment file.
        if (output_file != None) and (output_file_format != None):
            try:
                sample_segs.write(output_file, format=output_file_format)
                print(f"Sample segments written to {output_file}")
            except Exception as e:
                print(str(e))

        return

    def get_test_samples(
            self,
            start: float,
            end: float,
            output_file: str=None,
            output_file_format: str=None,
            window_length: float=4,
            stride_length: float=2,
    ):
        test_interval_seg = Segment(start, end)
        test_selected_segs = SegmentList([])
        
        # Pick out segments from raw science mode files
        for seg in self.selected_segs:
            if test_interval_seg.intersects(seg):
                test_selected_segs.append(test_interval_seg & seg)
        
        # Get the test samples
        test_selected_samples = SegmentList([])
        for seg in test_selected_segs:
            sample_st = seg.start
            seg_ed = seg.end
            while (seg_ed - sample_st) > window_length:
                seg_cached = Segment(sample_st, sample_st + window_length)
                test_selected_samples.append(seg_cached)
                sample_st += stride_length
            seg_cached = Segment(seg_ed - window_length, seg_ed)
            test_selected_samples.append(seg_cached)

        self.test_selected_samples = test_selected_samples

        # Write the segment file.
        if (output_file != None) and (output_file_format != None):
            try:
                test_selected_samples.write(output_file, format=output_file_format)
                print(f"Sample segment file written to {output_file}")
            except Exception as e:
                print(str(e))
        
        return