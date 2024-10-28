#!/bin/python
import h5py
import numpy as np
from gwpy.segments import Segment
from gwpy.segments import SegmentList

class segment_info():
    def __init__(
            self,
            ifo: str,
    ):
        """ The information about the science segments and glitch triggers from Omicron.
        """
        self.get_segs = None

    def load_segment_info(
            self,
            segment_files: list,
            glitch_info_file: str,
            start: float,
            end: float,
        ) -> None:
        # Get science segments with start and end time.
        self.loaded_segs = SegmentList([])
        for file in segment_files:
            self.loaded_segs.extend(SegmentList.read(file, format='segwizard'))

        self.target_seg = Segment(start, end)
        self.selected_segs = SegmentList([])
        for seg in self.loaded_segs:
            if self.target_seg.intersects(seg):
                self.selected_segs.append(self.target_seg & seg)

        self.selected_segs.sort()

        # Get the trigger times of the glitches.
        with h5py.File(glitch_info_file, 'r') as f:
            self.glitch_times = f['glitch_info']['time'][:]

        self.selected_times = []
        for time in self.glitch_times:
            for seg in self.selected_segs:
                if time >= seg.start and time <= seg.end:
                    self.selected_times.append(time)

        self.selected_times.sort()

    def get_segments(
            self,
            kind: str,
            output_file: str=None,
            output_file_format: str=None,
            glitch_window_length: float=4,
            min_window_length: float=4,
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
                if (time - seg_st >= 0 and time - seg_ed <=0):
                    bg_segs.append(Segment(previous, time))
                    previous = time
                    to_new_seg = False
                if time - seg_ed > 0:
                    bg_segs.append(Segment(previous, seg_ed))
                    self.selected_times.insert(0, time)
                    to_new_seg = True
                    seg_id += 1

            for seg in bg_segs:
                if (seg.end - seg.start) >= (min_window_length + glitch_window_length):
                    cropped_seg = Segment(seg.start+2, seg.end-2)
                    output_segs.append(cropped_seg)

        self.get_segs = output_segs

        # Write the segment file.
        if (output_file != None) and (output_file_format != None):
            output_segs.write(output_file, format=output_file_format)
            print(f"Segment file written to {output_file}")

        return output_segs

    def get_samples(
            self,
            number_of_samples: int,
            start: float,
            end: float,
            kind: str=None,
            segment_file: str=None,
            output_file: str=None,
            output_file_format: str=None,
            window_length: float=4,
    ):
        if (kind != 'background') and (kind != 'glitch'):
            raise RuntimeError("Please choose either \'glitch\' or \'background\'.")
        if (segment_file == None) and (self.get_segs == None):
            raise RuntimeError("Please get segments first or provide a segment file in order to sample from those segments.")
        if segment_file != None:
            try:
                try_segs = SegmentList.read(
                    segment_file,
                    format="segwizard",
                )
                try_segs.sort()
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
        if (kind == 'glitch') and (len(filtered_segs) < number_of_samples):
            print(f"Number of glitches between {start} and {end} is {len(filtered_segs)} which is smaller than {number_of_samples}. Changing the number of samples to {len(filtered_segs)}.")
            number_of_samples = len(filtered_segs)

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
            sample_segs.write(output_file, format=output_file_format)
            print(f"Sample segments written to {output_file}")

        return sample_segs