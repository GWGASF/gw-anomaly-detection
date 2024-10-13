#!/bin/python
import os
import boto3
import h5py
import toml
import numpy as np

# import gwpy
# from gwpy.segments import DataQualityFlag from gwpy.segments import Segment
# from gwpy.segments import SegmentList
# from gwpy.timeseries import TimeSeries
# from gwpy.timeseries import FrequencySeries

# import bilby
# from bilby.core.proir.analytical import Uniform, Cosine, Sine
# from bilby.gw.conversion import bilby_to_lalsimulation_spins

# import lal

# import pycbc
# from pycbc.waveform import get_sgburst_waveform
# from pycbc.waveform import get_td_waveform
# from pycbc.tpyes.timeseries import TimeSeries as pycbcts
# from pycbc.detector import Detector

class s3_session(boto3.Session):
    def set_client(
        self,
        access_key:str,
        secret_key:str,
        host_base:str,
    ):
        self.s3client = self.client(
            service_name='s3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=host_base,
        )

    def ls_bucket(self):
        response = self.s3client.list_buckets()
        for bucket in response['Buckets']:
            print(f"{bucket['Name']}")

    def ls_objects(
            self,
            Bucket:str,
            Prefix=None
        ):
        if Prefix == None:
            response = self.s3client.list_objects_v2(
                Bucket=Bucket,
                Delimiter="/",
            )
        else:
            response = self.s3client.list_objects_v2(
                Bucket=Bucket,
                Delimiter="/",
                Prefix=Prefix,
            )
        try:
            for prefix in response['CommonPrefixes']:
                print(f"DIR s3://{Bucket}/{prefix['Prefix']}")
        except KeyError:
            pass
        try:
            for content in response['Contents']:
                print(f"s3://{Bucket}/{content['Key']}")
        except KeyError:
            pass

def main():
    ACCESS_KEY = "XRIYL1052YCQ125N75UM"
    SECRET_KEY = "XES9mLNbxAWKZC1AUcYFlGd7ByyXXSYw5yB3UrwM"
    HOST_BASE = "https://s3-west.nrp-nautilus.io"
    s3 = s3_session()
    s3.set_client(ACCESS_KEY, SECRET_KEY, HOST_BASE)
    # s3.ls_bucket()
    s3.ls_objects("cchou", "figs/")
    s3.s3client.download_file("cchou", "figs/noise.png", "test/NOISE.png")

if __name__ == "__main__":
    main()