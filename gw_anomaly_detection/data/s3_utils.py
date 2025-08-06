#!/bin/python
import os
import boto3
# import h5py
# import toml
import numpy as np
# import yaml
import re
import sys

from botocore.config import Config  # <-- Make sure this is importedg  
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

class S3_session(boto3.Session):
    def __init__(
            self,
            s3_config: dict,
    ):
        super().__init__()
        self.config = s3_config
        self.access_key = s3_config['access_key']
        self.secret_key = s3_config['secret_key']
        self.host_base = s3_config['host_base']
        self.bucket = s3_config['bucket']

        self.s3client = self.client(
            service_name='s3',
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            endpoint_url=self.host_base,
            config=Config(
                signature_version='s3v4',
                s3={'addressing_style': 'path'},
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required"
            )
        )

    def ls_bucket(self):
        response = self.s3client.list_buckets()
        for bucket in response['Buckets']:
            print(f"{bucket['Name']}")

    def ls_objects(
            self,
            Bucket:str,
            Prefix=None, 
            Silent=True
        ):
        if Prefix == None:
            response = self.s3client.list_objects_v2(
                Bucket=Bucket,
                Delimiter='/',
            )
            file_list = [f"s3://{Bucket}/" + item['Prefix'] for item in response['CommonPrefixes']]
        else:
            response = self.s3client.list_objects_v2(
                Bucket=Bucket,
                Delimiter='/',
                Prefix=Prefix+"/",
            )
        try:
            if not Silent:
                for prefix in response['CommonPrefixes']:
                    print(f"DIR s3://{Bucket}/{prefix['Prefix']}")
        except KeyError:
            pass
        try:
            if not Silent:
                for content in response['Contents']:
                    print(f"s3://{Bucket}/{content['Key']}")
            file_list = [f"s3://{Bucket}/" + item['Key'] for item in response['Contents']]
        except KeyError:
            pass
        return file_list

    def read_file_dir(self, ifo: str):
        self.prefix = self.config['common_prefix']+ifo
        file_list = self.ls_objects(self.bucket, self.prefix)
        return file_list

    def fetch_data(
            self,
            ifo: str,
            start: int,
            end: int,
            data_cache: str=None,
    ):
        file_list = self.read_file_dir(ifo)
        download_list = []
        for file in file_list:
            match = re.search(r'-(\d+)-(\d+)', file)
            if match:
                file_start = int(match.group(1))
                file_end = int(match.group(1)) + int(match.group(2))
                overlap = not (file_end < start or file_start > end)
                if overlap:
                    download_list.append(file)

        if data_cache==None:
            data_cache = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'cache'))
        if not os.path.exists(data_cache):
            os.mkdir(data_cache)

        for file in download_list:
            if not os.path.exists(data_cache+'/'+ifo):
                os.mkdir(data_cache+'/'+ifo)
            file_name = data_cache+'/'+ifo+'/'+file.split('/')[-1]
            key = file.replace(f"s3://{self.bucket}/", "")
            self.s3client.download_file(
                self.bucket,
                key,
                file_name,
            )
            print(f"s3://{self.bucket}/{key} downloaded to {file_name}")


        return 0

    def download(
            self,
            file: str,
            target_file: str,
    ):
        try:
            key = file.replace(f"s3://{self.bucket}/", "")
            self.s3client.download_file(
                Bucket=self.bucket,
                Key=file,
                Filename=target_file,
            )
            print(f"s3://{self.bucket}/{file} downloaded to {target_file}")
        except Exception as e:
            print(str(e))
            return 1
        return 0

    def upload(
            self,
            file_name: str,
            upload_dir: str,
    ):
        ExtraArgs = {
            'ACL': 'public-read',
        }
        try:
            upload_file_name = file_name.split('/')[-1]
            key = f"{upload_dir}/{upload_file_name}"
            print(f"Uploading {file_name} to s3://{self.bucket}/{key}...")
            self.s3client.upload_file(
                Filename=file_name,
                Bucket=self.bucket,
                Key=key,
                ExtraArgs=ExtraArgs,
            ) 
            print(f"S3 upload complete.\n")
        except Exception as e:
            print(str(e))
            return 1

        return 0 