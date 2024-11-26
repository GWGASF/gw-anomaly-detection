#!/bin/python
import yaml
from gw_anomaly_detection.data.s3_utils import S3_session

def main():
    with open("./data_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    s3 = S3_session(config['s3'])
    file_name = "./data_config.yaml"
    upload_dir = "test"
    s3.upload(
        file_name=file_name,
        upload_dir=upload_dir,
    )

if __name__ == "__main__":
    main()