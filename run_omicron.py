#!/bin/python
import yaml
from gw_anomaly_detection.data.fetch_data import fetch_data

def main():
    with open("./omicron_config.yaml", "r") as file:
        omicron_config = yaml.safe_load(file)
    with open("./data_config.yaml", "r") as file:
        data_config = yaml.safe_load(file)

    sample_rate = omicron_config['omicron']['sample_rate']
    format = omicron_config['omicron']['format']
    ifo = "H1"
    start = omicron_config['omicron'][ifo]['start']
    end = omicron_config['omicron'][ifo]['end']
    data_cache = f"{data_config['data']['data_cache']}"
    fetch_data(
        ifo=ifo,
        start=start,
        end=end,
        sample_rate=sample_rate,
        format=format,
        data_cache=data_cache,
    )


if __name__ == "__main__":
    main()