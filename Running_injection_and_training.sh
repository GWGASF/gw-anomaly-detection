#!/bin/bash
set -e

mkdir /home/app/test_data/glitch /home/app/test_data/noise /home/app/test_data/bbh /home/app/test_data/lfsg /home/app/test_data/hfsg /home/app/test_data/test

python /opt/repo/gw-anomaly-detection/get_segments.py
python /opt/repo/gw-anomaly-detection/cli.py --sid 0 --eid 500 --pype 
python /opt/repo/gw-anomaly-detection/cli.py --sid 500 --eid 1000 --ptype 
python /opt/repo/gw-anomaly-detection/cli.py --sid 1000 --eid 1500 --ptype --itype 
python /opt/repo/gw-anomaly-detection/cli.py --sid 1500 --eid 2000 --ptype --itype 
python /opt/repo/gw-anomaly-detection/cli.py --sid 2000 --eid 2500 --ptype --itype 

python /opt/repo/gw-anomaly-detection/run_pipeline.py