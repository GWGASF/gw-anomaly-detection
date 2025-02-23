#!/bin/bash
set -e
cd /opt/repo/gw-anomaly-detection
mkdir -p /home/app/test_data/glitch /home/app/test_data/noise /home/app/test_data/bbh /home/app/test_data/lfsg /home/app/test_data/hfsg /home/app/test_data/test

python /opt/repo/gw-anomaly-detection/get_segments.py
python /opt/repo/gw-anomaly-detection/cli.py --ptype test
python /opt/repo/gw-anomaly-detection/cli.py --sid 0 --eid 500 --ptype glitch
python /opt/repo/gw-anomaly-detection/cli.py --sid 500 --eid 1000 --ptype background
python /opt/repo/gw-anomaly-detection/cli.py --sid 1000 --eid 1500 --ptype injection --itype bbh
python /opt/repo/gw-anomaly-detection/cli.py --sid 1500 --eid 2000 --ptype injection --itype hfsg
python /opt/repo/gw-anomaly-detection/cli.py --sid 2000 --eid 2500 --ptype injection --itype lfsg

python /opt/repo/gw-anomaly-detection/run_pipeline.py