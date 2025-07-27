#!/bin/bash
set -e
cd .
mkdir -p test_data/glitch test_data/noise test_data/bbh test_data/lfsg test_data/hfsg test_data/test data_cache/H1 data_cache/L1
python get_asds.py
python get_segments.py --samplenum 500
# python cli.py --ptype test
python cli.py --sid 0 --eid 50 --ptype glitch
python cli.py --sid 50 --eid 100 --ptype background

python cli.py --sid 100 --eid 150 --ptype injection --itype bbh
# python cli.py --sid 1500 --eid 2000 --ptype injection --itype hfsg
# python cli.py --sid 2000 --eid 2500 --ptype injection --itype lfsg

# python gw_anomaly_detection/gasf/src/main.py --newgasf --nbbh 500 --nbg 500 --nglitch 500
# python gw_anomaly_detection/gasf/src/main.py --train --nbbh 500 --nbg 500 --nglitch 500 --epoch 15 --batch 32
# python gw_anomaly_detection/gasf/src/main.py --nbbh 500 --nbg 500 --nglitch 500