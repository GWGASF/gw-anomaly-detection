#!/bin/bash
set -e
cd .
# mkdir -p test_data/glitch test_data/noise test_data/bbh test_data/lfsg test_data/hfsg test_data/test

# poetry run python get_segments.py --samplenum 5000
# poetry run python cli.py --ptype test
# poetry run python cli.py --sid 0 --eid 500 --ptype glitch
# poetry run python cli.py --sid 500 --eid 1000 --ptype background
poetry run python cli.py --sid 1000 --eid 1500 --ptype injection --itype bbh
# poetry run python cli.py --sid 1500 --eid 2000 --ptype injection --itype hfsg
# poetry run python cli.py --sid 2000 --eid 2500 --ptype injection --itype lfsg

poetry run python gw_anomaly_detection/gasf/src/main.py --newgasf --nbbh 500 --nbg 500 --nglitch 500
poetry run python gw_anomaly_detection/gasf/src/main.py --train --nbbh 500 --nbg 500 --nglitch 500 --epoch 15 --batch 32
poetry run python gw_anomaly_detection/gasf/src/main.py --nbbh 500 --nbg 500 --nglitch 500