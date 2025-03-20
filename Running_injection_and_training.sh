#!/bin/bash
set -e
cd /home/dfredin/repos/gw-anomaly-detection
# mkdir -p /home/app/test_data/glitch /home/app/test_data/noise /home/app/test_data/bbh /home/app/test_data/lfsg /home/app/test_data/hfsg /home/app/test_data/test

# python /home/dfredin/repos/gw-anomaly-detection/get_segments.py
# python /home/dfredin/repos/gw-anomaly-detection/cli.py --ptype test
# python /home/dfredin/repos/gw-anomaly-detection/cli.py --sid 0 --eid 500 --ptype glitch
# python /home/dfredin/repos/gw-anomaly-detection/cli.py --sid 500 --eid 1000 --ptype background
# python /home/dfredin/repos/gw-anomaly-detection/cli.py --sid 1000 --eid 1500 --ptype injection --itype bbh
# python /home/dfredin/repos/gw-anomaly-detection/cli.py --sid 1500 --eid 2000 --ptype injection --itype hfsg
# python /home/dfredin/repos/gw-anomaly-detection/cli.py --sid 2000 --eid 2500 --ptype injection --itype lfsg

python /home/dfredin/repos/gw-anomaly-detection/gasf/src/main.py --newgasf --nbbh 50 --nbg 50 --nglitch 50
python /home/dfredin/repos/gw-anomaly-detection/gasf/src/main.py --train --nbbh 50 --nbg 50 --nglitch 50 --epoch 5 --batch 1