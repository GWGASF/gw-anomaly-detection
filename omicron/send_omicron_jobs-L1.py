#!/bin/python3

import glob
import os
import multiprocessing

ifo = "L1"
# ifo = "H1"
job_repo = f"/home/chiajui.chou/GW-anomaly-detection/O3a/{ifo}"
idstart = 0
idend = 5

job_folders = glob.glob(f"{job_repo}/*")
job_folders = sorted(job_folders)
print(f"There are {len(job_folders)} jobs found in {job_repo}")

def send(i, job_folder):
    os.chdir(job_folder)
    os.system(f"bash {job_folder}/run_omicron.sh")
    

def main():
    iter_through = [(i, folder) for i, folder in enumerate(job_folders[idstart:idend])]
    pool = multiprocessing.Pool(5)
    pool.starmap_async(send, iter_through)

    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
