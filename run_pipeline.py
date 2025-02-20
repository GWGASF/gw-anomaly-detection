# if len(sys.argv) < 4 or len(sys.argv) > 5:
#     print("Usage: python script.py <device> <cnt_min> <cnt_max> <(suffix_for_output_file)>")
#     sys.exit(1)
# elif len(sys.argv) == 5:
#     suffix = str(sys.argv[4])
#     print("Suffix set to be {}".format(suffix))
    
# device_pick = int(sys.argv[1])
# cnt_min = int(sys.argv[2])
# cnt_max = int(sys.argv[3])


# # create logger
# logger = logging.getLogger('simple_example')
# logging.basicConfig(filename='../Log/cutscan_cntrange_{}-{}{}.log'.format(cnt_min, cnt_max, suffix), encoding='utf-8')
# logger.setLevel(logging.INFO)

# # create console handler and set level to debug
# ch = logging.StreamHandler()
# ch.setLevel(logging.INFO)

# # create formatter
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# # add formatter to ch
# ch.setFormatter(formatter)

# # add ch to logger
# logger.addHandler(ch)

# logger.info("Start to train the model with cnt in range {}-{}".format(cnt_min, cnt_max))


# # torch.cuda.set_device(0)

# device = torch.device("cuda:{}".format(int(device_pick-1)) if device_pick else "cpu")
# logger.info(f"Using device: {device}")


# dataDir = "../Data"
# outputDir = "../Output"
# modelDir = "../Model"

