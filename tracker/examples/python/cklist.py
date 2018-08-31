import argparse
import glob
import os

ckpt_dir = "/home/jaehyuk/code/github/vot-toolkit/tracker/examples/python/checkpoints_temp"
parser = argparse.ArgumentParser()
parser.add_argument("-c", help="changed name ... ex) DETVID_adj_lr1e567_", type=str)
args = parser.parse_args()
change_name = args.c

all_ckpt_meta = glob.glob(os.path.join(ckpt_dir, '*.meta'))

num = []
for ckpt_meta in all_ckpt_meta:
    num.append(int(ckpt_meta.split('-')[2].split('.')[0]))

# max_num = max(num)
# ckpt = os.path.join(ckpt_dir, 'checkpoint.ckpt-' + str(max_num))

num = sorted(num, reverse=True)
cklist = []
for ckpt_num in num:
    path = os.path.join(ckpt_dir, 'checkpoint.ckpt-' + str(ckpt_num))
    cklist.append(path)

print cklist

test_list = []

for ckpt_num in num:
    path = os.path.join(change_name + str(ckpt_num))
    test_list.append(path)

print test_list
