import numpy as np
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=256, help='Number of samples to extract')
args = parser.parse_args()

N = args.N

games = ['bigfish']

for game in games:
    paths = glob.glob(f"expert_data/{game}/train/*.npz")
    buf = []
    for p in paths:
        d = np.load(p)
        buf.append((d["obs"], d["ta"]))
    obs, ta = map(np.concatenate, zip(*buf))
    idx = np.random.choice(len(obs)-1, N, replace=False)
    np.savez(f"offline_decoder_data/{game}_{N}.npz",
             obs=obs[idx], ta=ta[idx])