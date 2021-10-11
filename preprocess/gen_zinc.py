import glob
import random
from tqdm import tqdm
import pandas as pd

BASE = 'data/zinc/'
files = glob.glob(BASE + '**/*.txt')
total_count = 944820131

# def count_file(name):
#     num_lines = sum(1 for line in open(name))
#     return num_lines - 1
#
#
# count = {name: count_file(name) for name in tqdm(files)}
# total_count = sum(count.values())
# print(total_count)

n_train, n_dev, n_test = 10000000, 50000, 50000
smiles = []
zinc_id = []

for file in tqdm(files):
    with open(file) as f:
        lines = f.readlines()[1:]
    sample_size = int((n_train + n_dev + n_test) / total_count * len(lines)) + 1
    random.shuffle(lines)
    sample_lines = lines[:sample_size]
    for line in sample_lines:
        x = line.split('\t')
        s, i = x[0], x[1]
        smiles.append(s)
        zinc_id.append(i)

df = pd.DataFrame({
    'zinc_id': zinc_id,
    'smiles': smiles
})
df = df.sample(frac=1).reset_index(drop=True)

train_df = df.loc[:n_train]
train_df.to_csv(BASE + 'train.csv', index=False)

dev_df = df.loc[n_train:n_train+n_dev]
dev_df.to_csv(BASE + 'dev.csv', index=False)

test_df = df.loc[n_train+n_dev:n_train+n_dev+n_test]
test_df.to_csv(BASE + 'test.csv', index=False)

