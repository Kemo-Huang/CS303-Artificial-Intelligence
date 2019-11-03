from random import sample

with open('seeds.txt', 'w') as f:
    arr = sample(range(15233), k=100)
    for seed in arr:
        f.write(str(seed + 1) + '\n')
