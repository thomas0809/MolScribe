import sys
import pandas as pd

pred1 = pd.read_csv(sys.argv[1])
pred2 = pd.read_csv(sys.argv[2])

inchi1 = pred1['InChI'].values
inchi2 = pred2['InChI'].values

replaced = 0

for i in range(len(inchi1)):
    if inchi1[i] == 'InChI=1S/H2O/h1H2':
        inchi1[i] = inchi2[i]
        replaced += 1
        # print(i, inchi2[i])

print(f'Replaced {replaced}')

pred1['InChI'] = inchi1
pred1.to_csv('ensemble.csv', index=False)
