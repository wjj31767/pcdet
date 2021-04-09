import matplotlib.pyplot as plt
import os
import numpy as np
mediums = []
files = []
for filename in sorted(os.listdir('sign')):
    files.append(float(filename[1:-4]))
    with open('sign/'+filename,'r') as f:
        for num,line in enumerate(f.readlines()):
            if num==11:
                mediums.append(float(line.split(',')[1]))
print(files,mediums)
plt.plot(files,mediums)
plt.savefig('process.png')

            
