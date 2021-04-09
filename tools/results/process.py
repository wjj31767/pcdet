import matplotlib.pyplot as plt
import os
import numpy as np
mediums = []
mediums1 = []
mediums2 = []
files = []
files1 = []
files2 = []
for filename in sorted(os.listdir('attack_defense_before0.02')):
    if 'fgsm' in filename[1:5] and '2.5' in filename:
        print(filename[-8:-4])
        files.append(float(filename[-8:-4]))

        with open('attack_defense_before0.02/'+filename,'r') as f:
            for num,line in enumerate(f.readlines()):
                if num==11:
                    mediums.append(float(line.split(',')[1]))
    if 'fgsm' in filename[1:5] and '1' in filename[:6]:
        print(filename[-8:-4])
        files1.append(float(filename[-8:-4]))

        with open('attack_defense_before0.02/'+filename,'r') as f:
            for num,line in enumerate(f.readlines()):
                if num==11:
                    mediums1.append(float(line.split(',')[1]))
print(mediums)
plt.plot(files,mediums)
plt.plot(files1,mediums1)

plt.show()
# plt.savefig('process.png')


            
