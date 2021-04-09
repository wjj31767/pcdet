import matplotlib.pyplot as plt
import os
import numpy as np

mediums = []
mediums1 = []
mediums2 = []
mediums3 = []

files = []
files1 = []
files2 = []
files3 = []
for filename in sorted(os.listdir('attack_defense_before0.02')):
    if 'fgsm' in filename[:5] and '2.5' in filename:
        files.append(float(filename[-8:-4]))
        with open('attack_defense_before0.02/' + filename, 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums.append(float(line.split(',')[1]))
    if 'fgsm' in filename[:5] and '20' in filename:
        files1.append(float(filename[-8:-4]))
        with open('attack_defense_before0.02/' + filename, 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums1.append(float(line.split(',')[1]))

    if 'fgsm' in filename[:5] and '1' in filename[5]:
        files2.append(float(filename[-8:-4]))
        with open('attack_defense_before0.02/' + filename, 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums2.append(float(line.split(',')[1]))
    if 'fgsm' in filename[:5] and 'i' in filename[5]:
        files3.append(float(filename[-8:-4]))
        with open('attack_defense_before0.02/' + filename, 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums3.append(float(line.split(',')[1]))
print(files, mediums)
plt.plot(files, mediums, label='norm 2.5')
plt.plot(files1, mediums1, label='norm 2')
plt.plot(files2, mediums2, label='norm 1')
print(files3, mediums3)
plt.plot(files3, mediums3, label='norm inf')
plt.legend()
plt.title("medium AP under (Car AP_R40@0.70, 0.70, 0.70) \n"
          "with different norm fgsm attack")
plt.show()
# plt.savefig('process.png')


# mediums = []
# mediums1 = []
# mediums2= []
# mediums3 = []
#
# files = []
# files1 = []
# files2 = []
# files3 = []
# for filename in sorted(os.listdir('attack_defense_before0.02')):
#     if 'fgsm' in filename[:5] and 'inf' in filename:
#         files.append(float(filename[-8:-4]))
#         with open('attack_defense_before0.02/'+filename,'r') as f:
#             for num,line in enumerate(f.readlines()):
#                 if num==11:
#                     mediums.append(float(line.split(',')[1]))
#     if 'ifgsm' in filename and 'inf' in filename:
#         files1.append(float(filename[-8:-4]))
#         with open('attack_defense_before0.02/'+filename,'r') as f:
#             for num,line in enumerate(f.readlines()):
#                 if num==11:
#                     mediums1.append(float(line.split(',')[1]))
#
#     if 'pgd' in filename and 'inf' in filename:
#         files2.append(float(filename[-8:-4]))
#         with open('attack_defense_before0.02/'+filename,'r') as f:
#             for num,line in enumerate(f.readlines()):
#                 if num==11:
#                     mediums2.append(float(line.split(',')[1]))
#     if 'momentum' in filename and 'inf' in filename:
#         files3.append(float(filename[-8:-4]))
#         with open('attack_defense_before0.02/'+filename,'r') as f:
#             for num,line in enumerate(f.readlines()):
#                 if num==11:
#                     mediums3.append(float(line.split(',')[1]))
# print(files,mediums)
# plt.plot(files,mediums,label = 'fgsm')
# plt.plot(files1,mediums1,label = 'ifgsm')
# plt.plot(files2,mediums2, label = 'pgd')
# plt.plot(files3,mediums3, label = 'momentum')
# plt.legend()
# plt.title("medium AP under (Car AP_R40@0.70, 0.70, 0.70) \n"
#           "with different attack method with norm inf")
# plt.show()

mediums = []
mediums1 = []
mediums2 = []
mediums3 = []
mediums4 = []

files = [0.01, 0.02, 0.04]
files1 = []
files2 = []
files3 = []
files4 = []

for filename in sorted(os.listdir('./')):
    if 'before' in filename and '.' not in filename:
        with open(filename + '/ofgsminf0.01.txt', 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums.append(float(line.split(',')[1]))
        with open(filename + '/ofgsminf0.02.txt', 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums.append(float(line.split(',')[1]))
        with open(filename + '/ofgsminf0.04.txt', 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums.append(float(line.split(',')[1]))
    if '2.5' in filename:
        with open(filename + '/ofgsminf0.01.txt', 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums1.append(float(line.split(',')[1]))
        with open(filename + '/ofgsminf0.02.txt', 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums1.append(float(line.split(',')[1]))
        with open(filename + '/ofgsminf0.04.txt', 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums1.append(float(line.split(',')[1]))
    if '20' in filename:
        with open(filename + '/ofgsminf0.01.txt', 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums2.append(float(line.split(',')[1]))
        with open(filename + '/ofgsminf0.02.txt', 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums2.append(float(line.split(',')[1]))
        with open(filename + '/ofgsminf0.04.txt', 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums2.append(float(line.split(',')[1]))
    if '10' in filename:
        with open(filename + '/ofgsminf0.01.txt', 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums3.append(float(line.split(',')[1]))
        with open(filename + '/ofgsminf0.02.txt', 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums3.append(float(line.split(',')[1]))
        with open(filename + '/ofgsminf0.04.txt', 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums3.append(float(line.split(',')[1]))
    if 'inf' in filename:
        with open(filename + '/ofgsminf0.01.txt', 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums4.append(float(line.split(',')[1]))
        with open(filename + '/ofgsminf0.02.txt', 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums4.append(float(line.split(',')[1]))
        with open(filename + '/ofgsminf0.04.txt', 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums4.append(float(line.split(',')[1]))

print(files, mediums)
plt.plot(files, mediums, label='before defense')
plt.plot(files, mediums1, label='defense norm 2.5')
plt.plot(files, mediums2, label='defense norm 2')
plt.plot(files, mediums3, label='defense norm 1')
plt.plot(files, mediums4, label="defense norm inf")
plt.legend()
plt.title("medium AP under (Car AP_R40@0.70, 0.70, 0.70) \n"
          "with different models under fgsm sign attack")
plt.show()

# sign different method
mediums = []
mediums1 = []
mediums2 = []
mediums3 = []
mediums4 = []

files = [0.01, 0.02, 0.04]
files1 = []
files2 = []
files3 = []
files4 = []

for filename in sorted(os.listdir('./')):
    if 'before' in filename and '.' not in filename:
        with open(filename + '/ofgsminf0.01.txt', 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums.append(float(line.split(',')[1]))
        with open(filename + '/ofgsminf0.02.txt', 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums.append(float(line.split(',')[1]))
        with open(filename + '/ofgsminf0.04.txt', 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums.append(float(line.split(',')[1]))
        with open(filename + '/oifgsm10.01.txt', 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums1.append(float(line.split(',')[1]))
        with open(filename + '/oifgsm10.02.txt', 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums1.append(float(line.split(',')[1]))
        with open(filename + '/oifgsm10.04.txt', 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums1.append(float(line.split(',')[1]))
        with open(filename + '/omomentuminf0.01.txt', 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums2.append(float(line.split(',')[1]))
        with open(filename + '/omomentuminf0.02.txt', 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums2.append(float(line.split(',')[1]))
        with open(filename + '/omomentuminf0.04.txt', 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums2.append(float(line.split(',')[1]))
        with open(filename + '/opgdinf0.01.txt', 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums3.append(float(line.split(',')[1]))
        with open(filename + '/opgdinf0.02.txt', 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums3.append(float(line.split(',')[1]))
        with open(filename + '/opgdinf0.04.txt', 'r') as f:
            for num, line in enumerate(f.readlines()):
                if num == 11:
                    mediums3.append(float(line.split(',')[1]))

print(files, mediums)
plt.plot(files, mediums, label='fgsm')
plt.plot(files, mediums1, label='ifgsm')
plt.plot(files, mediums2, label='momentum')
plt.plot(files, mediums3, label='pgd')
plt.legend()
plt.title("medium AP under (Car AP_R40@0.70, 0.70, 0.70) \n"
          "with different models under fgsm sign attack")
plt.show()