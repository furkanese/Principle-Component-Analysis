'''
MUSTAFA FURKAN ESEOGLU
16501034
MACHINE LEARNING HW2
'''

import numpy as np

# reading data and taking totalcount
with open('sayi.dat','r') as f:
    read_data = f.read()
f.close()
datacount = 0
for l in read_data:
    if l == '\n':
        datacount += 1

labels = np.zeros([1, datacount])
rawdata = np.zeros([64, datacount])

# assigning data to array as vectors
indx = 0
indy = 0
tmpdata = ''
for l in read_data:
    if indx == 64:
        labels[0, indy] = float(l)
        indx = 0
    else:
        if l != ',':
            tmpdata += l

        if l == '\n':
            tmpdata = ''
            indy += 1
        elif l == ',':
            rawdata[indx, indy] = float(tmpdata)
            tmpdata = ''
            indx += 1

# dividing training and test data
traindata = np.zeros([64, 50])
trainlabel = np.zeros([1, 50])
testdata = np.zeros([64, datacount-50])
testlabel = np.zeros([1, datacount-50])

trind = 0
tsind = 0
countX = 0
label = 0
while label != 10:
    for i in range(0, datacount):
        if labels[0, i] == label:
            if countX != 5:
                trainlabel[0, trind] = label
                for j in range(0, 64):
                    traindata[j, trind] = rawdata[j, i]
                trind += 1
                countX += 1
            else:
                testlabel[0, tsind] = label
                for j in range(0, 64):
                    testdata[j, tsind] = rawdata[j, i]
                tsind += 1
    label += 1
    countX = 0

# taking mean of training data
mean = np.zeros([64, 1])
for i in range(0, 64):
    for j in range(0,50):
        mean[i, 0] += traindata[i, j]
    # 50 is our training size
    mean[i, 0] /= 50

meancentered = traindata

for i in range(0, 64):
    for j in range(0, 50):
        meancentered[i, j] -= mean[i, 0]

# taking transpose of mean centered
meancentered_transpose = meancentered.transpose()

# co variance matrix
covariance = np.dot(meancentered, meancentered_transpose)

print covariance
