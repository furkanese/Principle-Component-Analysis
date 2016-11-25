'''
MUSTAFA FURKAN ESEOGLU
16501034
MACHINE LEARNING HW2
'''
from __future__ import division
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

eigenvalue, eigenvector = np.linalg.eig(covariance)

sortedIndex = eigenvalue.argsort()[::-1]
eigenvalue = eigenvalue[sortedIndex]
eigenvector = eigenvector[:, sortedIndex]

'''
# taking real portions of values
eigenvalue_Real = eigenvalue.real
eigenvector_Real = eigenvector.real

for i in range(0,64):
   print(eigenvalue_Real[i])
   print(eigenvector_Real[:,i])
print np.shape(eigenvalue)
print np.shape(eigenvector)
'''


#KULLANMA
# taking amount of vectors needed
thetreshold = 0.8
eigensum = 0
valuecount , = np.shape(eigenvalue)
for i in range(0,valuecount):
    eigensum += eigenvalue[i]
csum = 0
for i in range(0,valuecount):
    csum += eigenvalue[i]
    tv = csum / eigensum
    if tv > thetreshold:
        needed_eigen_count = i
        break

print needed_eigen_count


# eigenspace process
eigenX, eigenY = np.shape(eigenvector)

#eigen_perc = int(eigenY * 0.8)
eigen_perc = needed_eigen_count


mult_eigenvec = np.zeros([eigenX, eigen_perc])
for i in range(0, eigen_perc):
    mult_eigenvec[:, i] = np.real(eigenvector[:, i])
    #print(mult_eigenvec[:, i])

print np.shape(mult_eigenvec)

transposed_eigenvec = mult_eigenvec.transpose()

print np.shape(transposed_eigenvec)

eigenspaces = np.dot(transposed_eigenvec,meancentered)

print np.shape(eigenspaces)
eigentrainX, eigentrainY = np.shape(eigenspaces)

# testing
meancenteredtest = testdata
testX, testY = np.shape(testdata)
print(str(testY) + ' ornek icin test basliyor')

for i in range(0, 64):
    for j in range(0, testY):
        meancenteredtest[i, j] -= mean[i, 0]

testspace = np.dot(transposed_eigenvec, meancenteredtest)

distances = np.zeros([eigentrainY, testY])
for i in range(0,testY):
    for j in range(0, eigentrainY):
        dist = 0
        #euclidean distance with new attributes
        for k in range(0, eigen_perc):
            dist += (testspace[k, i] - eigenspaces[k, j]) * (testspace[k, i] - eigenspaces[k, j])
        dist = np.sqrt(dist)
        distances[j, i] = dist

print ('evaluation step')
distX, distY = np.shape(distances)
predict = np.zeros([2,distY])
for i in range(0,distY):
    ind = 0
    mindist = distances[0, i]
    for j in range(1,distX):
        if distances[j,i] < mindist:
            mindist = distances[j, i]
            ind = j
    predict[0, i] = trainlabel[0, ind]
    predict[1, i] = testlabel[0,i]

count = 0
for i in range(0,distY):
    #print('orj ' + str(predict[1,i]) + ' tahmin ' + str(predict[0,i]))
    if predict[0, i] == testlabel[0,i]:
        count += 1
success = float(count / testY)
print  success

confusion = np.zeros([10,10])
success_matrix = np.zeros([10,2])
for i in range (0,10):
    for j in range(0,distY):
        if predict[1,j] == i:
            confusion[i, predict[0,j]] += 1
            if predict[1,j] == predict[0,j]:
                success_matrix[i,0] +=1
            else:
                success_matrix[i,1] +=1


for i in range (0,10):
    print(confusion[i,:])

for i in range (0,10):
    print(str(i)+ ' basari ' + str(success_matrix[i,0] / (success_matrix[i,0] + success_matrix[i,1])))

for i in range (0,10):
    print(success_matrix[i,:])
