import csv
import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


teams = []
inputdata = []
outputdata = []

def teamfind(strN):
    if strN not in teams:
        teams.append(strN)
    return teams.index(strN)

def scorefind(strScore):
    if int(strScore[0]) > int(strScore[2]):
        return 1
    elif int(strScore[0]) < int(strScore[2]):
        return 0
    else:
        return 0.5

with open('1-premierleague.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter= ',',quotechar="'")
    for row in reader:
        inputdata.append([teamfind(row[1]),teamfind(row[2])])
        outputdata.append(scorefind(row[3]))
        inputdata.append([teamfind(row[2]),teamfind(row[1])])
        outputdata.append(scorefind(row[4]))

X = np.array(inputdata)
y = np.array(outputdata)

np.random.seed(1)
syn0 = 2*np.random.random((2,760)) - 1

# TRAIN
for iter in xrange(100):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)

    # update weights
    syn0 += np.dot(l0.T, l1_delta)

print "Game predictions based on training data:"
for x in l1:
    print teams[int(x[0]*10)],"\t",teams[int(x[1]*10)],"\t",x[2]