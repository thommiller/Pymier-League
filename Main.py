import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def displayPred(num):
    if(num> 0.5 and num <0.75):
        return " - Draw"
    elif(num>0.75):
        return " - Win"
    else:
        return " - Loss"

# for training data we will compare Man-Utd's last 10 games
# input data will be [homeTeam, awayTeam]
# output data will be [0 | loss, 0.5 | draw, 1 | win]

# input dataset - every football match from 2014-2015 (MASSIVE WEB SCRAPING TASK)
#man u = 0, stoke = 1, yeovil town = 2, QPR = 3, cambridge = 4, leicester = 5
teams = ["Man U", "Stoke", "Yeovil Town", "QPR", "Cambridge", "Leicester"]
X = np.array([  [1,0], #stoke vs man u - draw
                [0,2], #yeovil town vs man u - won
                [3,0],
                [4,0],
                [0,5]
            ])

# output dataset
y = np.array([[0.5,1,1,0.5,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 3*np.random.random((2,1)) - 1

for iter in xrange(10000):

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
print teams[1],"\t\tvs\t",teams[0], displayPred(l1[0])
print teams[0],"\t\tvs\t",teams[2], displayPred(l1[1])
print teams[3],"\t\tvs\t",teams[0], displayPred(l1[2])
print teams[4],"\tvs\t",teams[0], displayPred(l1[3])
print teams[0],"\t\tvs\t",teams[5], displayPred(l1[4])
