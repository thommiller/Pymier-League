from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SigmoidLayer
from pybrain.datasets import SupervisedDataSet

import csv

teams = []
ds = SupervisedDataSet(2, 1)

def teamfind(strN):
    if strN not in teams:
        teams.append(strN)
    return teams.index(strN)

def scorefind(strScore):
    if int(strScore[0]) > int(strScore[2]):
        return 1                                    #WIN
    elif int(strScore[0]) < int(strScore[2]):
        return 0                                    #LOSS
    else:
        return 0.5                                  #DRAW

def displaypred(num):
    if(num> 0.5 and num <0.75):
        return " Draw"
    elif(num>0.75):
        return " Win"
    else:
        return " Loss"

with open('1-premierleague.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter= ',',quotechar="'")
    for row in reader:
        ds.addSample((teamfind(row[1]), teamfind(row[2])), (scorefind(row[3]),))
        ds.addSample((teamfind(row[2]), teamfind(row[1])), (scorefind(row[4]),))

#back propagation neural network with 3 layers of 2,3,1 neurons respectively.
#2 input (one for each team), 3 hidden (to deal with complexity) and 1 output
net = buildNetwork(2,3,1, bias=True, hiddenclass=SigmoidLayer)
trainer = BackpropTrainer(net, ds)
trainer.trainUntilConvergence()

for i in teams[:5]+teams[6:]:
    print teams[5]," vs ",i," - ",displaypred(net.activate([5,teams.index(i)])[0])