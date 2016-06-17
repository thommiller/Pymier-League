# AI/ML Pymier-League
#### A series of supervised machine learning algorithms to attempt to predict premiere league game results.

Includes two neural networks all tested with hidden sigmoid, tanh and softmax layers and one SVM to classify teams and perform regression on game data to predict results.

Overfitting of the model has caused the majority of issues in the prediction and learning models, a smaller training set and larger testing set was used to avoid overfitting, which has recently led to inaccurate predictions. 

Current results from a feed-forward 4-55-1 neural network, with a hidden sigmoid layer trained using back-propagation on a dataset of 152 Tottenham home games (from 2010-2016), with 4 inputs: [homeTeamDefence, homeTeamAttack, awayTeamDefence, awayTeamAttack] -> [result] 
where all inputs and outputs are a member of [0,1)

*[The neural network was trained until the error was ~5% to avoid overfitting]*

+Tottenham  vs  Man United     home win probability:  0.53 
+Tottenham  vs  Arsenal        home win probability:  0.46 
+Tottenham  vs  Man City       home win probability:  0.83 
+Tottenham  vs  Southampton    home win probability:  0.74 
+Tottenham  vs  West Brom      home win probability:  0.71 
+Tottenham  vs  Liverpool      home win probability:  0.63 
+Tottenham  vs  Watford        home win probability:  0.77 
+Tottenham  vs  Crystal Palace home win probability:  0.78 
+Tottenham  vs  West Ham       home win probability:  0.95 
+Tottenham  vs  Swansea        home win probability:  0.81 
+Tottenham  vs  Chelsea        home win probability:  0.53 
+Tottenham  vs  Everton        home win probability:  0.95 
+Tottenham  vs  Stoke          home win probability:  0.83
+Tottenham  vs  Sunderland     home win probability:  0.94
+Tottenham  vs  Newcastle      home win probability:  0.93
+Tottenham  vs  Norwich City   home win probability:  0.91
+Tottenham  vs  Aston Villa    home win probability:  0.87

This is still a proof of concept and **very much a work in progress**. It is clear that the success of such a neural network is still unproven and predictions may well be arbitrary due  to the stochastic nature of the training data.

Work by Thom. 17/06/2016 [@thomx128]


