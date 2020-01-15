import csv
import random
import math
import operator

#function loads data into sets for processing
def loadData(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'r') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)-1):
	        for y in range(9):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])

#function calculates total distance between instances
def calcDist(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)
 
#calc finds k closest neigbors 
def findNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = calcDist(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors
 
#function makes prediction based on neighbors
def predict(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

#function analyzes algorithm and returns stats
def analyze(testSet, predictions):
	TN = 0
	TP = 0
	FN = 0
	FP = 0
	
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			if testSet[x][-1] == '2':
				TN += 1
			elif testSet[x][-1] == '4':
				TP += 1
		elif testSet[x][-1] != predictions[x]:
			if testSet[x][-1] == '2':
				FN += 1
			elif testSet[x][-1] == '4':
				FP += 1

	#calculate accuracy
	acc = ((TN + TP)/float(len(testSet))) * 100.0
	
	#calculate TPR, PPV, TNR, and F1 Score
	print('TN =' + repr(TN) + ' TP = ' + repr(TP) + ' FN = ' + repr(FN) + ' FP = ' + repr(FP))
	tpr = (TP/(TP + FN))
	ppv = (TP/(TP + FP))
	tnr = (TN/(TN + FP))
	fs = (2.0 * ppv * tpr)/(ppv + tpr)

	return acc,tpr,ppv,tnr,fs
	
def main():
	# prepare data
	trainingSet=[]
	testSet=[]
	split = 0.67
	loadData('p3/breast-cancer-wisconsin-test.csv', split, trainingSet, testSet)
	print('Train set: ' + repr(len(trainingSet)))
	print('Test set: ' + repr(len(testSet)))
	# generate predictions
	predictions=[]
	k = 2
	
	for x in range(len(testSet)):
		neighbors = findNeighbors(trainingSet, testSet[x], k)
		result = predict(neighbors)
		predictions.append(result)
	
	accuracy,TPR,PPV,TNR,Fscore = analyze(testSet, predictions)
	
	print('k = ' + repr(k))
	print('Accuracy: ' + repr(accuracy) + '%')
	print('TPR = ' + repr(TPR))
	print('PPV = ' + repr(PPV))
	print('TNR = ' + repr(TNR))
	print('F1 Score = ' + repr(Fscore))

main()