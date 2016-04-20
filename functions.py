import csv
import random
import math
import operator
import numpy
import scipy
import warnings
from sklearn.metrics import f1_score

def loadDataFromFile(filename,dataSet):
	with open(filename, 'r') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)):
	        for y in range(6):
	            dataset[x][y] = float(dataset[x][y])
	        dataSet.append(dataset[x])


def loadDataset(dataset, kfold, trainingSet=[] , testSet=[]):
	for x in range(len(dataset)):
	    if (x >=kfold*20) and (x<=((kfold+1)*20 )-1):  # make k-fold
	        testSet.append(dataset[x])
	    else:
	        trainingSet.append(dataset[x])

def getEuclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def getClosestNeibors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1

	for x in range(len(trainingSet)):
	    dist = getEuclideanDistance(testInstance, trainingSet[x], length)
	    distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
	    neighbors.append(distances[x][0])
	return neighbors

def getMajority(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getEachAccuracy(testSet, predictions,i):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[20*i + x]:
			correct += 1

	return (correct/float(len(testSet))) * 100.0

def getAvgAccuracy(accuracy):
	avg = 0
	for x in range(len(accuracy)):
		avg += accuracy[x];
	return avg/len(accuracy)

def printReport(data,accuracy, prediction):
	actual = [col[5] for col in data]
	TP = 0
	FN = 0
	FP = 0
	TN = 0

	for x in range(len(prediction)):
		if actual[x] == 1.0 and prediction[x]==1.0:
			TP += 1

		elif actual[x] == 1.0 and prediction[x]==0.0:
			FN +=1

		elif actual[x] == 0.0 and prediction[x]==1.0:
			FP +=1

		elif actual[x] == 0.0 and prediction[x]==0.0:
			TN +=1

	for x in range(len(accuracy)):
		print('Accuracy: ' + repr(accuracy[x]) + '%')

	print('\nAverage Accuracy: ' + repr(getAvgAccuracy(accuracy)) + '%')

	print('\nConfusion Matrix : ')
	print('%5d   %5d'%(TP, FN))
	print('%5d   %5d'%(FP, TN))
	print('\n     TP rate   FP rate')
	print('Yes | %2.4f   %2.4f '% (TP/ (TP + FN) , FP/ (FP + TN)))
	print('No  | %2.4f   %2.4f '% (TN/ (FP + TN) , FN/(TP + FN)))
	precision = TP/(TP+FP)
	recall    = TP/(TP+FN)
	print('\nPrecision : %2.4f'% (precision))
	print('Recall    : %2.4f'% (recall))

	#both micro-averaged f1 socres are the same
	print('Micro Averaged F1 score : %2.4f'%((2*precision*recall)/(precision+recall)))
	warnings.filterwarnings("ignore", category=DeprecationWarning)
	print('\nmicro-averaged F1 score : %2.4f'% (f1_score(actual, prediction, average='micro')))