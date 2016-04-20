from functions import loadDataset, getClosestNeibors, getMajority, getEachAccuracy, loadDataFromFile, getAvgAccuracy, \
    printReport


def kNN():
    data=[]
    loadDataFromFile('data_matrix.csv',data)

    # run kNN on testSet and generate predictions
    k = 3
    c = 5
    accuracy = []
    prediction = []
    for i in range(c):
        # load data and split it into trainingSet and testSet
        trainingSet = []
        testSet = []

        loadDataset(data, i, trainingSet, testSet)
        # print('Train set: ' + repr(len(trainingSet)))
        # print('Test set: ' + repr(len(testSet)))

        # print(' =============================')
        for x in range(len(testSet)):
            #get nearest k elements
            neighbors = getClosestNeibors(trainingSet, testSet[x], k)

            #follow majority of neighbors
            result = getMajority(neighbors)

            prediction.append(result)

            #store predictions
            # print(repr(x) + ' > predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
            # calculate accuracy
        # print(' =============================')

        accuracy.append(getEachAccuracy(testSet, prediction, i))
        # print('Accuracy: ' + repr(accuracy[-1]) + '%')


    printReport(data, accuracy, prediction)

kNN()