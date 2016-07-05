
''' 
KNN algorithm:
This algorithm needs a training set with the labels. And the input is a new data without
the labels. After running this algorithm,  this new data will fall into the closest k datas
,also this will choose these kind of labels.
By using the euclidean metric to calculate the cloest k datas 
	   According to the euclidean metric:
	   eg :(in the three-dimensional space)
	       d = sqrt ((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)
'''
from numpy import *
import operator

def createDataSet(): #Create the Data set and the labels.
	'''  This is the way to define the training set 
	    Please pay attention to the order''' 
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group ,labels 

def calssfiy0(inX,dataSet,lables ,k):# do the classfiy here and mainly is about the calculation  
	
	'''get the size of the DataSet,that means how many elements in this set '''
	dataSetSize = dataSet.shape[0]
	'''tile(element,(rowNumber,colNumber)): This function takes three inputs and generate an array([[]])
	          The element can be a number or everything include the number.
	          The rowNumber and colNumber are the format you want. 
	    The diffMat is going to do the subtraction between the One you create and the original one. 
	    The inX is the data you want to predict. The format must be the same as the DataSet 
	'''
	diffMat = tile(inX,(dataSetSize,1)) - dataSet
	'''Calculate the suqre of the data set  '''
	sqDiffMat =  diffMat**2 
	''' sum is going to calculate the sum of array, 1 means that put each of them after sum in a list '''
	sqDistances =  sqDiffMat.sum(axis=1)
	'''Calculate the root of the dataSet'''
	distances = sqDistances**0.5
	''' Sort the whole dataSet and the return value is the Index of the sorted value not the origin value '''
	sortedDistIndicies = distances.argsort()
	'''Create a new dict to store the k closest data ''' 
	classCount = {}
	''' '''
	for i in range(k):
		'''According to the indicies of the data, record the label '''
		voteIlabel = labels[sortedDistIndicies[i]]
		'''Create the element in the dict, if the element has already exsited , add one  '''
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	''' Sort the dict (According to the value not the key ,and the order is desecnding).
	    The return value is a list including the a pair of key and value not a dict '''
	sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]



'''
This file2matrix is going to transfer  all the data from text to the format we can use
Remember the data in text file has already had a special foramt.
'''

def file2matrix(filename):
	fr = open(filename)
	numberOflines = len(fr.readlines())
	returnMat = zeros((numberOflines,3))
	classLabelVector = []
	fr = open(filename)
	index = 0
	for line in fr.readlines():
		line = line.strip()
		ListFromLine = line.split('\t')
		returnMat[index,:]=ListFromLine[0:3]
		classLabelVector.append(int(ListFromLine[-1])) 
		index =+ 1
	return returnMat,classLabelVector



