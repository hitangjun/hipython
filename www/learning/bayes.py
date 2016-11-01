#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'John Tang'

from numpy import *

def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']
                   ]
    classVec = [0,1,0,1,0,1] #1代表侮辱性文字，0代表正常言论
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
        print vocabSet
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    print vocabList
    print inputSet
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print "the word %s is not in  my vocabulary!" % word
    return returnVec


def trainNB0(trainMatrix,trainCatrgory):
    numTrainDocs = len(trainMatrix)


if __name__=='__main__':
     listOPosts,listClasses = loadDataSet()
     myVocabList = createVocabList(listOPosts)
     for ele in listOPosts:
         print setOfWords2Vec(myVocabList,ele)