#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import time
import numpy as np
from _threading_local import local

#######################################
#              Common                 #
#######################################
Item = namedtuple("Item", ['index', 'value', 'weight'])
value = 0
weight = 0
taken = []

def vdensity(item):
    return 1.0*item.value/item.weight

#######################################
#         Greedy Algorithm            #
#######################################

def greedyAlgorithm(items,capacity,version):
    
    global value
    global weight
    global taken
    
    if version == "trivial":
        # a trivial greedy algorithm for filling the knapsack
        # it takes items in-order until the knapsack is full
        for item in items:
            if weight + item.weight <= capacity:
                taken[item.index] = 1
                value += item.value
                weight += item.weight
        
    elif version == "vdensity":
        # greedy algorithm which fills knapsack with highest
        # value density items first
        vdensityItems = sorted(items,key=vdensity,reverse=True)
        for item in vdensityItems:
            if weight + item.weight <= capacity:
                taken[item.index] = 1
                value += item.value
                weight += item.weight

#######################################
#     Branch and Bound Algorithm      #
#######################################
bestValue = 0
Node = namedtuple("Node",["bestEstimate","value","weight","taken","depth"])

def getBestEstimate(items,capacity,currentValue,currentWeight,depth):
    
    bestEstimate = 0
    bestPotentialValue = 0
    
    if depth < len(items)-1:
        vdensityPotentialItems = sorted(items[depth+1:],key=vdensity,reverse=True)
        for index,item in enumerate(vdensityPotentialItems,start=0):
            if currentWeight + item.weight <= capacity:
                bestPotentialValue += item.value    
                currentWeight += item.weight
            else:
                break                
        bestPotentialValue += (1.0*(capacity - currentWeight)/
                               vdensityPotentialItems[index].weight*
                               vdensityPotentialItems[index].value)
                
        bestEstimate = currentValue + bestPotentialValue
    else:
        bestEstimate = currentValue
        
    return bestEstimate

def bbAlgorithmDFS(items,capacity,tmpTaken,currentValue,currentWeight,depth):
    
    global bestValue
    global taken

    if currentWeight > capacity:
        return 0

    # get best estimate corresponding to highest possible value 
    # using linear relaxation with already chosen selection 
    # of taken items and remaining knapsack capacity
    if bestValue >= getBestEstimate(items,capacity,currentValue,currentWeight,depth):
        return 0    
    
    # if less than max depth, check left and right search spaces 
    # i.e. i'th item is taken or untaken         
    # else if depth == item count - 1, maximum depth reached, in  
    # which case determine if selection of items is the best value
    if depth < len(items) -1:
        tmpTaken[depth+1]=1
        lval = bbAlgorithmDFS(items,capacity,tmpTaken,
                              currentValue + items[depth+1].value,
                              currentWeight + items[depth+1].weight,
                              depth+1)
        tmpTaken[depth+1]=0
        rval = bbAlgorithmDFS(items,capacity,tmpTaken,
                              currentValue,
                              currentWeight,
                              depth+1)
        return max(lval,rval)
                                 
    elif depth == len(items)-1:

        if currentValue > bestValue:
            bestValue = currentValue
            taken=tmpTaken[:]
        return currentValue 
        
def bbAlgorithmBFS(items,capacity,bfsQueue):
    
    global bestValue
    global taken
    
    #dequeue node with highest value
    while len(bfsQueue):
        
        #print bfsQueue
        
        maxBestEstimateIndex, maxBestEstimate = max(enumerate(bfsQueue),key=lambda x:x[1].bestEstimate)
        currentNode = bfsQueue.pop(maxBestEstimateIndex) 
        currentValue = currentNode.value
        currentWeight = currentNode.weight
        depth = currentNode.depth
        tmpTaken = currentNode.taken
        
        bestEstimate = getBestEstimate(items, capacity, currentValue, currentWeight,depth)
        
        if currentWeight > capacity or bestEstimate == 0 or bestValue >= bestEstimate:
            continue
        
        if depth < len(items) -1:
            
            bfsQueue.append(Node(getBestEstimate(items,capacity,currentValue+items[depth+1].value,currentWeight+items[depth+1].weight,depth+1),
                                 currentValue+items[depth+1].value,
                                 currentWeight+items[depth+1].weight,
                                 tmpTaken + (0b1 << depth + 1),depth+1))
            bfsQueue.append(Node(getBestEstimate(items,capacity,currentValue,currentWeight,depth+1),
                                 currentValue,
                                 currentWeight,
                                 tmpTaken,depth+1))
            
            continue
            
            
        elif depth == len(items)-1:
            if currentValue > bestValue:
                bestValue = currentValue
                taken=[int(bit,2) for bit in format(tmpTaken,"#0"+str(len(items)+2)+"b")[:1:-1]]
            
    else:
        return bestValue               
         
# linear relaxation - best estimate allows fractional item selection
def bbAlgorithm(items,capacity,searchpattern):
    
    global value
    global weight 
    global taken
    
    print 'start bbAlgorithm'
    
    if searchpattern == "dfs":
        tmpTaken = taken[:]
        
        tmpTaken[0] = 1
        lval = bbAlgorithmDFS(items,capacity,tmpTaken,items[0].value,items[0].weight,0)
        
        tmpTaken[0] = 0    
        rval = bbAlgorithmDFS(items,capacity,tmpTaken,0,0,0)
        
        value = max(lval,rval)
         
        
    if searchpattern == 'bfs':
        bfsQueue = []
        
        bfsQueue.append(Node(getBestEstimate(items,capacity,items[0].value,items[0].weight,0),
                             items[0].value,items[0].weight,0b1,0))
        bfsQueue.append(Node(getBestEstimate(items,capacity,0,0,0),
                             0,0,0b0,0))
        
        value = bbAlgorithmBFS(items,capacity,bfsQueue)
        
    return    

#######################################
#    Dynamic Programming Algorithm    #
#######################################

def getCellValue(iRow,iCol,list):
    
    for idx,rows in enumerate(list[iCol]):
        if iRow == rows[1]:
            return rows[0]
        elif iRow - rows[1] < 0:
            return list[iCol][idx-1][0]
    
    return list[iCol][-1][0]
    

def dpAlgorithm(items,capacity):
    
    global taken
    global value
    global weight
    
    localOptimalityTable = np.zeros((capacity+1,2))
    localOptimalityTable.dtype = int
    
    prevOptimalityList = []
    
    for idx,item in enumerate([Item(-1,0,0)]+items):
        '''
        print 'localOptimalityTable'
        print localOptimalityTable
        print 'prevOptimalityList: '
        print prevOptimalityList
        '''
        # store in list of lists only the delta values,and the respective
        # current capacity, for each column looked at by 
        # localOptimalityTable (i.e. space optimization)
        if(idx > 1):
            prevOptimalityList.append([])
            prevOptimalityList[idx-2].append([0,0])
            for currCapacity,val in enumerate(localOptimalityTable[:capacity,0]):
                currRowCell = localOptimalityTable[currCapacity,0]
                nextRowCell = localOptimalityTable[currCapacity+1,0]
                # spotted delta value, include in list of lists
                if(currRowCell != nextRowCell):
                    prevOptimalityList[idx-2].append([nextRowCell,currCapacity+1])
        
        if(idx):    
            localOptimalityTable[:,[0,1]] = localOptimalityTable[:,[1,0]]
            
        for currCapacity,row in enumerate(localOptimalityTable):
            
            if item.weight > currCapacity:
                row[1] = row[0]
            else:
                prevBestLessCurrWeight = localOptimalityTable[currCapacity-item.weight][0]
                row[1] = max(item.value+prevBestLessCurrWeight,row[0])
    '''
    print 'localOptimalityTable'
    print localOptimalityTable
    print 'prevOptimalityList: '
    print prevOptimalityList
    '''
    # Trace back, first using the localOptimalityTable for the last 
    # two items, then the list of lists for all the rest of the items
    
    currRowIdx = capacity
    currColIdx = len(items)
    if(localOptimalityTable[currRowIdx][1]==localOptimalityTable[currRowIdx][0]):
        taken[currColIdx-1]=0
    else:
        taken[currColIdx-1]=1
        value+=items[currColIdx-1].value
        currRowIdx-=items[currColIdx-1].weight
    currColIdx-=1
    
    if(localOptimalityTable[currRowIdx][0]==getCellValue(currRowIdx,currColIdx-1,prevOptimalityList)):
        taken[currColIdx-1]=0
    else:
        taken[currColIdx-1]=1
        value+=items[currColIdx-1].value
        currRowIdx-=items[currColIdx-1].weight
    currColIdx-=1    
    
    for currColIdx in range(currColIdx,0,-1):
        
        if(getCellValue(currRowIdx,currColIdx,prevOptimalityList)==getCellValue(currRowIdx,currColIdx-1,prevOptimalityList)):
            taken[currColIdx-1]=0
        else:
            taken[currColIdx-1]=1
            value+=items[currColIdx-1].value
            currRowIdx-=items[currColIdx-1].weight
            
    return    
    

def chooseAlgorithm(items,capacity):
    
    global value
    global taken
    
    #greedyAlgorithm(items,capacity,"trivial")
    #greedyAlgorithm(items,capacity,"vdensity")
    #bbAlgorithm(items,capacity,"dfs")
    #bbAlgorithm(items,capacity,"bfs")
    dpAlgorithm(items,capacity)
    
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    global taken

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))
    
    taken = [0]*len(items)
    return chooseAlgorithm(items,capacity)
    

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

