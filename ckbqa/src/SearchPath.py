
import sys
import os
import json
import copy
import math
from typing import List, Dict, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.mysql.MysqlConnection import MysqlConnection
from src.QueryGraph import QueryGraph
from config.MysqlConfig import CCKSConfig, CCKSConfig2021
from src.utils.data_processing import preprocessRel

ccksConfig = CCKSConfig2021()

def mergeForSameAnswer(queryGraphs: QueryGraph):
    '''
    针对具有相同答案节点的查询图进行合并，解决的是常见的二合一问题。
    '''
    newQueryGraphs = []
    for i, queryGraph in enumerate(queryGraphs):
        flag = 0
        for queryGraph2 in queryGraphs[i + 1:]:
            if(queryGraph.answer == queryGraph2.answer):
                flag = 1
                newQueryGraphs.append(QueryGraph())


class SearchPath(object):
    def __init__(self) -> None:
        self.mysqlConnection = MysqlConnection(ccksConfig)

    def updateAvailableEntityIds(self, availableEntityIds, pos, conflictMatrix):
        newEntityIds = []
        for entityId in availableEntityIds:
            if(conflictMatrix[pos][entityId] == 0):
                newEntityIds.append(entityId)
        return newEntityIds


    def generateQueryGraphComp(self, entitys=[],
                                conflictMatrix=[], candRels = None,\
                                queType = ''):
        '''
        功能：生成处理比较逻辑问题的查询图
        '''
        queryGraphs = self.mysqlConnection.searchQueryGraphsWithBasePath(entitys, queType = queType)
        for i in range(len(queryGraphs)):
            queryGraph = self.mysqlConnection.addAnswerType(queryGraphs[i])
        return queryGraphs

    def generateQueryGraphNoEntity(self, entitys=[],
                                candRelsList = None,\
                                queType = ''):
        '''
        功能：生成处理复杂逻辑问题的查询图
        '''
        queryGraphs = self.mysqlConnection.generateQueryGraphsNoEntity(entitys, candRelsList, queType = queType)
        for i in range(len(queryGraphs)):
            queryGraph = self.mysqlConnection.addAnswerType(queryGraphs[i])
        return queryGraphs

    def generateQueryGraph(self, entitys: List[str] = ['北京大学', '天文学家'], \
                            virtualEntitys = [], higherOrder = [], \
                            conflictMatrix = [], candRels = None, candRelsList = None):
        '''
        virtualEntitys: [(('5', 6, 6, 'distance', '<'), '5')]
        higherOrder: [(('argmax', 10, 11, 'higher-order'), 'argmax')]
        '''
        queryGraphs = []
        usedLogs = []
        currentUsedEntitys = []
        virtualConflictMatrix = []
        for i in range(len(virtualEntitys)):
            virtualConflictMatrix.append([0] * len(virtualEntitys))
            virtualConflictMatrix[i][i] = 1
        higherOrderConflictMatrix = []
        for i in range(len(higherOrder)):
            higherOrderConflictMatrix.append([0] * len(higherOrder))
            higherOrderConflictMatrix[i][i] = 1
        # print(conflictMatrix)
        # 询问关系词
        currentQueryGraphs = self.mysqlConnection.generateRelWithEntity(entitys, conflictMatrix)
        queryGraphs.extend(copy.deepcopy(currentQueryGraphs))
        # print(entitys, conflictMatrix)
        # import pdb; pdb.set_trace()
        forwardFlag = True
        # print(entitys)
        for i, entity in enumerate(entitys):
            # import pdb; pdb.set_trace()
            backwardFlag = True
            initAvailableEntityIds = [entityId for entityId in range(len(entitys))]
            newAvailableEntityIds = copy.deepcopy(initAvailableEntityIds)
            initAvailableVirtualEntityIds = [entityId for entityId in range(len(virtualEntitys))]
            initAvailableHigherOrderIds = [higherOrderId for higherOrderId in range(len(higherOrder))]
            currentQueryGraphs = []
            for hopNum in range(4):
                ############### 关系扩展###########################
                if(hopNum == 0): # 第一次从实体出发
                    queryGraphsForward = self.mysqlConnection.searchWithEntityBasedProp(entity)
                    currentQueryGraphs.extend(queryGraphsForward)
                    queryGraphsFor1HopBackward = self.mysqlConnection.searchWithValueBasedProp(entity)
                    currentQueryGraphs.extend(queryGraphsFor1HopBackward)
                    if(len(currentQueryGraphs) > 0):
                        newAvailableEntityIds = self.updateAvailableEntityIds(initAvailableEntityIds, i, conflictMatrix)
                    for indexI in range(len(currentQueryGraphs)):
                        currentQueryGraphs[indexI].setAvailableEntityIds(newAvailableEntityIds)
                        currentQueryGraphs[indexI].setAvailableVirtualEntityIds(initAvailableVirtualEntityIds)
                        currentQueryGraphs[indexI].setAvailableHigherOrderIds(initAvailableHigherOrderIds)
                else: # 从当前所在的查询图集合出发
                    candRelsListNew = candRelsList[0: math.ceil(len(candRels) / pow(2, hopNum - 1))]
                    currentCandRels = {item: candRels[item] for item in candRelsListNew}
                    # print(hopNum, len(currentCandRels))
                    if(hopNum > 1):
                        backwardFlag = False
                    currentQueryGraphs = self.mysqlConnection.generateOneHopFromQueryGraphs(currentQueryGraphs, \
                                    currentCandRels, candRelsList = candRelsListNew, forwardFlag=forwardFlag, backwardFlag=backwardFlag)

                ##################### 约束挂载 ##################
                for j, queryGraph in enumerate(currentQueryGraphs):
                    constrainQueryGraphs = [copy.deepcopy(queryGraph)]
                    # print('开始挂载约束')
                    while(len(constrainQueryGraphs) > 0):
                        queryGraph = constrainQueryGraphs.pop()
                        availableEntityIds = queryGraph.availableEntityIds
                        for entityId in availableEntityIds:
                            entity = entitys[entityId]
                            # 判断是否可以增加约束，可以则更新newAvailableEntityIds
                            operationFlag, queryGraph = self.mysqlConnection.addEntityConstrainForQueryGraph(copy.deepcopy(queryGraph), entity)
                            if(operationFlag):
                                availableEntityIds = self.updateAvailableEntityIds(availableEntityIds, entityId, conflictMatrix)
                                queryGraph.setAvailableEntityIds(availableEntityIds)
                                constrainQueryGraphs.append(queryGraph)
                                currentQueryGraphs.append(queryGraph)
                                break       # 每次只加一个约束，循环加到无法再加
                        # print('一个查询图挂载结束')
                        availableVirtualEntityIds = queryGraph.availableVirtualEntityIds
                        # 增加关系约束(只在单跳查询图上加)
                        candRelsSet = self.mysqlConnection.getRelationsForRelConstrain(queryGraph, candRelsList)
                        for rel in candRelsSet:
                            operationFlag, queryGraph = self.mysqlConnection.addRelationConstrainForQueryGraph(copy.deepcopy(queryGraph), rel)
                            if(operationFlag):
                                constrainQueryGraphs.append(queryGraph)
                                currentQueryGraphs.append(queryGraph)
                                break       # 每次只加一个约束，循环加到无法再加
                        # import pdb; pdb.set_trace()
                        # 增加非实体约束
                        for virtualEntityId in availableVirtualEntityIds:
                            virtualEntity = virtualEntitys[virtualEntityId]
                            # import pdb; pdb.set_trace()
                            operationFlag, queryGraph = self.mysqlConnection.addVirtualConstrainForQueryGraph(copy.deepcopy(queryGraph), virtualEntity)
                            if(operationFlag):
                                availableVirtualEntityIds = self.updateAvailableEntityIds(availableVirtualEntityIds, virtualEntityId, virtualConflictMatrix)
                                queryGraph.setAvailableVirtualEntityIds(availableVirtualEntityIds)
                                # print(queryGraph.serialization())
                                # print(availableEntityIds, entitys)
                                constrainQueryGraphs.append(queryGraph)
                                currentQueryGraphs.append(queryGraph)
                                break       # 每次只加一个约束，循环加到无法再加
                        # 增加高阶约束
                        availableHigherOrderIds = queryGraph.availableHigherOrderIds
                        for higherOrderId in availableHigherOrderIds:
                            # print(higherOrderId, availableHigherOrderIds, len(constrainQueryGraphs))
                            higherOrderItem = higherOrder[higherOrderId]
                            # import pdb; pdb.set_trace()
                            operationFlag, queryGraph = self.mysqlConnection.addHigherOrderConstrainForQueryGraph(copy.deepcopy(queryGraph),\
                                                                             [higherOrderItem], candRelsList)
                            if(operationFlag):
                                # print('更新前：', availableHigherOrderIds)
                                availableHigherOrderIds = self.updateAvailableEntityIds(availableHigherOrderIds, higherOrderId, higherOrderConflictMatrix)
                                queryGraph.setAvailableHigherOrderIds(availableHigherOrderIds)
                                constrainQueryGraphs.append(queryGraph)
                                currentQueryGraphs.append(queryGraph)
                                break
                queryGraphs.extend(currentQueryGraphs)
        # 组合约束
        queryGraphs = self.mysqlConnection.queryGraphsCombine(queryGraphs, candRelsList[0:100])
        # 根据查询图结构从知识库中重新检索答案
        keys = {}
        newQueryGraphs = []
        for queryGraph in queryGraphs:
            queryGraph = self.mysqlConnection.addAnswerType(queryGraph)
            # import pdb; pdb.set_trace()
            queryGraph.getKey()
            if(queryGraph.key not in keys):
                keys[queryGraph.key] = 1
                newQueryGraphs.append(queryGraph)
        return newQueryGraphs
    
    
    def generateQueryGraphSTAGGUpdate(self, entitys: List[str] = ['北京大学', '天文学家'], \
                            virtualEntitys = [], higherOrder = [], \
                            conflictMatrix = [], candRels = None, candRelsList = None):
        '''
        virtualEntitys: [(('5', 6, 6, 'distance', '<'), '5')]
        higherOrder: [(('argmax', 10, 11, 'higher-order'), 'argmax')]
        '''
        queryGraphs = []
        usedLogs = []
        currentUsedEntitys = []
        virtualConflictMatrix = []
        for i in range(len(virtualEntitys)):
            virtualConflictMatrix.append([0] * len(virtualEntitys))
            virtualConflictMatrix[i][i] = 1
        higherOrderConflictMatrix = []
        for i in range(len(higherOrder)):
            higherOrderConflictMatrix.append([0] * len(higherOrder))
            higherOrderConflictMatrix[i][i] = 1
        forwardFlag = True
        # print(entitys)
        for i, entity in enumerate(entitys):
            # import pdb; pdb.set_trace()
            backwardFlag = True
            initAvailableEntityIds = [entityId for entityId in range(len(entitys))]
            newAvailableEntityIds = copy.deepcopy(initAvailableEntityIds)
            initAvailableVirtualEntityIds = [entityId for entityId in range(len(virtualEntitys))]
            initAvailableHigherOrderIds = [higherOrderId for higherOrderId in range(len(higherOrder))]
            currentQueryGraphs = []
            for hopNum in range(2):
                ############### 关系扩展###########################
                if(hopNum == 0): # 第一次从实体出发
                    queryGraphsForward = self.mysqlConnection.searchWithEntityBasedProp(entity)
                    currentQueryGraphs.extend(queryGraphsForward)
                    queryGraphsFor1HopBackward = self.mysqlConnection.searchWithValueBasedProp(entity)
                    currentQueryGraphs.extend(queryGraphsFor1HopBackward)
                    if(len(currentQueryGraphs) > 0):
                        newAvailableEntityIds = self.updateAvailableEntityIds(initAvailableEntityIds, i, conflictMatrix)
                    for indexI in range(len(currentQueryGraphs)):
                        currentQueryGraphs[indexI].setAvailableEntityIds(newAvailableEntityIds)
                        currentQueryGraphs[indexI].setAvailableVirtualEntityIds(initAvailableVirtualEntityIds)
                        currentQueryGraphs[indexI].setAvailableHigherOrderIds(initAvailableHigherOrderIds)
                else: # 从当前所在的查询图集合出发
                    # import pdb; pdb.set_trace()
                    candRelsListNew = candRelsList[0: ]
                    currentCandRels = {item: candRels[item] for item in candRelsListNew}
                    # print(hopNum, len(currentCandRels))
                    if(hopNum >= 1):
                        backwardFlag = False
                    # print('开始延伸', len(currentQueryGraphs))
                    currentQueryGraphs = self.mysqlConnection.generateOneHopFromQueryGraphs(currentQueryGraphs, \
                                    currentCandRels, candRelsList = candRelsListNew, forwardFlag=forwardFlag, backwardFlag=backwardFlag)
                    # print('延伸结束')
                ##################### 约束挂载 ##################
                for j, queryGraph in enumerate(currentQueryGraphs):
                    constrainQueryGraphs = [copy.deepcopy(queryGraph)]
                    # print('开始挂载约束')
                    while(len(constrainQueryGraphs) > 0):
                        queryGraph = constrainQueryGraphs.pop()
                        availableEntityIds = queryGraph.availableEntityIds
                        for entityId in availableEntityIds:
                            entity = entitys[entityId]
                            # 判断是否可以增加约束，可以则更新newAvailableEntityIds
                            operationFlag, queryGraph = self.mysqlConnection.addEntityConstrainForQueryGraph(copy.deepcopy(queryGraph), entity)
                            if(operationFlag):
                                availableEntityIds = self.updateAvailableEntityIds(availableEntityIds, entityId, conflictMatrix)
                                queryGraph.setAvailableEntityIds(availableEntityIds)
                                constrainQueryGraphs.append(queryGraph)
                                currentQueryGraphs.append(queryGraph)
                                break       # 每次只加一个约束，循环加到无法再加
                        # print('一个查询图挂载结束')
                        availableVirtualEntityIds = queryGraph.availableVirtualEntityIds
                        # import pdb; pdb.set_trace()
                        # 增加非实体约束
                        for virtualEntityId in availableVirtualEntityIds:
                            virtualEntity = virtualEntitys[virtualEntityId]
                            # import pdb; pdb.set_trace()
                            operationFlag, queryGraph = self.mysqlConnection.addVirtualConstrainForQueryGraph(copy.deepcopy(queryGraph), virtualEntity)
                            if(operationFlag):
                                availableVirtualEntityIds = self.updateAvailableEntityIds(availableVirtualEntityIds, virtualEntityId, virtualConflictMatrix)
                                queryGraph.setAvailableVirtualEntityIds(availableVirtualEntityIds)
                                # print(queryGraph.serialization())
                                # print(availableEntityIds, entitys)
                                constrainQueryGraphs.append(queryGraph)
                                currentQueryGraphs.append(queryGraph)
                                break       # 每次只加一个约束，循环加到无法再加
                        # 增加高阶约束
                        availableHigherOrderIds = queryGraph.availableHigherOrderIds
                        for higherOrderId in availableHigherOrderIds:
                            # print(higherOrderId, availableHigherOrderIds, len(constrainQueryGraphs))
                            higherOrderItem = higherOrder[higherOrderId]
                            # import pdb; pdb.set_trace()
                            operationFlag, queryGraph = self.mysqlConnection.addHigherOrderConstrainForQueryGraph(copy.deepcopy(queryGraph),\
                                                                             [higherOrderItem], candRelsList)
                            if(operationFlag):
                                # print('更新前：', availableHigherOrderIds)
                                availableHigherOrderIds = self.updateAvailableEntityIds(availableHigherOrderIds, higherOrderId, higherOrderConflictMatrix)
                                queryGraph.setAvailableHigherOrderIds(availableHigherOrderIds)
                                constrainQueryGraphs.append(queryGraph)
                                currentQueryGraphs.append(queryGraph)
                                break
                queryGraphs.extend(currentQueryGraphs)
        # 根据查询图结构从知识库中重新检索答案
        keys = {}
        newQueryGraphs = []
        for queryGraph in queryGraphs:
            queryGraph.getKey()
            if(queryGraph.key not in keys):
                keys[queryGraph.key] = 1
                newQueryGraphs.append(queryGraph)
        return newQueryGraphs

    
    def generateQueryGraphSTAGG(self, entitys: List[str] = ['北京大学', '天文学家'], conflictMatrix = []):
        queryGraphs = []
        usedLogs = []
        currentUsedEntitys = []
        # print(conflictMatrix)
        for i, entity in enumerate(entitys):
            availableEntityIds = [entityId for entityId in range(len(entitys))]
            newAvailableEntityIds = copy.deepcopy(availableEntityIds)
            currentQueryGraphs = []
            ############### 主路径搜索 ##########################
            for hopNum in range(2):
                if(hopNum == 0): # 第一次从实体出发
                    queryGraphsForward = self.mysqlConnection.searchWithEntityBasedProp(entity)
                    currentQueryGraphs.extend(queryGraphsForward)
                    queryGraphsFor1HopBackward = self.mysqlConnection.searchWithValueBasedProp(entity)
                    currentQueryGraphs.extend(queryGraphsFor1HopBackward)
                else: # 从当前所在的查询图集合出发
                    currentQueryGraphs = self.mysqlConnection.searchOneHopFromQueryGraphs(currentQueryGraphs)
                if(len(currentQueryGraphs) > 0):
                    newAvailableEntityIds = self.updateAvailableEntityIds(availableEntityIds, i, conflictMatrix)
                for indexI in range(len(currentQueryGraphs)):
                    currentQueryGraphs[indexI].setAvailableEntityIds(newAvailableEntityIds)
                queryGraphs.extend(currentQueryGraphs)
            ################## 约束挂载 ###########################
            constrainQueryGraphs = copy.deepcopy(queryGraphs)
            while(len(constrainQueryGraphs) > 0):
                queryGraph = constrainQueryGraphs.pop()
                availableEntityIds = queryGraph.availableEntityIds
                # print('1:', queryGraph.serialization())
                # print('1:', availableEntityIds, entitys)
                for entityId in availableEntityIds:
                    entity = entitys[entityId]
                    # print('entity:', entity)
                    # 判断是否可以增加约束，可以则更新newAvailableEntityIds
                    operationFlag, queryGraph = self.mysqlConnection.addEntityConstrainForQueryGraphSTAGG(queryGraph, entity)
                    if(operationFlag):
                        availableEntityIds = self.updateAvailableEntityIds(availableEntityIds, entityId, conflictMatrix)
                        queryGraph.setAvailableEntityIds(availableEntityIds)
                        constrainQueryGraphs.append(queryGraph)
                        queryGraphs.append(queryGraph)
                        break       # 每次只加一个约束，循环加到无法再加
        keys = {}
        newQueryGraphs = []
        for queryGraph in queryGraphs:
            queryGraph.getKey()
            if(queryGraph.key not in keys):
                # answerList = self.mysqlConnection.searchAnswer(queryGraph=queryGraph)
                answerList = self.mysqlConnection.searchAnswerBySQL(queryGraph=queryGraph)
                if(len(answerList) == 0):
                    continue
                queryGraph.updateAnswer('\t'.join(answerList))
                # queryGraph.serialization()
                keys[queryGraph.key] = 1
                newQueryGraphs.append(queryGraph)
        return newQueryGraphs



if __name__ == '__main__':
    searchPath = SearchPath()
    searchPath.searchQueryGraph()