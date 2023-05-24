# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import torch.optim.lr_scheduler.StepLR
#from __future__ import print_function
import torch.utils.data as data
import sympy
import numpy as np
from sympy import *
import copy
from scipy.optimize import least_squares, minimize
from regularization_L12 import L12Smooth
def GetComplexity(express):
    expressTmp = express
    for a in sympy.preorder_traversal(express):
        if isinstance(a, sympy.Float):
            expressTmp = expressTmp.subs(a, 0.5)
        # if isinstance(a, sympy.Integer):
        #     expressTmp = expressTmp.subs(a, 2)
    expressTmp = str(expressTmp)
    complexity = 0
    complexity = complexity + expressTmp.count('+')
    complexity = complexity + expressTmp.count('-')
    complexity = complexity + expressTmp.count('*') - expressTmp.count('**')
    complexity = complexity + expressTmp.count('/')
    complexity = complexity + expressTmp.count('sin')
    complexity = complexity + expressTmp.count('cos')
    complexity = complexity + expressTmp.count('exp')
    complexity = complexity + expressTmp.count('log')
    complexity = complexity + expressTmp.count('sqrt')

    return complexity

def square(x):
    a = x**2
    return a
def outPutSymRltAllWeightInn(variNum, symNum, hiddenNo, paramNums, indexSel, indexList):
    if 2 == paramNums[3]:
        symNameList = ["+", "-", "*", "/", "sin", "cos", "exp", "log", "square"]
    else:
        symNameList = ["+", "-", "*", "sin", "cos", "exp", "log", "square"]
    variNameListTmp = ["a", "b", "c", "d", "e"]
    variNameList = []
    for i in range(variNum):
        variNameList.append(variNameListTmp[i])
    variNameList.append("1")
    allIndex = indexList[hiddenNo]
    outStr = ""
    flag = 1
    if indexSel >= symNum:
        if hiddenNo > 0:
            outStr, flag = outPutSymRltAllWeightInn(variNum, symNum, hiddenNo-1, paramNums, indexSel-symNum, indexList)
        else:
            outStr = variNameList[indexSel - symNum]
            flag = 0
    else:
        if paramNums[indexSel] == 2:
            if hiddenNo > 0:
                tmp1, tmpFlag1 = outPutSymRltAllWeightInn(variNum, symNum, hiddenNo - 1, paramNums, allIndex[indexSel, 0], indexList)
                tmp2 = symNameList[indexSel]
                tmp3, tmpFlag2 = outPutSymRltAllWeightInn(variNum, symNum, hiddenNo - 1, paramNums, allIndex[indexSel, 1], indexList)

                symTmp1 = ""
                if 0 < tmpFlag1:
                    symTmp1 = symNameList[tmpFlag1-1]
                if "+" == symTmp1 or "-" == symTmp1 or "/" == symTmp1 or "square" == symTmp1:
                    tmp1 = "(" + tmp1 + ")" + "*" + "weight"
                else:
                    tmp1 = tmp1 + "*" + "weight"

                symTmp2 = ""
                if 0 < tmpFlag2:
                    symTmp2 = symNameList[tmpFlag2 - 1]
                if "+" == symTmp2 or "-" == symTmp2 or "/" == symTmp2 or "square" == symTmp2:
                    tmp3 = "(" + tmp3 + ")" + "*" + "weight"
                else:
                    tmp3 = tmp3 + "*" + "weight"

                if "/" == tmp2:
                    outStr = "(" + tmp1 + ")" + tmp2 + "(" + tmp3 + ")"
                else:
                    outStr = tmp1 + tmp2 + tmp3
            else:
                if "/" == symNameList[indexSel]:
                    outStr = "(" + variNameList[allIndex[indexSel, 0]] + "*" + "weight" + ")" + symNameList[indexSel] + "(" + variNameList[allIndex[indexSel, 1]] + "*" + "weight" + ")"
                else:
                    outStr = variNameList[allIndex[indexSel, 0]] + "*" + "weight" + symNameList[indexSel] + variNameList[allIndex[indexSel, 1]] + "*" + "weight"
            flag = indexSel + 1

        if paramNums[indexSel] == 1:
            #if symNum ==
            if hiddenNo > 0:
                tmp, tmpFlag = outPutSymRltAllWeightInn(variNum, symNum, hiddenNo - 1, paramNums, allIndex[indexSel, 0], indexList)
                symTmp = ""
                if 0 < tmpFlag:
                    symTmp = symNameList[tmpFlag - 1]
                if "+" == symTmp or "-" == symTmp or "/" == symTmp or "square" == symTmp:
                    if "square" == symNameList[indexSel]:
                        outStr = "(" + "(" + tmp + ")" + "*" + "weight" + ")" + "**2"
                    else:
                        outStr = symNameList[indexSel] + "(" + "(" + tmp + ")" + "*" + "weight" + ")"
                else:
                    if "square" == symNameList[indexSel]:
                        outStr = "(" + tmp + "*" + "weight" + ")" + "**2"
                    else:
                        outStr = symNameList[indexSel] + "(" + tmp + "*" + "weight" + ")"
            else:
                if "square" == symNameList[indexSel]:
                    outStr = "(" + variNameList[allIndex[indexSel, 0]] + "*" + "weight" + ")" + "**2"
                else:
                    outStr = symNameList[indexSel] + "(" + variNameList[allIndex[indexSel, 0]] + "*" + "weight" + ")"
            flag = indexSel + 1 #flag = 0
    return outStr, flag

def outPutSymRltAllWeight(variNum, symNum, paramNums, finalSel, indexList):
    if 2 == paramNums[3]:
        symNameList = ["+", "-", "*", "/", "sin", "cos", "exp", "log", "square"]
    else:
        symNameList = ["+", "-", "*", "sin", "cos", "exp", "log", "square"]

    hiddenLayerNum = len(indexList)
    hiddenNo = hiddenLayerNum - 1
    outStr, flag = outPutSymRltAllWeightInn(variNum, symNum, hiddenNo, paramNums, finalSel, indexList)
    symTmp = ""
    if 0 < flag:
        symTmp = symNameList[flag - 1]
    if "+" == symTmp or "-" == symTmp or "/" == symTmp or "square" == symTmp:
        outStr = "(" + outStr + ")" + "*" + "weight"
    else:
        outStr = outStr + "*" + "weight"
    return outStr

def outPutSymRltAllWeight2(variNum, symNum, paramNums, finalSelList, indexList):
    outStr1 = outPutSymRltAllWeight(variNum, symNum, paramNums, finalSelList[0], indexList)
    if 1 == len(finalSelList):
        outStr = outStr1
    else:
        outStr2 = outPutSymRltAllWeight(variNum, symNum, paramNums, finalSelList[1], indexList)
        outStr = outStr1 + "+" + outStr2
    return outStr

def outPutSymRltAllTrueWeightInn(variNum, symNum, hiddenNo, paramNums, indexSel, indexList, node2OpSList, modelParamsList):
    if 2 == paramNums[3]:
        symNameList = ["+", "-", "*", "/", "sin", "cos", "exp", "log", "square"]
    else:
        symNameList = ["+", "-", "*", "sin", "cos", "exp", "log", "square"]

    variNameListTmp = ["a", "b", "c", "d", "e"]
    variNameList = []
    for i in range(variNum):
        variNameList.append(variNameListTmp[i])
    variNameList.append("1")
    allIndex = indexList[hiddenNo]
    outStr = ""
    flag = 1
    if indexSel >= symNum:
        if hiddenNo > 0:
            outStr, flag = outPutSymRltAllTrueWeightInn(variNum, symNum, hiddenNo-1, paramNums, indexSel-symNum, indexList, node2OpSList, modelParamsList)
        else:
            outStr = variNameList[indexSel - symNum]
            flag = 0
    else:
        if paramNums[indexSel] == 2:
            opStart = node2OpSList[indexSel]
            w1 = modelParamsList[hiddenNo][opStart][allIndex[indexSel, 0]]
            w2 = modelParamsList[hiddenNo][opStart + 1][allIndex[indexSel, 1]]
            strWeight1 = str(w1.detach().numpy().tolist()) # ' '.join(map(str, w1.detach().numpy().tolist()))
            strWeight2 = str(w2.detach().numpy().tolist()) # ' '.join(map(str, w2.detach().numpy().tolist()))
            if w1 < 0:
                strWeight1 = "(" + strWeight1 + ")"
            if w2 < 0:
                strWeight2 = "(" + strWeight2 + ")"

            if hiddenNo > 0:
                tmp1, tmpFlag1 = outPutSymRltAllTrueWeightInn(variNum, symNum, hiddenNo - 1, paramNums, allIndex[indexSel, 0], indexList, node2OpSList, modelParamsList)
                tmp2 = symNameList[indexSel]
                tmp3, tmpFlag2 = outPutSymRltAllTrueWeightInn(variNum, symNum, hiddenNo - 1, paramNums, allIndex[indexSel, 1], indexList, node2OpSList, modelParamsList)

                symTmp1 = ""
                if 0 < tmpFlag1:
                    symTmp1 = symNameList[tmpFlag1 - 1]
                if "+" == symTmp1 or "-" == symTmp1 or "/" == symTmp1 or "square" == symTmp1:
                    tmp1 = "(" + tmp1 + ")" + "*" + strWeight1
                else:
                    tmp1 = tmp1 + "*" + strWeight1

                symTmp2 = ""
                if 0 < tmpFlag2:
                    symTmp2 = symNameList[tmpFlag2 - 1]
                if "+" == symTmp2 or "-" == symTmp2 or "/" == symTmp2 or "square" == symTmp2:
                    tmp3 = "(" + tmp3 + ")" + "*" + strWeight2
                else:
                    tmp3 = tmp3 + "*" + strWeight2

                if "/" == tmp2:
                    outStr = "(" + tmp1 + ")" + tmp2 + "(" + tmp3 + ")"
                else:
                    outStr = tmp1 + tmp2 + tmp3
            else:
                if "/" == symNameList[indexSel]:
                    outStr = "(" + variNameList[allIndex[indexSel, 0]] + "*" + strWeight1 + ")" + symNameList[indexSel] + "(" + variNameList[allIndex[indexSel, 1]] + "*" + strWeight2 + ")"
                else:
                    outStr = variNameList[allIndex[indexSel, 0]] + "*" + strWeight1 + symNameList[indexSel] + variNameList[allIndex[indexSel, 1]] + "*" + strWeight2
            flag = indexSel + 1

        if paramNums[indexSel] == 1:
            opStart = node2OpSList[indexSel]
            w = modelParamsList[hiddenNo][opStart][allIndex[indexSel, 0]]
            strWeight = str(w.detach().numpy().tolist()) # ' '.join(map(str, w.detach().numpy().tolist()))
            if w < 0:
                strWeight = "(" + strWeight + ")"
            if hiddenNo > 0:
                tmp, tmpFlag = outPutSymRltAllTrueWeightInn(variNum, symNum, hiddenNo - 1, paramNums, allIndex[indexSel, 0], indexList, node2OpSList, modelParamsList)

                symTmp = ""
                if 0 < tmpFlag:
                    symTmp = symNameList[tmpFlag - 1]
                if "+" == symTmp or "-" == symTmp or "/" == symTmp or "square" == symTmp:
                    if "square" == symNameList[indexSel]:
                        outStr = "(" + "(" + tmp + ")" + "*" + strWeight + ")" + "**2"
                    else:
                        outStr = symNameList[indexSel] + "(" + "(" + tmp + ")" + "*" + strWeight + ")"
                else:
                    if "square" == symNameList[indexSel]:
                        outStr ="(" + tmp + "*" + strWeight + ")" + "**2"
                    else:
                        outStr = symNameList[indexSel] + "(" + tmp + "*" + strWeight + ")"
            else:
                if "square" == symNameList[indexSel]:
                    outStr = "(" + variNameList[allIndex[indexSel, 0]] + "*" + strWeight + ")" + "**2"
                else:
                    outStr = symNameList[indexSel] + "(" + variNameList[allIndex[indexSel, 0]] + "*" + strWeight + ")"
            flag = indexSel + 1 #flag = 0
    return outStr, flag

def outPutSymRltAllTrueWeight(variNum, symNum, paramNums, finalSel, indexList, node2OpSList, modelParamsList):
    if 2 == paramNums[3]:
        symNameList = ["+", "-", "*", "/", "sin", "cos", "exp", "log", "square"]
    else:
        symNameList = ["+", "-", "*", "sin", "cos", "exp", "log", "square"]

    hiddenLayerNum = len(indexList)
    hiddenNo = hiddenLayerNum-1
    outStr, flag = outPutSymRltAllTrueWeightInn(variNum, symNum, hiddenNo, paramNums, finalSel, indexList, node2OpSList, modelParamsList)
    w = modelParamsList[hiddenNo+1][0][finalSel]
    strWeight = str(w.detach().numpy().tolist()) # ' '.join(map(str, w.detach().numpy().tolist()))
    if w < 0:
        strWeight = "(" + strWeight + ")"
    symTmp = ""
    if 0 < flag:
        symTmp = symNameList[flag - 1]
    if "+" == symTmp or "-" == symTmp or "/" == symTmp or "square" == symTmp:
        outStr = "(" + outStr + ")" + "*" + strWeight
    else:
        outStr = outStr + "*" + strWeight
    return outStr
def outPutSymRltAllTrueWeight2(variNum, symNum, paramNums, finalSelList, indexList, node2OpSList, modelParamsList):
    outStr1 = outPutSymRltAllTrueWeight(variNum, symNum, paramNums, finalSelList[0], indexList, node2OpSList, modelParamsList)
    if 1 == len(finalSelList):
        outStr = outStr1
    else:
        outStr2 = outPutSymRltAllTrueWeight(variNum, symNum, paramNums, finalSelList[1], indexList, node2OpSList, modelParamsList)
        outStr = outStr1 + "+" + outStr2
    return outStr

def outPutFullSymRltAllTrueWeight(variNum, symNum, paramLayerNo, paramNums, indexSel, node2OpSList, modelParamsList):
    if 2 == paramNums[3]:
        symNameList = ["+", "-", "*", "/", "sin", "cos", "exp", "log", "square"]
    else:
        symNameList = ["+", "-", "*", "sin", "cos", "exp", "log", "square"]

    variNameListTmp = ["a", "b", "c", "d", "e"]
    variNameList = []
    for i in range(variNum):
        variNameList.append(variNameListTmp[i])
    variNameList.append("1")
    outStr = ""
    flag = 1
    hiddenLayerNum = len(modelParamsList) - 1
    currLayerParam = modelParamsList[paramLayerNo]
    #variDim = currLayerParam.size()[0]
    inputDim = currLayerParam.size()[1]
    if indexSel >= symNum: #id operator
        if paramLayerNo > 0:
            outStr, flag = outPutFullSymRltAllTrueWeight(variNum, symNum, paramLayerNo-1, paramNums, indexSel-symNum, node2OpSList, modelParamsList)
        else:
            outStr = variNameList[indexSel - symNum]
            flag = 0
    else:
        if paramNums[indexSel] == 2 and paramLayerNo != hiddenLayerNum:
            opStart = node2OpSList[indexSel]
            funStr1 = ""
            for i in range(inputDim):
                if 0 == currLayerParam[opStart][i]:
                    continue
                w1 = currLayerParam[opStart][i]
                strWeight1 = str(w1.detach().numpy().tolist())
                if w1 < 0:
                    strWeight1 = "(" + strWeight1 + ")"
                if 0 < paramLayerNo:
                    tmp1, tmpFlag1 = outPutFullSymRltAllTrueWeight(variNum, symNum, paramLayerNo - 1, paramNums, i, node2OpSList, modelParamsList)
                else:
                    tmp1 = variNameList[i]
                tmp1 = "(" + tmp1 + ")" + "*" + strWeight1
                if "" == funStr1:
                    funStr1 = tmp1
                else:
                    funStr1 = funStr1 + "+" + tmp1
            if "" == funStr1:
                funStr1 = "0"


            funStr2 = ""
            for i in range(inputDim):
                if 0 == currLayerParam[opStart+1][i]:
                    continue
                w2 = currLayerParam[opStart+1][i]
                strWeight2 = str(w2.detach().numpy().tolist())
                if w2 < 0:
                    strWeight2 = "(" + strWeight2 + ")"
                if 0 < paramLayerNo:
                    tmp2, tmpFlag2 = outPutFullSymRltAllTrueWeight(variNum, symNum, paramLayerNo - 1, paramNums, i, node2OpSList, modelParamsList)
                else:
                    tmp2 = variNameList[i]
                tmp2 = "(" + tmp2 + ")" + "*" + strWeight2
                if "" == funStr2:
                    funStr2 = tmp2
                else:
                    funStr2 = funStr2 + "+" + tmp2
            if "" == funStr2:
                funStr2 = "0"

            tmp2 = symNameList[indexSel]
            if "/" == tmp2:
                outStr = "(" + funStr1 + ")" + tmp2 + "(" + funStr2 + ")"
            else:
                outStr = funStr1 + tmp2 + funStr2
            flag = indexSel + 1

        if paramNums[indexSel] == 1 or paramLayerNo == hiddenLayerNum:
            opStart = node2OpSList[indexSel]
            if paramLayerNo == hiddenLayerNum:
                opStart = 0
            funStr = ""
            for i in range(inputDim):
                if 0 == currLayerParam[opStart][i]:
                    continue
                w = currLayerParam[opStart][i]
                strWeight = str(w.detach().numpy().tolist())
                if w < 0:
                    strWeight = "(" + strWeight + ")"
                if 0 < paramLayerNo:
                    tmp, tmpFlag = outPutFullSymRltAllTrueWeight(variNum, symNum, paramLayerNo - 1, paramNums, i, node2OpSList, modelParamsList)
                else:
                    tmp = variNameList[i]
                tmp = "(" + tmp + ")" + "*" + strWeight
                if "" == funStr:
                    funStr = tmp
                else:
                    funStr = funStr + "+" + tmp
            if "" == funStr:
                funStr = "0"

            if paramLayerNo == hiddenLayerNum:
                outStr = funStr
            elif "square" == symNameList[indexSel]:
                outStr = "(" + funStr + ")" + "**2"
            else:
                outStr = symNameList[indexSel] + "(" + funStr + ")"
            flag = indexSel + 1 #flag = 0
    return outStr, flag

class NetNew(nn.Module):
    def __init__(self, variNum, constNum, hiddenLayerNum, calList, opNumList):
        super(NetNew, self).__init__()
        calNum = len(calList)
        variDim = sum(opNumList)
        # an affine operation: y = Wx + b
        self.nanFlag = 0
        self.isCheckNan = 1

        self.hiddenLayerNum = hiddenLayerNum
        inputDim = variNum + constNum
        self.fc1 = nn.Linear(inputDim, variDim, bias=False)

        if 1 < hiddenLayerNum:
            inputDim = inputDim + calNum
            self.fc2 = nn.Linear(inputDim, variDim, bias=False)

        if 2 < hiddenLayerNum:
            inputDim = inputDim + calNum
            self.fc3 = nn.Linear(inputDim, variDim, bias=False)

        if 3 < hiddenLayerNum:
            inputDim = inputDim + calNum
            self.fc4 = nn.Linear(inputDim, variDim, bias=False)

        if 4 < hiddenLayerNum:
            inputDim = inputDim + calNum
            self.fc5 = nn.Linear(inputDim, variDim, bias=False)

        if 5 < hiddenLayerNum:
            inputDim = inputDim + calNum
            self.fc6 = nn.Linear(inputDim, variDim, bias=False)
        #20230204
        if 6 < hiddenLayerNum:
            inputDim = inputDim + calNum
            self.fc7 = nn.Linear(inputDim, variDim, bias=False)

        if 7 < hiddenLayerNum:
            inputDim = inputDim + calNum
            self.fc8 = nn.Linear(inputDim, variDim, bias=False)

        inputDim = inputDim + calNum
        self.fcFinal = nn.Linear(inputDim, 1, bias=False)
        self.calList = calList
        self.opNumList = opNumList

    def SetIsCheckNan(self, isCheck):
        self.isCheckNan = isCheck

    def forwardTmp(self, x1, x2):
        mltFlag = 0
        divFlag = 0
        expFlag = 0
        sqreFlag = 0

        outList = []
        calNum = len(self.calList)
        s = 0
        for i in range(calNum):
            if '+' == self.calList[i]:
                outList.append((x2[:, s] + x2[:, s+1]).unsqueeze(1))
                s = s + 2
                continue
            if '-' == self.calList[i]:
                outList.append((x2[:, s] - x2[:, s+1]).unsqueeze(1))
                s = s + 2
                continue
            if '*' == self.calList[i]:
                if 1 == self.isCheckNan:
                    ############################### protect mul
                    tmpMlt = x2[:, s] * x2[:, s + 1]
                    num1 = tmpMlt.size(0)
                    flag = 0
                    maxMlt = 99999999
                    for j in range(num1):
                        if abs(tmpMlt[j]) >= maxMlt:
                            flag = 1
                            self.nanFlag = 1
                            mltFlag = 1
                            break
                    if 1 == flag:
                        tmpTensor = torch.zeros(num1, dtype=torch.float32)
                        for j in range(num1):
                            if abs(tmpMlt[j]) >= maxMlt:
                                with torch.no_grad():
                                    tmp = abs(maxMlt/tmpMlt[j])
                                tmpTensor[j] = tmpMlt[j]*tmp
                            else:
                                tmpTensor[j] = tmpMlt[j]
                        tmpRlt = tmpTensor
                    else:
                        tmpRlt = tmpMlt
                    outList.append((tmpRlt).unsqueeze(1))
                    ###############################
                else:
                    outList.append((x2[:, s] * x2[:, s+1]).unsqueeze(1))
                s = s + 2
                continue
            if '/' == self.calList[i]:
                if 1 == self.isCheckNan:
                    ################################################# protect div
                    num1 = x2[:, s + 1].size(0)
                    flag = 0
                    minV = 0.0001  # 0.000001
                    tmpX2 = torch.zeros(num1, dtype=torch.float32)
                    for j in range(num1):
                        if x2[j, s + 1] == 0:
                            tmpX2[j] = x2[j, s + 1] + minV
                            self.nanFlag = 1
                        else:
                            tmpX2[j] = x2[j, s + 1]
                    tmpDiv = x2[:, s] / tmpX2
                    maxDiv = 9999# 99
                    for j in range(num1):
                        if abs(tmpDiv[j]) >= maxDiv:
                            flag = 1
                            self.nanFlag = 1
                            divFlag = 1
                            break
                    if 1 == flag:
                        tmpTensor = torch.zeros(num1, dtype=torch.float32)
                        for j in range(num1):
                            if abs(tmpDiv[j]) >= maxDiv:
                                with torch.no_grad():
                                    tmp = abs(maxDiv / tmpDiv[j])
                                tmpTensor[j] = tmpDiv[j] * tmp
                            else:
                                tmpTensor[j] = tmpDiv[j]
                        tmpRlt = tmpTensor
                    else:
                        tmpRlt = tmpDiv
                    outList.append((tmpRlt).unsqueeze(1))
                    #################################################
                else:
                    outList.append((x2[:, s] / (x2[:, s+1] + 0.00)).unsqueeze(1))
                s = s + 2
                continue
            if 'sin' == self.calList[i]:
                outList.append((torch.sin(x2[:, s])).unsqueeze(1))
                s = s + 1
                continue
            if 'cos' == self.calList[i]:
                outList.append((torch.cos(x2[:, s])).unsqueeze(1))
                s = s + 1
                continue
            if 'exp' == self.calList[i]: #protect exp
                if 1 == self.isCheckNan:
                    ###############################
                    tmpX2 = x2[:, s]
                    num1 = tmpX2.size(0)
                    flag = 0
                    maxExp = 17  # 12
                    for j in range(num1):
                        if tmpX2[j] >= maxExp:
                            flag = 1
                            self.nanFlag = 1
                            expFlag = 1
                            break
                    if 1 == flag:
                        tmpTensor = torch.zeros(num1, dtype=torch.float32)
                        for j in range(num1):
                            if tmpX2[j] >= maxExp:
                                with torch.no_grad():
                                    tmp = maxExp/tmpX2[j]
                                tmpTensor[j] = tmpX2[j]*tmp
                            else:
                                tmpTensor[j] = tmpX2[j]
                        tmpRlt = tmpTensor
                    else:
                        tmpRlt = x2[:, s]
                    outList.append((torch.exp(tmpRlt)).unsqueeze(1))
                    #################################################
                else:
                    outList.append((torch.exp(x2[:, s])).unsqueeze(1))
                s = s + 1
                continue
            if 'log' == self.calList[i]:
                outList.append((torch.log(abs(x2[:, s])+0)).unsqueeze(1))
                s = s + 1
            if "square" == self.calList[i]: #protect square
                if 1 == self.isCheckNan:
                    ###############################
                    tmpSqre = x2[:, s] * x2[:, s]
                    num1 = tmpSqre.size(0)
                    flag = 0
                    maxSqre = 99999999
                    for j in range(num1):
                        if abs(tmpSqre[j]) >= maxSqre:
                            flag = 1
                            self.nanFlag = 1
                            sqreFlag = 1
                            break
                    if 1 == flag:
                        tmpTensor = torch.zeros(num1, dtype=torch.float32)
                        for j in range(num1):
                            if abs(tmpSqre[j]) >= maxSqre:
                                with torch.no_grad():
                                    tmp = abs(maxSqre/tmpSqre[j])
                                tmpTensor[j] = tmpSqre[j]*tmp
                            else:
                                tmpTensor[j] = tmpSqre[j]
                        tmpRlt = tmpTensor
                    else:
                        tmpRlt = tmpSqre
                    outList.append((tmpRlt).unsqueeze(1))
                    ###############################
                else:
                    outList.append((x2[:, s] * x2[:, s]).unsqueeze(1))
                s = s + 1
                # outList.append((torch.square(x2[:, s])).unsqueeze(1))
                # s = s + 1

        # x = torch.cat(outList.append(x1), 1)
        # x = torch.cat([outList, x1], 1)
        x = outList[0]
        for i in range(len(outList)-1):
            x = torch.cat([x, outList[i+1]], 1)
        x = torch.cat([x, x1], 1)

        return x, mltFlag, divFlag, expFlag, sqreFlag
    def forward(self, x):
        self.nanFlag = 0
        layerOutList = []
        mltFlagList = []
        divFlagList = []
        expFlagList = []
        sqreFlagList = []
        paraLayerNo = 0
        # layer 1
        x2 = self.fc1(x)
        y2, mltFlag, divFlag, expFlag, sqreFlag = self.forwardTmp(x, x2)
        if 1 == mltFlag:
            mltFlagList.append(paraLayerNo)
        if 1 == divFlag:
            divFlagList.append(paraLayerNo)
        if 1 == expFlag:
            expFlagList.append(paraLayerNo)
        if 1 == sqreFlag:
            sqreFlagList.append(paraLayerNo)
        layerOutList.append(y2)


        # layer 2
        if 1 < self.hiddenLayerNum:
            x3 = self.fc2(y2)
            y3, mltFlag, divFlag, expFlag, sqreFlag = self.forwardTmp(y2, x3)
            paraLayerNo = paraLayerNo+1
            if 1 == mltFlag:
                mltFlagList.append(paraLayerNo)
            if 1 == divFlag:
                divFlagList.append(paraLayerNo)
            if 1 == expFlag:
                expFlagList.append(paraLayerNo)
            if 1 == sqreFlag:
                sqreFlagList.append(paraLayerNo)
            layerOutList.append(y3)

        # layer 3
        if 2 < self.hiddenLayerNum:
            x4 = self.fc3(y3)
            y4, mltFlag, divFlag, expFlag, sqreFlag = self.forwardTmp(y3, x4)
            paraLayerNo = paraLayerNo + 1
            if 1 == mltFlag:
                mltFlagList.append(paraLayerNo)
            if 1 == divFlag:
                divFlagList.append(paraLayerNo)
            if 1 == expFlag:
                expFlagList.append(paraLayerNo)
            if 1 == sqreFlag:
                sqreFlagList.append(paraLayerNo)
            layerOutList.append(y4)

        # layer 4
        if 3 < self.hiddenLayerNum:
            x5 = self.fc4(y4)
            y5, mltFlag, divFlag, expFlag, sqreFlag = self.forwardTmp(y4, x5)
            paraLayerNo = paraLayerNo + 1
            if 1 == mltFlag:
                mltFlagList.append(paraLayerNo)
            if 1 == divFlag:
                divFlagList.append(paraLayerNo)
            if 1 == expFlag:
                expFlagList.append(paraLayerNo)
            if 1 == sqreFlag:
                sqreFlagList.append(paraLayerNo)
            layerOutList.append(y5)

        # layer 5
        if 4 < self.hiddenLayerNum:
            x6 = self.fc5(y5)
            y6, mltFlag, divFlag, expFlag, sqreFlag = self.forwardTmp(y5, x6)
            paraLayerNo = paraLayerNo + 1
            if 1 == mltFlag:
                mltFlagList.append(paraLayerNo)
            if 1 == divFlag:
                divFlagList.append(paraLayerNo)
            if 1 == expFlag:
                expFlagList.append(paraLayerNo)
            if 1 == sqreFlag:
                sqreFlagList.append(paraLayerNo)
            layerOutList.append(y6)

        # layer 6
        if 5 < self.hiddenLayerNum:
            x7 = self.fc6(y6)
            y7, mltFlag, divFlag, expFlag, sqreFlag = self.forwardTmp(y6, x7)
            paraLayerNo = paraLayerNo + 1
            if 1 == mltFlag:
                mltFlagList.append(paraLayerNo)
            if 1 == divFlag:
                divFlagList.append(paraLayerNo)
            if 1 == expFlag:
                expFlagList.append(paraLayerNo)
            if 1 == sqreFlag:
                sqreFlagList.append(paraLayerNo)
            layerOutList.append(y7)

        #layer 7
        if 6 < self.hiddenLayerNum:
            x8 = self.fc7(y7)
            y8, mltFlag, divFlag, expFlag, sqreFlag = self.forwardTmp(y7, x8)
            paraLayerNo = paraLayerNo + 1
            if 1 == mltFlag:
                mltFlagList.append(paraLayerNo)
            if 1 == divFlag:
                divFlagList.append(paraLayerNo)
            if 1 == expFlag:
                expFlagList.append(paraLayerNo)
            if 1 == sqreFlag:
                sqreFlagList.append(paraLayerNo)
            layerOutList.append(y8)
        #layer 8
        if 7 < self.hiddenLayerNum:
            x9 = self.fc8(y8)
            y9, mltFlag, divFlag, expFlag, sqreFlag = self.forwardTmp(y8, x9)
            paraLayerNo = paraLayerNo + 1
            if 1 == mltFlag:
                mltFlagList.append(paraLayerNo)
            if 1 == divFlag:
                divFlagList.append(paraLayerNo)
            if 1 == expFlag:
                expFlagList.append(paraLayerNo)
            if 1 == sqreFlag:
                sqreFlagList.append(paraLayerNo)
            layerOutList.append(y9)

        # output layer
        if 1 == self.hiddenLayerNum:
            xOut = self.fcFinal(y2)

        if 2 == self.hiddenLayerNum:
            xOut = self.fcFinal(y3)

        if 3 == self.hiddenLayerNum:
            xOut = self.fcFinal(y4)

        if 4 == self.hiddenLayerNum:
            xOut = self.fcFinal(y5)

        if 5 == self.hiddenLayerNum:
            xOut = self.fcFinal(y6)

        if 6 == self.hiddenLayerNum:
            xOut = self.fcFinal(y7)

        if 7 == self.hiddenLayerNum:
            xOut = self.fcFinal(y8)

        if 8 == self.hiddenLayerNum:
            xOut = self.fcFinal(y9)

        mltDivExpFlagList = [mltFlagList, divFlagList, expFlagList, sqreFlagList]

        return xOut, self.nanFlag, mltDivExpFlagList

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features = num_features * s
        return num_features

class MyDataset(data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):# return tensor
        img, target = self.images[index], self.labels[index]
        return img, target

    def __len__(self):
        return len(self.images)

def PruneABSTh(model, absTh):
    with torch.no_grad():
        count = 0
        totalNum = 0
        modelParamsList = list(model.parameters())
        layerNum = len(modelParamsList)
        for i in range(layerNum):
            currLayerParam = modelParamsList[i]
            variDim = currLayerParam.size()[0]
            inputDim = currLayerParam.size()[1]
            totalNum = totalNum + variDim*inputDim
            for j in range(variDim):
                for k in range(inputDim):
                    if abs(currLayerParam[j][k]) < absTh:
                        currLayerParam[j][k] = 0
                        count = count + 1
        return totalNum, count

def ProcessPruneFull(modelTmp, variNum, symNum, paramNums, node2OpSList, input, target):
    modelParamsList = list(modelTmp.parameters())
    hiddenLayerNum = len(modelParamsList)-1
    outStrTrue, flag = outPutFullSymRltAllTrueWeight(variNum, symNum, hiddenLayerNum, paramNums, 0, node2OpSList, modelParamsList)
    outStrTrueSymPy = sympify(outStrTrue)
    outStrTrueSimple = simplify(outStrTrueSymPy)
    outStrTrueSimpleExpand = expand(outStrTrueSimple)
    print("Full outStrTrue:", outStrTrue)
    print("Full outStrTrueSimple:", outStrTrueSimple)
    print("Full outStrTrueSimpleExpand:", outStrTrueSimpleExpand)

    minConstList, bfgsError, codeStrBFGS = SolveByBFGS(input[0:20, 0:variNum].numpy(), target[0:20, :].squeeze(1).numpy(), outStrTrueSimpleExpand)
    minConstList, bfgsError, codeStrBFGS = ProcessTail(codeStrBFGS, minConstList, bfgsError, input[0:20, 0:variNum].numpy(), target[0:20, :].squeeze(1).numpy())
    print("Full BFGS error:", bfgsError)
    print("Full minConstList:", minConstList)
    # if bfgsError < 1.0e-6 and -1 != bfgsError: #-8
    #     lossTmp = bfgsError

    if 0 < len(minConstList):
        test(str(outStrTrueSimpleExpand), variNum, input, target)
    return bfgsError, codeStrBFGS
def PruneReverse(model, variNum, symNum, paramNums, node2OpSList, criterion, input, target):
    modelTmp = copy.deepcopy(model)
    maxPrunNodeNum = 1
    step = 1
    absTh = 0.01
    totalNum, count = PruneABSTh(modelTmp, absTh)

    output, nanFlag, mltDivExpFlagList = modelTmp(input)
    loss = criterion(output, target)
    print("PruneReverseNet loss:", loss)
    pruneReverseLoss = -1
    reverseOutStr = ""
    print("totalNum, count, totalNum-count:", totalNum, count, totalNum-count)
    if 30 >= totalNum-count:
        pruneReverseLoss, reverseOutStr = ProcessPruneFull(modelTmp, variNum, symNum, paramNums, node2OpSList, input, target)
    return pruneReverseLoss, reverseOutStr

def MaskOperator(model, symNum, op2NodeList, maskOpIDList):
    with torch.no_grad():
        modelParamsList = list(model.parameters())
        paramLayerNum = len(modelParamsList)
        for i in range(paramLayerNum):
            currLayerParam = modelParamsList[i]
            variDim = currLayerParam.size()[0]
            for j in range(variDim):
                opId = op2NodeList[j]
                if opId in maskOpIDList:
                    #currLayerParam[j, :] = 0 # No process is ok
                    haha = 1
                else:
                    addNum = 0
                    for k in range(i):
                        for mskId in maskOpIDList:
                            currLayerParam[j, mskId+addNum] = 0
                        addNum = addNum + symNum

#FixWeight on subnet
def FixWeight(model, selLayerList):
    with torch.no_grad():
        modelParamsList = list(model.parameters())
        paramLayerNum = len(modelParamsList)
        for i in range(paramLayerNum):
            layerNo = paramLayerNum - i - 1 #from back to front
            currLayerParam = modelParamsList[layerNo]
            currLayerParamBack = currLayerParam.clone()
            for k in range(len(selLayerList[i])):
                m = selLayerList[i][k][0]
                currLayerParam[m, :] = 0
            for k in range(len(selLayerList[i])):
                m = selLayerList[i][k][0]
                n = selLayerList[i][k][1]
                currLayerParam[m, n] = currLayerParamBack[m, n]

def GetIndexListFromSelLayerList(model, symNum, paramNums, selLayerList, op2NodeList):
    with torch.no_grad():
        modelParamsList = list(model.parameters())
        paramLayerNum = len(modelParamsList)
        indexList = []

        finalSelList = []
        for i in range(len(selLayerList[0])):
            finalSelList.append(selLayerList[0][i][1])

        for j in range(paramLayerNum-1):
            allIndex = torch.zeros(symNum, 2, dtype=torch.int)
            allIndex[:] = -1
            layerIndex = paramLayerNum - j - 1
            for k in range(len(selLayerList[layerIndex])):
                m = selLayerList[layerIndex][k][0]
                nodeNo = op2NodeList[m]

                n = selLayerList[layerIndex][k][1]
                if 2 == paramNums[nodeNo] and 0 != (m%2):
                    allIndex[nodeNo, 1] = n
                else:
                    allIndex[nodeNo, 0] = n
            indexList.append(allIndex)

        return finalSelList, indexList

def GetSincosFlagList(variConstNum, allLayerNum, biSymNum, uniSymNum):
    with torch.no_grad():
        symNum = biSymNum + uniSymNum
        sincosFlagList = []
        #maxNodeNum = variConstNum + (allLayerNum - 2) * symNum
        nodeNum = variConstNum
        for i in range(allLayerNum):
            if allLayerNum-1 == i:
                nodeNum = 1
            tmpFlag = torch.zeros(nodeNum, dtype=torch.int)
            if 0 == i or allLayerNum-1 == i:
                sincosFlagList.append(tmpFlag)
                nodeNum = nodeNum + symNum
                continue
            tmpNum = min(uniSymNum, 4)
            for j in range(tmpNum):
                tmpFlag[biSymNum+j] = 1
            for j in range(nodeNum-symNum):
                tmpFlag[symNum + j] = sincosFlagList[i-1][j]

            sincosFlagList.append(tmpFlag)
            nodeNum = nodeNum + symNum

        return sincosFlagList

############################################################################BeamSearch op start
def FixWeightTmp(model, selLayerListTmp):
    with torch.no_grad():
        modelParamsList = list(model.parameters())
        paramLayerNum = len(modelParamsList)
        currLayerNum = len(selLayerListTmp)
        for i in range(currLayerNum):
            layerNo = paramLayerNum - i - 1 #from back to front
            currLayerParam = modelParamsList[layerNo]
            currLayerParamBack = currLayerParam.clone()
            for k in range(len(selLayerListTmp[i])):
                m = selLayerListTmp[i][k][0]
                currLayerParam[m, :] = 0
            for k in range(len(selLayerListTmp[i])):
                m = selLayerListTmp[i][k][0]
                n = selLayerListTmp[i][k][1]
                currLayerParam[m, n] = currLayerParamBack[m, n]
#SortTensorOrg according loss
def SortTensorOrg(inputValue, count, sortIndex, bSize):
    with torch.no_grad():
        outValue = inputValue[0:count, :]
        outValue = outValue[outValue[:, sortIndex].sort(descending=False)[1]]
        return outValue

def SampleProb(currProbIn):
    with torch.no_grad():
        count = currProbIn.size(0) # len(currVoteList)
        totalNum = 0
        for i in range(count):
            totalNum = totalNum + currProbIn[i]
        currProbList = []
        for i in range(count):
            tmp = currProbIn[i]/totalNum
            currProbList.append(tmp)
        randNum = torch.rand(1, 1)
        tmp1 = 0.0
        for i in range(count):
            tmp1 = tmp1 + currProbList[i] #prob
            if tmp1 >= randNum:
                break
        currProbIn[i] = 0
        index = int(i)
        return index

#SortTensorRand
def SortTensorRand(inputValue, count, sortIndex, bSize):
    with torch.no_grad():
        inputTmp = - inputValue[0:count, sortIndex]
        inputTmp = F.normalize(inputTmp, dim=0)
        currProb = F.softmax(inputTmp, dim=0)
        num = min(bSize, count)
        sampleList = []
        for i in range(num):
            index = SampleProb(currProb)
            if index in sampleList:
                print("Same index haha!!!!!!!!!!!!!!!!!!!!")
            sampleList.append(index)

        dim1 = inputValue.size(1)
        outValue = torch.zeros(num, dim1, dtype=torch.float32)
        count = 0
        for index in sampleList:
            outValue[count] = inputValue[index] #inputValue[index, :]
            count = count + 1

        return outValue

def PreProcessData1(inputValue, count, sortIndex):
    with torch.no_grad():
        dim1 = inputValue.size(1)

        validNum = 0
        for i in range(count):
            if not torch.isnan(inputValue[i, sortIndex]):
                validNum = validNum + 1  # inputTmp[i] = float('-inf')
        if 0 < validNum:
            if validNum == count:
                inputValueTmp = inputValue
            else:
                inputValueTmp = torch.zeros(validNum, dim1, dtype=torch.float32)
                num = 0
                for i in range(count):
                    if not torch.isnan(inputValue[i, sortIndex]):
                        inputValueTmp[num] = inputValue[i]
                        num = num + 1
        else:
            inputValueTmp = inputValue
            inputValueTmp[:, sortIndex] = 1
            validNum = count
        return inputValueTmp, validNum

#process nan
def PreProcessData2(inputValue, count, sortIndex):
    with torch.no_grad():
        for i in range(count):
            if torch.isnan(inputValue[i, sortIndex]):
                inputValue[i, sortIndex] = 9999999999

        return inputValue, count
def SortTensor(inputValue, count, sortIndex, bSize, currLayerNo):
    with torch.no_grad():
        #inputValueTmp, validNum = PreProcessData1(inputValue, count, sortIndex)
        inputValueTmp, validNum = PreProcessData2(inputValue, count, sortIndex)  #process nan
        #sort
        if 2 <= currLayerNo:
            outValue = SortTensorOrg(inputValueTmp, validNum, sortIndex, bSize)
        else:
            outValue = SortTensorRand(inputValueTmp, validNum, sortIndex, bSize)
        return outValue

def ProcessFinal2(model, currLayerNo, currLayerParam, sincosFlagList, criterion, input, target, bSize):
    inputDim = currLayerParam.size()[1]
    currLayerParamBack = currLayerParam.clone()

    candList = []
    lossList = []
    sincosFlagCandList = []
    opStart = 0
    currLayerParam[opStart, :] = 0
    currSel = torch.zeros(inputDim*inputDim, 3, dtype=torch.float32)
    count = 0
    for t in range(inputDim):
        if 0 == currLayerParamBack[opStart, t]:
            continue
        currLayerParam[opStart, t] = currLayerParamBack[opStart, t]
        for k in range(t, inputDim):
            # if t == k:
            #     continue
            if 0 == currLayerParamBack[opStart, k]:
                continue
            currLayerParam[opStart, k] = currLayerParamBack[opStart, k]
            output, nanFlag, mltDivExpFlagList = model(input)
            loss = criterion(output, target)
            currSel[count, 0] = t
            currSel[count, 1] = k
            currSel[count, 2] = loss
            count = count + 1
            if t != k:
                currLayerParam[opStart, k] = 0
        currLayerParam[opStart, t] = 0
    currSel = SortTensor(currSel, count, 2, bSize, currLayerNo)
    count = currSel.size(0)
    num = min(bSize, count)
    for ii in range(num):
        currCand = []
        currLayerList = []
        index1 = int(currSel[ii, 0])
        index2 = int(currSel[ii, 1])
        currLayerList.append([0, index1])
        if index1 != index2:
            currLayerList.append([0, index2])
        currCand.append(currLayerList)
        candList.append(currCand)
        lossList.append(currSel[ii, 2])
        sincosFlagCandList.append(sincosFlagList)
    currLayerParam[opStart, :] = currLayerParamBack[opStart, :]
    return candList, lossList, sincosFlagCandList

def GetNodeList(symNum, currCand):
    nodeList = [0]
    currLayerNum = len(currCand)
    for i in range(currLayerNum):
        nodeListTmp = []
        for j in range(len(currCand[i])):
            if not currCand[i][j][1] in nodeListTmp:
                nodeListTmp.append(currCand[i][j][1])
        for j in range(len(nodeList)):
            if symNum <= nodeList[j]:
                if not (nodeList[j]-symNum) in nodeListTmp:
                    nodeListTmp.append(nodeList[j]-symNum)
        nodeList = nodeListTmp

    return nodeList

def UpdateSincosFlag(symNum, op2NodeList, sincosFlag, selLayerList):
    selLayerNum = len(selLayerList)
    currLayerNo = len(sincosFlag) - selLayerNum - 1
    nodeList = GetNodeList(symNum, selLayerList[0:selLayerNum-1])
    for nodeNo in nodeList:
        if symNum <= nodeNo and 1 == sincosFlag[currLayerNo+1][nodeNo]:
            sincosFlag[currLayerNo][nodeNo-symNum] = 1
    for sel in selLayerList[selLayerNum-1]:
        opNo = sel[0]
        node1 = op2NodeList[opNo]
        node2 = sel[1]
        if 1 == sincosFlag[currLayerNo+1][node1]:
            sincosFlag[currLayerNo][node2] = 1
    return sincosFlag

def processOneCandCurrLayer(modelTmp, criterion, symNum, currLayerNo, cand, sincosFlagList, node2OpSList, paramNums, input, target, bSize):
    modelParamsList = list(modelTmp.parameters())
    paramLayerNum = len(modelParamsList)
    currLayerParam = modelParamsList[currLayerNo]
    inputDim = currLayerParam.size()[1]
    currLayerParamBack = currLayerParam.clone()

    currCandLayerList = [[]]
    lossList = []
    currLayerSincosFlag = copy.deepcopy(sincosFlagList[currLayerNo])
    nodeList = GetNodeList(symNum, cand) #candList[i]
    for j in range(len(nodeList)):
        nodeNo = nodeList[j]
        if nodeNo >= symNum:
            # if 1 == sincosFlagList[currLayerNo + 1][nodeNo]:
            #     sincosFlagList[currLayerNo][nodeNo - symNum] = 1
            continue

        opStart = node2OpSList[nodeNo]
        opNum = paramNums[nodeNo]
        if paramLayerNum - 1 == currLayerNo:# and 1 == finalNum:
            opNum = 1

        for k in range(opNum):
            currSel = torch.zeros(inputDim * bSize, 3, dtype=torch.float32)
            count = 0
            for kk in range(len(currCandLayerList)):
                for t in range(len(currCandLayerList[kk])):
                    opNoTmp = currCandLayerList[kk][t][0]
                    nodeNoTmp = currCandLayerList[kk][t][1]
                    currLayerParam[opNoTmp, :] = 0
                    currLayerParam[opNoTmp, nodeNoTmp] = currLayerParamBack[opNoTmp, nodeNoTmp]
                currLayerParam[opStart+k, :] = 0
                for t in range(inputDim):
                    if 1 == sincosFlagList[currLayerNo + 1][nodeNo] and 1 == currLayerSincosFlag[t]:  # sincosFlagList[currLayerNo][t]:
                        continue
                    if 1 == nodeNo and 1 == k and ([opStart, t] in currCandLayerList[kk]):
                        continue
                    if 0 == currLayerParamBack[opStart + k, t]:
                        continue
                    currLayerParam[opStart + k, t] = currLayerParamBack[opStart + k, t]
                    output, nanFlag, mltDivExpFlagList = modelTmp(input)
                    loss = criterion(output, target)
                    currSel[count, 0] = kk
                    currSel[count, 1] = t
                    currSel[count, 2] = loss
                    count = count + 1
                    currLayerParam[opStart + k, t] = 0
                currLayerParam[opStart + k, :] = currLayerParamBack[opStart + k, :]

            currCandLayerListTmp = []
            lossListTmp = []
            currSel = SortTensor(currSel, count, 2, bSize, currLayerNo)
            count = currSel.size(0)
            num = min(bSize, count)
            for ii in range(num):
                index1 = int(currSel[ii, 0])
                currLayerList = copy.deepcopy(currCandLayerList[index1])
                index2 = int(currSel[ii, 1])
                currLayerList.append([opStart + k, index2])
                currCandLayerListTmp.append(currLayerList)
                lossListTmp.append(currSel[ii, 2])
            currCandLayerList = currCandLayerListTmp
            lossList = lossListTmp
    return currCandLayerList, lossList

def ProcessAllCandCurrLayer(model, criterion, paramNums, symNum, currLayerNo, node2OpSList, op2NodeList, bSize, candList, lossList, sincosFlagCandList, input, target):
    candListNew = []
    lossListNew = []
    candId1List = []
    for i in range(len(candList)):
        modelTmp = copy.deepcopy(model)
        FixWeightTmp(modelTmp, candList[i])
        currCandLayerList, currCandLayerlossList = processOneCandCurrLayer(modelTmp, criterion, symNum, currLayerNo, candList[i], sincosFlagCandList[i], node2OpSList, paramNums, input, target, bSize)
        #candList[i] append!!!
        for j in range(len(currCandLayerList)):
            candTmp = copy.deepcopy(candList[i])
            candTmp.append(currCandLayerList[j])
            candListNew.append(candTmp)
            if [] != currCandLayerlossList:
                lossListNew.append(currCandLayerlossList[j])
            else: #all id
                lossListNew.append(lossList[i])
            candId1List.append(i)
    allNum = len(candListNew)
    currSel = torch.zeros(allNum, 3, dtype=torch.float32)
    count = 0
    for i in range(allNum):
        currSel[count, 0] = candId1List[i]
        currSel[count, 1] = i
        currSel[count, 2] = lossListNew[i]
        count = count + 1
    currSel = SortTensor(currSel, count, 2, bSize, currLayerNo)
    count = currSel.size(0)
    num = min(bSize, count)
    candList = []
    sincosFlagCandListTmp = []
    lossList = []
    for i in range(num):
        candId1 = int(currSel[i, 0])
        candId2 = int(currSel[i, 1])
        candTmp = candListNew[candId2]
        candList.append(candTmp)
        sincosFlagTmp = UpdateSincosFlag(symNum, op2NodeList, sincosFlagCandList[candId1], candTmp)
        sincosFlagCandListTmp.append(sincosFlagTmp)
        lossList.append(currSel[i, 2])

    sincosFlagCandList = sincosFlagCandListTmp
    return candList, lossList, sincosFlagCandList

def PruneGreedOpBeamSearch1(model, paramNums, symNum, node2OpSList, op2NodeList, sincosFlagList, input, target, finalNum, bSize):
    with torch.no_grad():
        criterion = nn.MSELoss()
        modelParamsList = list(model.parameters())
        paramLayerNum = len(modelParamsList)

        candList = [[]]
        lossList = []
        sincosFlagCandList = [sincosFlagList]
        for i in range(paramLayerNum):
            currLayerNo = paramLayerNum - i - 1 #from back to front
            currLayerParam = modelParamsList[currLayerNo]

            if 0 == i and 2 == finalNum:
                candList, lossList, sincosFlagCandList = ProcessFinal2(model, currLayerNo, currLayerParam, sincosFlagList, criterion, input, target, bSize)
                continue
            candList, lossList, sincosFlagCandList = ProcessAllCandCurrLayer(model, criterion, paramNums, symNum, currLayerNo, node2OpSList, op2NodeList, bSize, candList, lossList, sincosFlagCandList, input, target)

        return candList, lossList

def ProcessPruneInTrainBS(model, node2OpSList, op2NodeList, paramNums, variNum, hiddenLayerNum, biSymNum, uniSymNum, epoch, EpochTh, nanFlag, input, target, divFlag, bSize):
    #print((epoch + 1) / EpochTh, "Before prun:")
    sincosFlagList = GetSincosFlagList(variNum + 1, hiddenLayerNum + 2, biSymNum, uniSymNum) #variNum + constNum,
    if 0 == nanFlag:
        model.SetIsCheckNan(0)
    symNum = biSymNum+uniSymNum
    candList, lossList = PruneGreedOpBeamSearch1(model, paramNums, symNum, node2OpSList, op2NodeList, sincosFlagList, input, target, 1, bSize)
    model.SetIsCheckNan(1)
    #print((epoch + 1) / EpochTh, "After prun:")

    params = list(model.parameters())

    minLoss = -1
    minSelLayerList = []
    minOutStr = ''
    for i in range(len(candList)):
        selLayerList = candList[i]

        print("[Cand No:", i,"]")
        print("PruneGreedNode symNet loss: ", lossList[i])
        #print("selLayerList:", selLayerList)

        finalSelList, indexList = GetIndexListFromSelLayerList(model, symNum, paramNums, selLayerList, op2NodeList)
        outStr = outPutSymRltAllWeight2(variNum, symNum, paramNums, finalSelList, indexList)
        outStrTrue = outPutSymRltAllTrueWeight2(variNum, symNum, paramNums, finalSelList, indexList, node2OpSList, params)
        outStrTrueSymPy = sympify(outStrTrue)
        outStrTrueSimple = simplify(outStrTrueSymPy)
        outStrTrueSimpleExpand = expand(outStrTrueSimple)
        # print("outStr:", outStr)
        # print("outStrTrue:", outStrTrue)
        # print("outStrTrueSimple:", outStrTrueSimple)
        print("outStrTrueSimpleExpand:", outStrTrueSimpleExpand)
        minConstList, bfgsError, codeStrBFGS = SolveByBFGS(input[0:20, 0:variNum].numpy(), target[0:20, :].squeeze(1).numpy(), outStrTrueSimpleExpand)
        minConstList, bfgsError, codeStrBFGS = ProcessTail(codeStrBFGS, minConstList, bfgsError, input[0:20, 0:variNum].numpy(), target[0:20, :].squeeze(1).numpy())
        print("BFGS error:", bfgsError)
        print("codeStrBFGS:", codeStrBFGS)
        #print("minConstList:", minConstList)
        if -1 == bfgsError:
            continue
        if bfgsError < 1.0e-6: #-8
            minLoss = bfgsError
            minSelLayerList = selLayerList
            minOutStr = codeStrBFGS #outStrTrueSimpleExpand
            break
        if -1 == minLoss or bfgsError < minLoss:
            minLoss = bfgsError
            minSelLayerList = selLayerList
            minOutStr = codeStrBFGS #outStrTrueSimpleExpand
        # if 0 < len(minConstList):
        #     test(str(outStrTrueSimpleExpand), variNum, input, target)
    # if -1 != minLoss:
    #     print("prune, haha!!!")
    #     FixWeight(model, minSelLayerList) # prune

    return minLoss, minSelLayerList, minOutStr, candList
############################################################################ BeamSearch op end

#init div
def ZeroModelPostParamDiv(model, symNum):
    with torch.no_grad():
        modelParamsList = list(model.parameters())
        layerNum = len(modelParamsList)
        for i in range(layerNum):
            if 0 == i:
                continue
            divNum = i
            currLayerParam = modelParamsList[i]
            variDim = currLayerParam.size()[0]
            #inputDim = currLayerParam.size()[1]
            for j in range(variDim):
                s = 3 #div No
                for k in range(divNum):
                    currLayerParam[j][s] = 0
                    s = s + symNum

def initModel(variNum, constNum, hiddenLayerNum, calList, opNumList, op2NodeList, maskOpIDList, input, divFlag):
    symNum = len(calList)
    model = NetNew(variNum, constNum, hiddenLayerNum, calList, opNumList)
    if 1 == divFlag:
        ZeroModelPostParamDiv(model, symNum)
    ZeroSinCosParam(model, symNum, divFlag)
    MaskOperator(model, symNum, op2NodeList, maskOpIDList)
    return model
    with torch.no_grad():
        model.SetIsCheckNan(1)
        while True:
            outputs, nanFlag, mltDivExpFlagList = model(input)
            if 0 == nanFlag:
                ##model.SetIsCheckNan(0)
                break
            else:
                model = NetNew(variNum, constNum, hiddenLayerNum, calList, opNumList)
                if 1 == divFlag:
                    ZeroModelPostParamDiv(model, symNum)
        ZeroSinCosParam(model, symNum, divFlag)
        MaskOperator(model, symNum, op2NodeList, maskOpIDList)
        return model

def ZeroModelPreParamDiv(model, mltDivExpFlagList, oldLoss, input, target, divFlag):
    with torch.no_grad():
        criterion = nn.MSELoss()
        modelParamsList = list(model.parameters())

        ##########################################################
        mltIndex = 4
        divIndex = 6
        if 1 == divFlag:
            expIndex = 10
        else:
            expIndex = 8

        sqreIndex = 12
        if 0 == divFlag:
            sqreIndex = 10

        divFlagList = mltDivExpFlagList[1]
        for i in range(len(divFlagList)):
            currLayerNo = divFlagList[i]
            currLayerParam = modelParamsList[currLayerNo]
            currLayerParam[divIndex, :] = 0  # 6
        #return

        mltFlagList = mltDivExpFlagList[0]
        for i in range(len(mltFlagList)):
            currLayerNo = mltFlagList[i]
            currLayerParam = modelParamsList[currLayerNo]
            currLayerParam[mltIndex, :] = 0 # 4

        expFlagList = mltDivExpFlagList[2]
        for i in range(len(expFlagList)):
            currLayerNo = expFlagList[i]
            currLayerParam = modelParamsList[currLayerNo]
            currLayerParam[expIndex, :] = 0 #10

        sqreFlagList = mltDivExpFlagList[3]
        for i in range(len(sqreFlagList)):
            currLayerNo = sqreFlagList[i]
            currLayerParam = modelParamsList[currLayerNo]
            currLayerParam[sqreIndex, :] = 0
        ##########################################################

def RecoverParam(model, modelBak):
    with torch.no_grad():
        modelParamsList = list(model.parameters())
        paramLayerNum = len(modelParamsList)
        modelParamsListBak = list(modelBak.parameters())
        for i in range(paramLayerNum):
            modelParamsList[i][:] = modelParamsListBak[i][:]

def ZeroSinCosParam(model, symNum, divFlag):
    with torch.no_grad():
        modelParamsList = list(model.parameters())
        paramLayerNum = len(modelParamsList)
        for i in range(1,paramLayerNum-1):
            currLayerParam = modelParamsList[i]
            if 1 == divFlag:
                startIndex = 8
                s = 4
            else:
                startIndex = 6
                s = 3
            for j in range(i):
                # currLayerParam[8][s:s + 4] = 0
                # currLayerParam[9][s:s + 4] = 0
                # currLayerParam[10][s:s + 4] = 0
                # currLayerParam[11][s:s + 4] = 0
                currLayerParam[startIndex][s:s + 4] = 0
                currLayerParam[startIndex+1][s:s + 4] = 0
                currLayerParam[startIndex+2][s:s + 4] = 0
                currLayerParam[startIndex+3][s:s + 4] = 0
                s = s + symNum

def myLog(x):
    tmp = np.log(abs(x))
    return tmp

def func(p, x, codeStr):
    """
    Define the form of fitting function.
    """
    constTable = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11", "c12", "c13", "c14", "c15", "c16", "c17", "c18", "c19", "c20"]
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20 = p

    if torch.is_tensor(x):
        variNum = x.size(1)
    else:
        variNum = x.shape[1]
    a = x[:, 0]
    if 2 <= variNum:
        b = x[:, 1]
    if 3 <= variNum:
        c = x[:, 2]
    if 4 <= variNum:
        d = x[:, 3]
    if 5 <= variNum:
        e = x[:, 4]
    if 6 <= variNum:
        f = x[:, 5]
    if 7 <= variNum:
        g = x[:, 6]
    if 8 <= variNum:
        h = x[:, 7]
    if 9 <= variNum:
        i = x[:, 8]

    tmp = codeStr
    for i in range(len(constTable)):
        tmp = codeStr.replace("const", constTable[i], 1)

    tmp = tmp.replace("sin", "np.sin", 20)
    tmp = tmp.replace("cos", "np.cos", 20)
    tmp = tmp.replace("exp", "np.exp", 20)
    tmp = tmp.replace("log", "myLog", 20) #np.log
    tmp = tmp.replace("sqrt", "np.sqrt", 20)
    outPut = eval(tmp)
    return outPut

def error(p, x, y, codeStr):
    """
    Fitting residuals.
    """
    return y - func(p, x, codeStr)

def objFunc(p, x, y, codeStr):
    """
    Fitting residuals.
    """
    #x, y, codeStr = args
    z = y - func(p, x, codeStr)
    totalError = sum(z ** 2)
    return totalError

def TransCoderStr2CodestrConst(codeStr, constTable):
    #constTable = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11", "c12", "c13", "c14", "c15", "c16", "c17", "c18", "c19", "c20"]
    constTableLen = len(constTable)
    constList = []
    express = sympy.sympify(codeStr)
    expressTmp = express
    count = 0
    constTmptList = []
    for a in sympy.preorder_traversal(express): #express
        if isinstance(a, sympy.Float):
            if count < constTableLen:
                expressTmp = expressTmp.subs(a, constTable[count])
            if not a in constTmptList:
                constTmptList.append(a)
                constList.append(float(a))
                count = count + 1
            #expressTmp = expressTmp.subs(a, 0.5)
    CodestrConst = str(expressTmp)
    for i in range(constTableLen - len(constList)):
        constList.append(0.0)

    return CodestrConst, constList

def ProcessTail(codeStrIn, constList, oldError, data_xIn, data_yIn):
    maxValue = max(abs(data_yIn))
    th1 = maxValue/1000000
    if 1 < th1:
        th1 = 1
    elif 0.01 > th1:
        th1 = 0.01

    th2 = maxValue/1000000
    th2 = th2*0.01
    if 0.1 < th2:
        th2 = 0.1
    if 0.001 > th2:
        th2 = 0.001

    th3 = maxValue / 1000000
    th3 = th3 * 0.01
    if 0.001 < th3:
        th3 = 0.001
    if 1.0e-6 > th3:
        th3 = 1.0e-6
#####################################################################
    if th1 <= oldError or -1 == oldError: #0.01
        return constList, oldError, codeStrIn
    oldErrorBak = oldError
    codeStrTmp = codeStrIn
    codeStrOut = codeStrIn
    while True:
        flag1 = 0
        flag2 = 0
        for i in range(len(constList)):
            intConst = round(constList[i])
            strConst = str(constList[i])
            if 0 == constList[i]:
                continue
            if th2 >= abs(constList[i]): #0.001
                codeStrTmp = codeStrTmp.replace(strConst, "0", 100)
                flag1 = 1
            elif th2 >= abs(constList[i]-intConst) and 0 != abs(constList[i]-intConst): #0.001
                codeStrTmp = codeStrTmp.replace(strConst, str(intConst), 100)
                flag2 = 1

        if 1 == flag1 or 1 == flag2:
            strSymPy = sympify(codeStrTmp)
            strSimple = simplify(strSymPy)
            strSimpleExpand = expand(strSimple)
            tmpBfgsConstList, tmpBfgsError, tmpBfgsCodeStr = SolveByBFGS(data_xIn, data_yIn, strSimpleExpand)
            if -1 != tmpBfgsError and (tmpBfgsError < oldError or 1.0e-6 > tmpBfgsError): #1.0e-6 或者 th3
                codeStrTmp = tmpBfgsCodeStr
                constList = tmpBfgsConstList
                oldError = tmpBfgsError
                codeStrOut = tmpBfgsCodeStr
                continue
        break

        if 0 == flag2:
            break

        output = func((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), data_xIn, codeStrTmp) #input[:, 0:2].numpy()
        num = len(output)
        tmpError = sum((output-data_yIn) ** 2) / num

        if tmpError < oldError or 1.0e-6 > tmpError:
            strSymPy = sympify(codeStrTmp)
            strSimple = simplify(strSymPy)
            codeStrOut = expand(strSimple) #codeStr
            oldError = tmpError
        break
    if 1.0e-6 <= oldErrorBak and 1.0e-6 > oldError:
        print("Save a true expression!!!")

    codeStrOutTmp = sympy.sympify(codeStrOut)
    complexityTmp = GetComplexity(codeStrOutTmp)
    print("ProcessTail expression complexityTmp:", complexityTmp)
    if 30 < complexityTmp:
        print("bfgsError:", oldError, "but 30 < ProcessTail expression complexityTmp:", complexityTmp, "!!!!!!!!!!!")
        oldError = -1

    return constList, oldError, codeStrOut

def SolveByBFGS(data_xIn, data_yIn, codeStrIn):
    with torch.no_grad():
        constTable = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11", "c12", "c13", "c14", "c15", "c16", "c17", "c18", "c19", "c20"]
        data_x = data_xIn.astype(np.float64)
        data_y = data_yIn.astype(np.float64)
        minError = -1
        minConstList = []
        num1 = str(codeStrIn).count('pi')
        num2 = str(codeStrIn).count('I')
        num3 = str(codeStrIn).count('nan')
        num4 = str(codeStrIn).count('zoo')
        if 0 < num1 + num2 + num3 + num4:
            print("pi I nan zoo can't SolveByBFGS !!!")
            return minConstList, minError, str(codeStrIn)

        codeStr, constList = TransCoderStr2CodestrConst(codeStrIn, constTable)
        if 20 < len(constList):
            print("too many const can't SolveByBFGS !!!")
            return minConstList, minError, str(codeStrIn)
        print("codeStr abstract:", codeStr)

        codeStrInTmp = sympy.sympify(codeStrIn)
        complexityTmp = GetComplexity(codeStrInTmp)
        print("SolveByBFGS expression complexityTmp:", complexityTmp)
        if 40 < complexityTmp:
            print("40 < SolveByBFGS expression complexityTmp:", complexityTmp, "!!!!!!!!!!!")
            return minConstList, minError, str(codeStrIn)
        pointNum = data_x.shape[0]
        for i in range(1):
            p0 = np.array(constList)
            fit_res = minimize(objFunc, p0, args=(data_x, data_y, codeStr), method='BFGS', options={'gtol': 1e-6, 'disp': False})

            rlt = dict(fit_res)
            totalError1 = sum(error(rlt["x"], data_x, data_y, codeStr)**2) / pointNum
            totalError = sum(error(rlt["x"], data_xIn, data_yIn, codeStr) ** 2) / pointNum
            if totalError1 < totalError:
                totalError = totalError1

            #totalError = sum(abs(error(rlt["x"], data_x, data_y, codeStr)))/20
            #print("totalError:", totalError)
            if np.isnan(totalError): #if nan == totalError: #if torch.isnan(totalError):
                continue
            if totalError < minError or -1 == minError:
                minError = totalError
                minConstList = rlt["x"]
            if minError < 1.0e-6 and -1 != minError: #-8
                break
        if 0 < len(minConstList):
            for ii in range(len(constTable)):
                i = len(constTable) - ii - 1 #
                # if 0 == minConstList[i]:
                #     continue
                constName = constTable[i]
                w = minConstList[i]
                if 0 > w:
                    strConst = "(" + str(w) + ")"
                else:
                    strConst = str(w) #
                codeStr = codeStr.replace(constName, strConst, 100)

        return minConstList, minError, codeStr

def GetCandListOverlap(oldCandList, newCandList):
    num = len(newCandList)
    count = 0
    for i in range(num):
        if newCandList[i] in oldCandList:
            count = count + 1
    overlapRatio = count/num
    return overlapRatio
def SetLR(optimizer, lr):
    with torch.no_grad():
        for p in optimizer.param_groups:
            p['lr'] = lr