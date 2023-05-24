# -*- coding: utf-8 -*-

import torch.utils.data as data
from PruneSymNet_kernel import * #MyDataset, initModel, ZeroModelPreParamDiv, ZeroSinCosParam, MaskOperator, PruneReverse, ProcessPruneInTrainBS, GetCandListOverlap, SetLR



def GetTestData(pointNum, variNum, codeStr):
    data_x = torch.rand(pointNum, variNum+1) * 10.0 - 5
    data_x[:, variNum] = 1
    # for i in range(pointNum):
    #     data_x.data[i][variNum] = 1
    data_y = func((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), data_x, codeStr)

    data_y = data_y.unsqueeze(1)
    # testData = torch.cat([data_x, data_y], 1)
    # testData = testData.unsqueeze(0)

    return data_x, data_y

def GetTrainParam(divFlag, maskOpIDList):
    squareFlag = 1
    if 1 == divFlag:
        calList = ['+', '-', '*', '/', 'sin', 'cos', 'exp', 'log']
        opNumList = [2, 2, 2, 2, 1, 1, 1, 1]
    else:
        calList = ['+', '-', '*', 'sin', 'cos', 'exp', 'log']
        opNumList = [2, 2, 2, 1, 1, 1, 1]
        for i in range(len(maskOpIDList)):
            maskOpIDList[i] = maskOpIDList[i] - 1
    if 1 == squareFlag:
        calList.append('square')
        opNumList.append(1)

    paramNums = torch.tensor(opNumList, dtype=torch.long)

    with torch.no_grad():
        symNum = len(opNumList)
        biSymNum = 0
        uniSymNum = 0
        for i in range(symNum):
            if 2 == opNumList[i]:
                biSymNum = biSymNum + 1
            else:
                uniSymNum = uniSymNum + 1

        node2OpSList = []
        op2NodeList = []
        symNum = len(opNumList)
        s = 0
        for i in range(symNum):
            node2OpSList.append(s)
            s = s + opNumList[i]
            op2NodeList.append(i)
            if 2 == opNumList[i]:
                op2NodeList.append(i)
    return symNum, paramNums, calList, opNumList, node2OpSList, op2NodeList, biSymNum, uniSymNum

def testOneExpression(objectExpressName, input, target, variNum, candNum):
    constNum = 1
    hiddenLayerNum = 6

    divFlag = 1
    maskOpIDList = []  #[6, 7]
    reInitTh = 600
    print("divFlag, maskOpIDList:", divFlag, maskOpIDList)
    symNum, paramNums, calList, opNumList, node2OpSList, op2NodeList, biSymNum, uniSymNum = GetTrainParam(divFlag, maskOpIDList)
############################################################################################################################################
    batch_size = 128
    trainloader = torch.utils.data.DataLoader(MyDataset(input, target), batch_size, shuffle=True, num_workers=2)
    model = initModel(variNum, constNum, hiddenLayerNum, calList, opNumList, op2NodeList, maskOpIDList, input, divFlag)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    with torch.autograd.set_detect_anomaly(True):
        EpochTh = 20
        oldLoss = -1
        minPruneLoss = -1
        minSelLayerList = []
        minOutStr = ''
        minIndex = -1
        oldCandList = []
        repeatNum = 0
        intervalNum = 0
        currLR = 0.01
        nanFlagNum = 0
        pruneReverseEpochNum = 0
        for epoch in range(3000):  # loop over the dataset multiple times
            pruneReverseEpochNum = pruneReverseEpochNum + 1
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data
                flag = 0
                while True:
                    outputs, nanFlag, mltDivExpFlagList = model(inputs)
                    if 0 == nanFlag and 20000 <= epoch:
                        model.SetIsCheckNan(0)
                    loss1 = criterion(outputs, labels)
                    regularization = L12Smooth()
                    modelParamsList = list(model.parameters())
                    reg_loss = regularization(modelParamsList)
                    loss = loss1 + 0.005*reg_loss # + 0.01 * regu_loss

                    if -1 != oldLoss and 10 < loss1/oldLoss and 0 == flag:
                        ZeroModelPreParamDiv(model, mltDivExpFlagList, oldLoss, inputs, labels, divFlag)
                        flag = 1
                        print("loss fly!!! loss, oldLoss:", loss, oldLoss)
                        continue
                    else:
                        break
                oldLoss = loss1

                optimizer.zero_grad()  # zero the gradient buffers
                loss.backward()
                #torch.nn.utils.clip_grad_value_(model.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()  # Does the update
                #scheduler.step(loss)
                ZeroSinCosParam(model, symNum, divFlag)
                MaskOperator(model, symNum, op2NodeList, maskOpIDList)
                print('[%d, %5d] loss1: %.3f loss2: %.3f lr: %.6f, nanFlag:%d' % (epoch + 1, i + 1, loss1.item(), loss.item(), optimizer.param_groups[0]['lr'], nanFlag))

            #scheduler.step()
            with torch.no_grad():
                if 1 == nanFlag:
                    nanFlagNum = nanFlagNum + 1
                else:
                    nanFlagNum = 0
                if (epoch + 1) % EpochTh == 0:
                    pruneReverseLoss = -1
                    if 100 < pruneReverseEpochNum:
                        pruneReverseLoss, reverseOutStr = PruneReverse(model, variNum, symNum, paramNums, node2OpSList, criterion, input, target)
                        print("pruneReverseLoss:", pruneReverseLoss)
                        print("reverseOutStr:", reverseOutStr)
                    pruneLoss, selLayerList, outStr, newCandList = ProcessPruneInTrainBS(model, node2OpSList, op2NodeList, paramNums, variNum, hiddenLayerNum, biSymNum, uniSymNum, epoch, EpochTh, nanFlag, input, target, divFlag, candNum)
                    if -1 != pruneReverseLoss and pruneReverseLoss < pruneLoss:
                        pruneLoss = pruneReverseLoss
                        outStr = reverseOutStr
                        selLayerList = []
                    overlapRatio = GetCandListOverlap(oldCandList, newCandList)
                    print("overlapRatio:", overlapRatio)
                    intervalNum = intervalNum + 1
                    if 0.8 <= overlapRatio:
                        repeatNum = repeatNum + 1
                        if 2 < repeatNum and 2 < intervalNum:
                            currLR = currLR + 0.01
                            if 0.08 < currLR:
                                print("currLR too large:", currLR)
                                currLR = 0.08
                            SetLR(optimizer, currLR)
                            intervalNum = 0
                            print("SetLR(optimizer, currLR and overlapRatio:)", currLR, overlapRatio, "   !!!")
                    else:
                        repeatNum = 0
                        intervalNum = 0
                        if 0.01 != currLR:
                            currLR = 0.01
                            SetLR(optimizer, currLR)
                            print("SetLR(optimizer, currLR:) default", currLR, "   !!!")
                    oldCandList = newCandList

                    if -1 != pruneLoss and (-1 == minPruneLoss or pruneLoss < minPruneLoss):
                        minPruneLoss = pruneLoss
                        minSelLayerList = selLayerList
                        minOutStr = outStr
                        minIndex = epoch+1
                    if pruneLoss < 1.0e-6 and -1 != pruneLoss:  #-8
                        print("Find the optimal solution !!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        break
                    oldLoss = -1
                    print("objectExpressName:", objectExpressName)
                    print("minIndex:", minIndex)
                    print("minPruneLoss:", minPruneLoss)
                    print("minOutStr:", minOutStr)
                if (epoch + 1) % reInitTh == 0 or 20 <= nanFlagNum:
                    #############################################
                    model = initModel(variNum, constNum, hiddenLayerNum, calList, opNumList, op2NodeList, maskOpIDList, input, divFlag)  # 重新初始化模型，跳出局部最优
                    optimizer = optim.Adam(model.parameters(), lr=0.01)
                    oldLoss = -1
                    pruneReverseEpochNum = 0
                    print("ReInitModel !!!")
        print("objectExpressName:", objectExpressName)
        print("minIndex:", minIndex)
        # print("minPruneLoss:", minPruneLoss)
        # print("minOutStr:", minOutStr)
        return minPruneLoss, minOutStr

def GetExpressVariNum(expressStr):
    tmp = expressStr.replace("sin", "mm", 100)
    tmp = tmp.replace("cos", "mm", 100)
    tmp = tmp.replace("exp", "mm", 100)
    tmp = tmp.replace("log", "mm", 100)
    tmp = tmp.replace("sqrt", "mm", 100)
    variList = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    variNum = 0
    for variStr in variList:
        variNum = variNum + min(1, tmp.count(variStr))
    return variNum

####################################################################################################################################
if __name__ == '__main__':
    expressList = ['a + a**2 + a**3', 'a + a**2 + a**3 + a**4', 'a + a**2 + a**3 + a**4 + a**5', 'a + a**2 + a**3 + a**4 + a**5 + a**6',
                   'a + a**2 + a**3 + a**4 + a**5 + a**6 + a**7', 'a + a**2 + a**3 + a**4 + a**5 + a**6 + a**7 + a**8',
                   'a**5 - 2*a**3 + a', 'a**6 - 2*a**4 + a**2', '3.39*a**3 + 2.12*a**2 + 1.78*a', '0.48*a**4 + 3.39*a**3 + 2.12*a**2 + 1.78*a',
                   'a/b', '(a**2)/(b**2)', '213.80940889*(1-exp(-0.54723748542*a))', '3*a-2*a*b-a*a', '2*b-a*b-b*b', 'exp(-0.5*a**2)', '(a*b)/(4*3.14159*c)',
                   'a*b*c/2']
    beamSize = 5
    for i in range(len(expressList)):
        strExpress = expressList[i]
        pointNum = 128
        variNum = GetExpressVariNum(strExpress)
        input, target = GetTestData(pointNum, variNum, strExpress)
        print(expressList[i], "   start!!!")
        minPruneLoss, minOutStr = testOneExpression(expressList[i], input, target, variNum, beamSize)
        print("minPruneLoss:", minPruneLoss)
        print("minOutStr:", minOutStr)
        print('\n')
    print('Finish All Data Test!')

