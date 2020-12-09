import random
import numpy as np
import matplotlib.pyplot as plt

import CIFAR100
import Preproc


def generators(BatchSize, preprocSize=[32, 32, 3], numSame=1, numDiff=1):
    ''' generators for multi-let
    Args:
        numSame: number of samples in the same coarse class; 
        numDiff: number of sample in different coarse class. 
    Return:
        genTrain: an iterator for the training set
        genTest:  an iterator for the testCIFAR10 set'''
    (dataTrain, labelsCoarseTrain, labelsFineTrain, dataTest, labelsCoarseTest, labelsFineTest) = \
        CIFAR100.loadHDF5()
    indicesCoarseTrain, indicesFineTrain = \
        Preproc.indicesInverted(dataTrain, labelsCoarseTrain, labelsFineTrain)
    indicesCoarseTest, indicesFineTest = \
        Preproc.indicesInverted(dataTest, labelsCoarseTest, labelsFineTest)
        
    def genTrainDatum():
        index = Preproc.genIndex(dataTrain.shape[0], shuffle=True)
        while True:
            indexAnchor       = next(index)
            imageAnchor       = dataTrain[indexAnchor]
            labelCoarseAnchor = labelsCoarseTrain[indexAnchor]
            labelFineAnchor   = labelsFineTrain[indexAnchor]
            indexPos       = random.randint(0, len(indicesFineTrain[labelFineAnchor])-1)
            while indicesFineTrain[labelFineAnchor][indexPos] == indexAnchor:
                indexPos   = random.randint(0, len(indicesFineTrain[labelFineAnchor])-1)
            imagePos       = dataTrain[indicesFineTrain[labelFineAnchor][indexPos]]
            labelCoarsePos = labelsCoarseTrain[indicesFineTrain[labelFineAnchor][indexPos]]
            labelFinePos   = labelsFineTrain[indicesFineTrain[labelFineAnchor][indexPos]]
            imagesSame       = []
            labelsCoarseSame = []
            labelsFineSame   = []
            for _ in range(numSame): 
                indexSame = random.randint(0, len(indicesCoarseTrain[labelCoarseAnchor])-1)
                while labelFineAnchor == labelsFineTrain[indicesCoarseTrain[labelCoarseAnchor][indexSame]]:
                    indexSame = random.randint(0, len(indicesCoarseTrain[labelCoarseAnchor])-1)
                imagesSame.append(dataTrain[indicesCoarseTrain[labelCoarseAnchor][indexSame]])
                labelsCoarseSame.append(labelsCoarseTrain[indicesCoarseTrain[labelCoarseAnchor][indexSame]])
                labelsFineSame.append(labelsFineTrain[indicesCoarseTrain[labelCoarseAnchor][indexSame]])
            imagesDiff = []
            labelsCoarseDiff = []
            labelsFineDiff   = []
            for _ in range(numDiff): 
                classDiff = random.randint(0, len(indicesFineTrain)-1)
                while labelCoarseAnchor == labelsCoarseTrain[indicesFineTrain[classDiff][0]]: 
                    classDiff = random.randint(0, len(indicesFineTrain)-1)
                indexDiff = random.randint(0, len(indicesFineTrain[classDiff])-1)
                imagesDiff.append(dataTrain[indicesFineTrain[classDiff][indexDiff]])
                labelsCoarseDiff.append(labelsCoarseTrain[indicesFineTrain[classDiff][indexDiff]])
                labelsFineDiff.append(labelsFineTrain[indicesFineTrain[classDiff][indexDiff]])
            images       = [imageAnchor]       + [imagePos]       + imagesSame       + imagesDiff
            labelsCoarse = [labelCoarseAnchor] + [labelCoarsePos] + labelsCoarseSame + labelsCoarseDiff
            labelsFine   = [labelFineAnchor]   + [labelFinePos]   + labelsFineSame   + labelsFineDiff
            
            yield images, labelsCoarse, labelsFine
        
    def genTestDatum():
        index = Preproc.genIndex(dataTest.shape[0], shuffle=False)
        while True:
            indexAnchor       = next(index)
            imageAnchor       = dataTest[indexAnchor]
            labelCoarseAnchor = labelsCoarseTest[indexAnchor]
            labelFineAnchor   = labelsFineTest[indexAnchor]
            indexPos       = random.randint(0, len(indicesFineTest[labelFineAnchor])-1)
            while indicesFineTest[labelFineAnchor][indexPos] == indexAnchor:
                indexPos   = random.randint(0, len(indicesFineTest[labelFineAnchor])-1)
            imagePos       = dataTest[indicesFineTest[labelFineAnchor][indexPos]]
            labelCoarsePos = labelsCoarseTest[indicesFineTest[labelFineAnchor][indexPos]]
            labelFinePos   = labelsFineTest[indicesFineTest[labelFineAnchor][indexPos]]
            imagesSame       = []
            labelsCoarseSame = []
            labelsFineSame   = []
            for _ in range(numSame): 
                indexSame = random.randint(0, len(indicesCoarseTest[labelCoarseAnchor])-1)
                while labelFineAnchor == labelsFineTest[indicesCoarseTest[labelCoarseAnchor][indexSame]]:
                    indexSame = random.randint(0, len(indicesCoarseTest[labelCoarseAnchor])-1)
                imagesSame.append(dataTest[indicesCoarseTest[labelCoarseAnchor][indexSame]])
                labelsCoarseSame.append(labelsCoarseTest[indicesCoarseTest[labelCoarseAnchor][indexSame]])
                labelsFineSame.append(labelsFineTest[indicesCoarseTest[labelCoarseAnchor][indexSame]])
            imagesDiff = []
            labelsCoarseDiff = []
            labelsFineDiff   = []
            for _ in range(numDiff): 
                classDiff = random.randint(0, len(indicesFineTest)-1)
                while labelCoarseAnchor == labelsCoarseTest[indicesFineTest[classDiff][0]]: 
                    classDiff = random.randint(0, len(indicesFineTest)-1)
                indexDiff = random.randint(0, len(indicesFineTest[classDiff])-1)
                imagesDiff.append(dataTest[indicesFineTest[classDiff][indexDiff]])
                labelsCoarseDiff.append(labelsCoarseTest[indicesFineTest[classDiff][indexDiff]])
                labelsFineDiff.append(labelsFineTest[indicesFineTest[classDiff][indexDiff]])
            images       = [imageAnchor]       + [imagePos]       + imagesSame       + imagesDiff
            labelsCoarse = [labelCoarseAnchor] + [labelCoarsePos] + labelsCoarseSame + labelsCoarseDiff
            labelsFine   = [labelFineAnchor]   + [labelFinePos]   + labelsFineSame   + labelsFineDiff
            
            yield images, labelsCoarse, labelsFine
    
    def preprocTrain(images, size): 
        results = np.ndarray([images.shape[0]]+size, np.uint8)
        for idx in range(images.shape[0]): 
            distorted     = Preproc.randomFlipH(images[idx])
            distorted     = Preproc.randomShift(distorted, rng=4)
            #distorted     = Preproc.randomRotate(distorted, rng=30)
            #distorted     = Preproc.randomCrop(distorted, size)
            #distorted     = Preproc.randomContrast(distorted, 0.5, 1.5)
            #distorted     = Preproc.randomBrightness(distorted, 32)
            results[idx]  = distorted
        
        return results
    
    def preprocTest(images, size): 
        results = np.ndarray([images.shape[0]]+size, np.uint8)
        for idx in range(images.shape[0]): 
            distorted     = Preproc.centerCrop(images[idx], size)
            results[idx]  = distorted
        
        return results
    
    def genTrainBatch(BatchSize):
        datum = genTrainDatum()
        while True:
            batchImages       = []
            batchLabelsCoarse = []
            batchLabelsFine   = []
            for _ in range(BatchSize):
                images, labelsCoarse, labelsFine = next(datum)
                batchImages.append(images)
                batchLabelsCoarse.append(labelsCoarse)
                batchLabelsFine.append(labelsFine)
            batchImages       = preprocTrain(np.concatenate(batchImages, axis=0), preprocSize)
            batchLabelsCoarse = np.concatenate(batchLabelsCoarse, axis=0)
            batchLabelsFine   = np.concatenate(batchLabelsFine, axis=0)
            
            yield batchImages, batchLabelsCoarse, batchLabelsFine
            
    def genTestBatch(BatchSize):
        datum = genTestDatum()
        while True:
            batchImages       = []
            batchLabelsCoarse = []
            batchLabelsFine   = []
            for _ in range(BatchSize):
                images, labelsCoarse, labelsFine = next(datum)
                batchImages.append(images)
                batchLabelsCoarse.append(labelsCoarse)
                batchLabelsFine.append(labelsFine)
            batchImages       = preprocTest(np.concatenate(batchImages, axis=0), preprocSize)
            batchLabelsCoarse = np.concatenate(batchLabelsCoarse, axis=0)
            batchLabelsFine   = np.concatenate(batchLabelsFine, axis=0)
            yield batchImages, batchLabelsCoarse, batchLabelsFine
        
    return genTrainBatch(BatchSize), genTestBatch(BatchSize)

def _testShowData(): 
    batchTrain, batchTest = generators(BatchSize=50, numDiff=2)
    batchData, batchLabelsCoarse, batchLabelsFine = next(batchTrain)
    print(batchData.shape)
    print(batchLabelsCoarse.shape)
    print(batchLabelsFine.shape)
    plt.subplot(6, 6, 1)
    plt.imshow(batchData[0])
    print('Index: ', 0, ' Type: ', CIFAR100.labelsCoarse[batchLabelsCoarse[0]], \
              ', ', CIFAR100.labelsFine[batchLabelsFine[0]])
    plt.subplot(6, 6, 2)
    plt.imshow(batchData[1])
    print('Index: ', 1, ' Type: ', CIFAR100.labelsCoarse[batchLabelsCoarse[1]], \
              ', ', CIFAR100.labelsFine[batchLabelsFine[1]])
    plt.subplot(6, 6, 3)
    plt.imshow(batchData[2])
    print('Index: ', 2, ' Type: ', CIFAR100.labelsCoarse[batchLabelsCoarse[2]], \
              ', ', CIFAR100.labelsFine[batchLabelsFine[2]])
    plt.subplot(6, 6, 4)
    plt.imshow(batchData[3])
    print('Index: ', 3, ' Type: ', CIFAR100.labelsCoarse[batchLabelsCoarse[3]], \
              ', ', CIFAR100.labelsFine[batchLabelsFine[3]])
    plt.subplot(6, 6, 5)
    plt.imshow(batchData[4])
    print('Index: ', 4, ' Type: ', CIFAR100.labelsCoarse[batchLabelsCoarse[4]], \
              ', ', CIFAR100.labelsFine[batchLabelsFine[4]])
    plt.subplot(6, 6, 6)
    plt.imshow(batchData[5])
    print('Index: ', 5, ' Type: ', CIFAR100.labelsCoarse[batchLabelsCoarse[5]], \
              ', ', CIFAR100.labelsFine[batchLabelsFine[5]])
    batchData, batchLabelsCoarse, batchLabelsFine = next(batchTrain)
    plt.subplot(6, 6, 7)
    plt.imshow(batchData[0])
    print('Index: ', 6, ' Type: ', CIFAR100.labelsCoarse[batchLabelsCoarse[7]], \
              ', ', CIFAR100.labelsFine[batchLabelsFine[7]])
    plt.subplot(6, 6, 8)
    plt.imshow(batchData[8])
    print('Index: ', 7, ' Type: ', CIFAR100.labelsCoarse[batchLabelsCoarse[8]], \
              ', ', CIFAR100.labelsFine[batchLabelsFine[8]])
    plt.subplot(6, 6, 9)
    plt.imshow(batchData[9])
    print('Index: ', 8, ' Type: ', CIFAR100.labelsCoarse[batchLabelsCoarse[9]], \
              ', ', CIFAR100.labelsFine[batchLabelsFine[9]])
    plt.subplot(6, 6, 10)
    plt.imshow(batchData[10])
    print('Index: ', 9, ' Type: ', CIFAR100.labelsCoarse[batchLabelsCoarse[10]], \
              ', ', CIFAR100.labelsFine[batchLabelsFine[10]])
    plt.subplot(6, 6, 11)
    plt.imshow(batchData[11])
    print('Index: ', 10, ' Type: ', CIFAR100.labelsCoarse[batchLabelsCoarse[11]], \
              ', ', CIFAR100.labelsFine[batchLabelsFine[11]])
    plt.subplot(6, 6, 12)
    plt.imshow(batchData[12])
    print('Index: ', 11, ' Type: ', CIFAR100.labelsCoarse[batchLabelsCoarse[12]], \
              ', ', CIFAR100.labelsFine[batchLabelsFine[12]])
    plt.subplot(6, 6, 13)
    plt.imshow(batchData[13])
    print('Index: ', 12, ' Type: ', CIFAR100.labelsCoarse[batchLabelsCoarse[13]], \
              ', ', CIFAR100.labelsFine[batchLabelsFine[13]])
    plt.subplot(6, 6, 14)
    plt.imshow(batchData[14])
    print('Index: ', 13, ' Type: ', CIFAR100.labelsCoarse[batchLabelsCoarse[14]], \
              ', ', CIFAR100.labelsFine[batchLabelsFine[14]])
    plt.subplot(6, 6, 15)
    plt.imshow(batchData[0])
    print('Index: ', 14, ' Type: ', CIFAR100.labelsCoarse[batchLabelsCoarse[0]], \
              ', ', CIFAR100.labelsFine[batchLabelsFine[0]])
    for idx in range(15, 36):
        batchData, batchLabelsCoarse, batchLabelsFine = next(batchTest)
        plt.subplot(6, 6, idx+1)
        plt.imshow(batchData[idx])
        print('Index: ', idx, ' Type: ', CIFAR100.labelsCoarse[batchLabelsCoarse[idx]], \
                  ', ', CIFAR100.labelsFine[batchLabelsFine[idx]])
    
    
    plt.show()


if __name__ == '__main__': 
    # _testShowData()
    batchTrain, batchTest = generators(BatchSize=50, preprocSize=[32, 32, 3], numSame=0, numDiff=0)
    batchData, batchLabelsCoarse, batchLabelsFine = next(batchTrain)
    for idx in range(36): 
        plt.subplot(6, 6, idx+1)
        plt.imshow(batchData[idx])
    plt.show() 
    
    
