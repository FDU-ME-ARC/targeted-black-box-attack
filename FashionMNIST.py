import random
import h5py
import numpy as np

import tensorflow as tf

import Preproc
import Layers
import Nets

def loadHDF5():
    with h5py.File('FashionMNIST.h5', 'r') as f:
        dataTrain   = np.expand_dims(np.array(f['Train']['images'])[:, :, :, 0], axis=-1)
        labelsTrain = np.array(f['Train']['labels']).reshape([-1])
        dataTest    = np.expand_dims(np.array(f['Test']['images'])[:, :, :, 0], axis=-1)
        labelsTest  = np.array(f['Test']['labels']).reshape([-1])
        
    return (dataTrain, labelsTrain, dataTest, labelsTest)

def preproc(images, size): 
    results = np.ndarray([images.shape[0]]+size, np.uint8)
    for idx in range(images.shape[0]): 
        distorted     = Preproc.centerCrop(images[idx], size)
        results[idx]  = distorted
    
    return results

def generator(BatchSize, preprocSize=[28, 28, 1]): 
    dataTrain, labelsTrain, dataTest, labelsTest = loadHDF5()
    data = np.concatenate([dataTrain, dataTest], axis=0)
    labels = np.concatenate([labelsTrain, labelsTest], axis=0)
    
    invertedIdx = [[] for _ in range(10)]
    
    for idx in range(len(data)):
        invertedIdx[labels[idx]].append(idx)
    
    def genCIFAR10():
        now = 0
        batchData   = []
        batchLabels = []
        for _ in range(BatchSize):
            classAnchor = labels[now]
            classPos    = classAnchor
            idxAnchor   = now
            idxPos      = random.randint(0, len(invertedIdx[classPos])-1)
            while idxPos == now:
                idxPos  = random.randint(0, len(invertedIdx[classPos])-1)
            idxPos      = invertedIdx[idxPos]
            classNeg    = random.randint(0, 9)
            while classNeg == classPos:
                classNeg = random.randint(0, 9)
            idxNeg      = random.randint(0, len(invertedIdx[classNeg])-1)
            idxNeg      = invertedIdx[idxNeg]
            batchData.extend([data[idxAnchor], data[idxPos], data[idxNeg]])
            batchLabels.extend([classAnchor, classPos, classNeg])
            now += 1
            if now >= 70000: 
                now = 0
        batchData = preproc(np.array(batchData), preprocSize)
        batchLabels = np.array(batchLabels)
        assert batchData.shape[0] == BatchSize*3, "CIFAR10: size is wrong"
        assert len(batchData.shape) == 4, "CIFAR10: size is wrong"
        yield batchData, batchLabels
    
    return genCIFAR10()


def allData(preprocSize=[28, 28, 1]): 
    dataTrain, labelsTrain, dataTest, labelsTest = loadHDF5()
    data = np.concatenate([dataTrain, dataTest], axis=0)
    labels = np.concatenate([labelsTrain, labelsTest], axis=0)
    
    invertedIdx = [[] for _ in range(10)]
    
    for idx in range(len(data)):
        invertedIdx[labels[idx]].append(idx)
    
    return preproc(data, preprocSize), labels, invertedIdx


def generators(BatchSize, preprocSize=[28, 28, 1], numSame=1, numDiff=1):
    ''' generators for multi-let
    Args:
        numSame: number of samples in the same coarse class; 
        numDiff: number of sample in different coarse class. 
    Return:
        genTrain: an iterator for the training set
        genTest:  an iterator for the test set'''
    (dataTrain, labelsTrain,  dataTest, labelsTest) = loadHDF5()
        
    def genTrainDatum():
        index = Preproc.genIndex(dataTrain.shape[0], shuffle=True)
        while True:
            indexAnchor = next(index)
            imageAnchor = dataTrain[indexAnchor]
            labelAnchor = labelsTrain[indexAnchor]
            images      = [imageAnchor]
            labels      = [labelAnchor]
            
            yield images, labels
        
    def genTestDatum():
        index = Preproc.genIndex(dataTest.shape[0], shuffle=False)
        while True:
            indexAnchor = next(index)
            imageAnchor = dataTest[indexAnchor]
            labelAnchor = labelsTest[indexAnchor]
            images      = [imageAnchor]
            labels      = [labelAnchor]
            
            yield images, labels
    
    def preprocTrain(images, size): 
        results = np.ndarray([images.shape[0]]+size, np.uint8)
        for idx in range(images.shape[0]): 
            distorted     = Preproc.randomFlipH(images[idx])
            distorted     = Preproc.randomShift(distorted, rng=4)
            #distorted     = Preproc.randomRotate(distorted, rng=30)
            # distorted     = Preproc.randomRotate(images[idx], rng=30)
            # distorted     = Preproc.randomCrop(distorted, size)
            #distorted     = Preproc.randomContrast(distorted, 0.5, 1.5)
            #distorted     = Preproc.randomBrightness(distorted, 32)
            results[idx]  = distorted.reshape([28, 28, 1])
        
        return results
    
    def preprocTest(images, size): 
        results = np.ndarray([images.shape[0]]+size, np.uint8)
        for idx in range(images.shape[0]): 
            distorted = images[idx]
            distorted     = Preproc.centerCrop(distorted, size)
            results[idx]  = distorted
        
        return results
    
    def genTrainBatch(BatchSize):
        datum = genTrainDatum()
        while True:
            batchImages = []
            batchLabels = []
            for _ in range(BatchSize):
                images, labels = next(datum)
                batchImages.append(images)
                batchLabels.append(labels)
            batchImages = preprocTrain(np.concatenate(batchImages, axis=0), preprocSize)
            batchLabels = np.concatenate(batchLabels, axis=0)
            
            yield batchImages, batchLabels
            
    def genTestBatch(BatchSize):
        datum = genTestDatum()
        while True:
            batchImages = []
            batchLabels = []
            for _ in range(BatchSize):
                images, labels = next(datum)
                batchImages.append(images)
                batchLabels.append(labels)
            batchImages = preprocTest(np.concatenate(batchImages, axis=0), preprocSize)
            batchLabels = np.concatenate(batchLabels, axis=0)
            
            yield batchImages, batchLabels
        
    return genTrainBatch(BatchSize), genTestBatch(BatchSize)

def generatorsAdv(BatchSize, preprocSize=[28, 28, 1]):
    ''' generators for multi-let
    Args:
    Return:
        genTrain: an iterator for the training set
        genTest:  an iterator for the test set'''
    (dataTrain, labelsTrain,  dataTest, labelsTest) = loadHDF5()
        
    def genTrainDatum():
        index = Preproc.genIndex(dataTrain.shape[0], shuffle=True)
        while True:
            indexAnchor = next(index)
            imageAnchor = dataTrain[indexAnchor]
            labelAnchor = labelsTrain[indexAnchor]
            images      = [imageAnchor]
            labels      = [labelAnchor]
            
            yield images, labels
        
    def genTestDatum():
        index = Preproc.genIndex(dataTest.shape[0], shuffle=False)
        while True:
            indexAnchor = next(index)
            imageAnchor = dataTest[indexAnchor]
            labelAnchor = labelsTest[indexAnchor]
            images      = [imageAnchor]
            labels      = [labelAnchor]
            
            yield images, labels
    
    def preprocTrain(images, size): 
        results = np.ndarray([images.shape[0]]+size, np.uint8)
        for idx in range(images.shape[0]): 
            distorted     = Preproc.randomFlipH(images[idx])
            distorted     = Preproc.randomShift(distorted, rng=4)
            #distorted     = Preproc.randomRotate(distorted, rng=30)
            # distorted     = Preproc.randomRotate(images[idx], rng=30)
            #distorted     = Preproc.randomCrop(distorted, size)
            #distorted     = Preproc.randomContrast(distorted, 0.5, 1.5)
            #distorted     = Preproc.randomBrightness(distorted, 32)
            results[idx]  = distorted.reshape([28, 28, 1])
        
        return results
    
    def preprocTest(images, size): 
        results = np.ndarray([images.shape[0]]+size, np.uint8)
        for idx in range(images.shape[0]): 
            distorted = images[idx]
            distorted     = Preproc.centerCrop(distorted, size)
            results[idx]  = distorted
        
        return results
    
    def genTrainBatch(BatchSize):
        datum = genTrainDatum()
        while True:
            batchImages = []
            batchLabels = []
            batchTargets = []
            for _ in range(BatchSize):
                images, labels = next(datum)
                batchImages.append(images)
                batchLabels.append(labels)
                batchTargets.append(random.randint(0, 9))
            batchImages = preprocTrain(np.concatenate(batchImages, axis=0), preprocSize)
            batchLabels = np.concatenate(batchLabels, axis=0)
            batchTargets = np.array(batchTargets)
            
            yield batchImages, batchLabels, batchTargets
            
    def genTestBatch(BatchSize):
        datum = genTestDatum()
        while True:
            batchImages = []
            batchLabels = []
            batchTargets = []
            for _ in range(BatchSize):
                images, labels = next(datum)
                batchImages.append(images)
                batchLabels.append(labels)
                batchTargets.append(random.randint(0, 9))
            batchImages = preprocTest(np.concatenate(batchImages, axis=0), preprocSize)
            batchLabels = np.concatenate(batchLabels, axis=0)
            batchTargets = np.array(batchTargets)
            
            yield batchImages, batchLabels, batchTargets
        
    return genTrainBatch(BatchSize), genTestBatch(BatchSize)

HParamMNIST = {'BatchSize': 200, 
                  'LearningRate': 1e-3, 
                  'MinLearningRate': 1e-5, 
                  'DecayAfter': 300,
                  'ValidateAfter': 300,
                  'TestSteps': 50,
                  'TotalSteps': 40000}

class NetMNIST(Nets.Net):
    
    def __init__(self, shapeImages, numMiddle=2, HParam=HParamMNIST):
        Nets.Net.__init__(self)
        
        self._init = False
        self._numMiddle    = numMiddle
        self._HParam       = HParam
        self._graph        = tf.Graph()
        self._sess         = tf.Session(graph=self._graph)
        
        with self._graph.as_default(): 
            self._ifTest        = tf.Variable(False, name='ifTest', trainable=False, dtype=tf.bool)
            self._step          = tf.Variable(0, name='step', trainable=False, dtype=tf.int32)
            self._phaseTrain    = tf.assign(self._ifTest, False)
            self._phaseTest     = tf.assign(self._ifTest, True)
            
            # Inputs
            self._images = tf.placeholder(dtype=tf.float32, shape=[None]+shapeImages, \
                                                  name='CIFAR10_images')
            self._labels = tf.placeholder(dtype=tf.int64, shape=[None], \
                                                  name='CIFAR10_labels_class')
            
            # Net
            self._body      = self.body(self._images)
            self._inference = self.inference(self._body)
            self._accuracy  = tf.reduce_mean(tf.cast(tf.equal(self._inference, self._labels), tf.float32))
            self._loss      = self.lossClassify(self._body, self._labels)
            self._loss      = 0
            self._updateOps = []
            for elem in self._layers: 
                if len(elem.losses) > 0: 
                    for tmp in elem.losses: 
                        self._loss += tmp
            for elem in self._layers: 
                if len(elem.updateOps) > 0: 
                    for tmp in elem.updateOps: 
                        self._updateOps.append(tmp)
            print(self.summary)
            print("\n Begin Training: \n")
                    
            # Saver
            self._saver = tf.train.Saver(max_to_keep=5)
        
    def preproc(self, images):
        # Preprocessings
        casted        = tf.cast(images, tf.float32)
        standardized  = tf.identity(casted / 127.5 - 1.0, name='training_standardized')
            
        return standardized
        
    def body(self, images):
        # Preprocessings
        standardized = self.preproc(images)
        # Body
        net = Nets.SimpleV1C(standardized, self._step, self._ifTest, self._layers)
        
        class10 = Layers.FullyConnected(net.output, outputSize=10, weightInit=Layers.XavierInit, wd=1e-4, \
                                    biasInit=Layers.ConstInit(0.0), \
                                    activation=Layers.Linear, \
                                    name='FC_Coarse', dtype=tf.float32)
        self._layers.append(class10)
        
        return class10.output
        
    def inference(self, logits):
        return tf.argmax(logits, axis=-1, name='inference')
    
    def lossClassify(self, logits, labels, name='cross_entropy'):
        net = Layers.CrossEntropy(logits, labels, name=name)
        self._layers.append(net)
        return net.output
    
    def train(self, genTrain, genTest, pathLoad=None, pathSave=None):
        with self._graph.as_default(): 
            self._lr = tf.train.exponential_decay(self._HParam['LearningRate'], \
                                                  global_step=self._step, \
                                                  decay_steps=self._HParam['DecayAfter']*5, \
                                                  decay_rate=0.20) + self._HParam['MinLearningRate']
            self._optimizer = tf.train.AdamOptimizer(self._lr, epsilon=1e-8).minimize(self._loss, global_step=self._step)
            # Initialize all
            self._sess.run(tf.global_variables_initializer())
            
            if pathLoad is not None:
                self.load(pathLoad)
                
            self.evaluate(genTest)
#             self.sample(genTest)
            
            self._sess.run([self._phaseTrain])
            if pathSave is not None:
                self.save(pathSave)
            for _ in range(self._HParam['TotalSteps']): 
                
                data, label = next(genTrain)
                
                loss, accu, step, _ = \
                    self._sess.run([self._loss, \
                                    self._accuracy, self._step, self._optimizer], \
                                   feed_dict={self._images: data, \
                                              self._labels: label})
                self._sess.run(self._updateOps)
                print('\rStep: ', step, \
                      '; L: %.3f'% loss, \
                      '; A: %.3f'% accu, \
                      end='')
                
                if step % self._HParam['ValidateAfter'] == 0: 
                    self.evaluate(genTest)
                    if pathSave is not None:
                        self.save(pathSave)
                    self._sess.run([self._phaseTrain])
            
    def evaluate(self, genTest, path=None):
        if path is not None:
            self.load(path)
        
        totalLoss  = 0.0
        totalAccu  = 0.0
        self._sess.run([self._phaseTest])  
        for _ in range(self._HParam['TestSteps']): 
            data, label = next(genTest)
            loss, accu = \
                self._sess.run([self._loss, \
                                self._accuracy], \
                               feed_dict={self._images: data, \
                                          self._labels: label})
            totalLoss += loss
            totalAccu += accu
        totalLoss /= self._HParam['TestSteps']
        totalAccu /= self._HParam['TestSteps']
        print('\nTest: Loss: ', totalLoss, \
              '; Accu: ', totalAccu)
        
    def infer(self, images):
        
        self._sess.run([self._phaseTest]) 
        
        return self._sess.run(self._inference, feed_dict={self._images: images})
        
    def save(self, path):
        self._saver.save(self._sess, path, global_step=self._step)
    
    def load(self, path):
        self._saver.restore(self._sess, path)
            
if __name__ == '__main__':
    net = NetMNIST([28, 28, 1], 2) # 8 
    batchTrain, batchTest = generators(BatchSize=HParamMNIST['BatchSize'], preprocSize=[28, 28, 1], numSame=0, numDiff=0)
    net.train(batchTrain, batchTest, pathSave='./ClassifyFashionMNIST/netcifar10.ckpt')
# The best configuration is 64 features and 8 middle layers



