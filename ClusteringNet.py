import tensorflow as tf

import Layers
import Data
import Nets

HParamCIFAR100 = {'BatchSize': 60, 
                  'LearningRate': 1e-3, 
                  'MinLearningRate': 1e-5, 
                  'DecayAfter': 1000,
                  'ValidateAfter': 1000,
                  'TestSteps': 170,
                  'TotalSteps': 60000}

class NetCIFAR100(Nets.Net):
    
    def __init__(self, shapeImages, numMiddle=2, HParam=HParamCIFAR100):
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
            self._images         = tf.placeholder(dtype=tf.float32, shape=[None]+shapeImages, \
                                                  name='CIFAR100_images')
            self._labelsClass20  = tf.placeholder(dtype=tf.int64, shape=[None], \
                                                  name='CIFAR100_labels_class20')
            self._labelsClass100 = tf.placeholder(dtype=tf.int64, shape=[None], \
                                                  name='CIFAR100_labels_class100')
            
            # Net
            self._bodyClass20, self._bodyClass100, self._embedding, self._trashClass20, self._trashClass100 = \
                self.body(self._images)
            self._inferenceClass20  = self.inference(self._bodyClass20)
            self._inferenceClass100 = self.inference(self._bodyClass100)
            self._accuracyClass20   = tf.reduce_mean(tf.cast(tf.equal(self._inferenceClass20, self._labelsClass20), tf.float32))
            self._accuracyClass100  = tf.reduce_mean(tf.cast(tf.equal(self._inferenceClass100, self._labelsClass100), tf.float32))
            self._lossClass20       = self.lossClassify(self._bodyClass20, self._labelsClass20)
            self._lossClass100      = self.lossClassify(self._bodyClass100, self._labelsClass100)
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
            self._lossTrashClass20  = self.lossClassify(self._trashClass20, self._labelsClass20)
            self._lossTrashClass100 = self.lossClassify(self._trashClass100, self._labelsClass100)
            self._inferenceTrashClass20  = self.inference(self._trashClass20)
            self._inferenceTrashClass100 = self.inference(self._trashClass100)
            self._accuracyTrashClass20   = tf.reduce_mean(tf.cast(tf.equal(self._inferenceTrashClass20, self._labelsClass20), tf.float32))
            self._accuracyTrashClass100  = tf.reduce_mean(tf.cast(tf.equal(self._inferenceTrashClass100, self._labelsClass100), tf.float32))
            self._lossMultilet = self.lossMultilet(self._embedding) + self._lossTrashClass20 + self._lossTrashClass100
            self._accuTriplet = self.accuTriplet(self._embedding)
            self._accuMultilet = self.accuMultilet(self._embedding)
            self._weightMultiletLoss = tf.Variable(0.1, trainable=False)
            self._changeWeightMultiletLoss = tf.assign(self._weightMultiletLoss, self._weightMultiletLoss*1.2)
            self._loss = self._weightMultiletLoss * self._lossMultilet + self._loss
            print(self.summary)
            print("\n Net Initialized: \n")
                    
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
#         net = Nets.SimpleV1(standardized, self._step, self._ifTest, self._layers)
#         net = Nets.SimpleV2(standardized, self._step, self._ifTest, self._layers)
#         net = Nets.SimpleV3(standardized, self._step, self._ifTest, self._layers)
        net = Nets.SimpleV7(standardized, self._step, self._ifTest, self._layers)
#         net = Nets.SimpleV11(standardized, self._step, self._ifTest, self._layers)
#         net = Nets.SimpleV7X(standardized, self._step, self._ifTest, self._layers, 2)
#         net = Nets.Xcpetion(standardized, self._step, self._ifTest, self._layers, numMiddle=self._numMiddle)
#         net = Nets.XcpetionM(standardized, self._step, self._ifTest, self._layers, numMiddle=self._numMiddle)
#         net = Nets.XcpetionM3X(standardized, self._step, self._ifTest, self._layers, numMiddle=self._numMiddle)
#         net = Nets.ResNet(standardized, self._step, self._ifTest, self._layers)
#         net = Nets.VGG(standardized, self._step, self._ifTest, self._layers)
#         net = Nets.VGGM(standardized, self._step, self._ifTest, self._layers)
        
        class20 = Layers.FullyConnected(net.output, outputSize=20, weightInit=Layers.XavierInit, wd=1e-4, \
                                    biasInit=Layers.ConstInit(0.0), \
                                    activation=Layers.Linear, \
                                    name='FC_Coarse', dtype=tf.float32)
        self._layers.append(class20)
        class100 = Layers.FullyConnected(net.output, outputSize=100, weightInit=Layers.XavierInit, wd=1e-4, \
                                    biasInit=Layers.ConstInit(0.0), \
                                    activation=Layers.Linear, \
                                    name='FC_Fine', dtype=tf.float32)
        self._layers.append(class100)
        embedding = Layers.FullyConnected(net.output, outputSize=64, weightInit=Layers.XavierInit, wd=1e-4, \
                                        biasInit=Layers.ConstInit(0.0), \
                                        activation=Layers.Tanh, \
                                        name='FC_embedding', dtype=tf.float32)
        self._layers.append(embedding)
        multiletClass20 = Layers.FullyConnected(embedding.output, outputSize=20, weightInit=Layers.XavierInit, wd=1e-4, \
                                    biasInit=Layers.ConstInit(0.0), \
                                    activation=Layers.Linear, \
                                    name='Multilet_Coarse', dtype=tf.float32)
        self._layers.append(multiletClass20)
        multiletClass100 = Layers.FullyConnected(embedding.output, outputSize=100, weightInit=Layers.XavierInit, wd=1e-4, \
                                    biasInit=Layers.ConstInit(0.0), \
                                    activation=Layers.Linear, \
                                    name='Multilet_Fine', dtype=tf.float32)
        self._layers.append(multiletClass100)
        embedding = tf.nn.l2_normalize(embedding.output, 1)
        
        return class20.output, class100.output, embedding, multiletClass20.output, multiletClass100.output
        
    def inference(self, logits):
        return tf.argmax(logits, axis=-1, name='inference')
    
    def lossMultilet(self, logits, name='multilet'): 
        net = Layers.MultiletLossFinal(logits, numSame=1, numDiff=1, weightSame=1.0, weightDiff=3.0, name=name)
        self._layers.append(net)
        return net.output
    
    def lossClassify(self, logits, labels, name='cross_entropy'):
        net = Layers.CrossEntropy(logits, labels, name=name)
        self._layers.append(net)
        return net.output
    
    def accuTriplet(self, logits, name='triplet_accu'): 
        net = Layers.TripletAccu(logits, numSame=1, numDiff=1, name=name)
        self._layers.append(net)
        return net.output
    
    def accuMultilet(self, logits, name='multilet_accu'): 
        net = Layers.MultiletAccu(logits, numSame=1, numDiff=1, name=name)
        self._layers.append(net)
        return net.output
    
    def train(self, genTrain, genTest, pathLoad=None, pathSave=None):
        with self._graph.as_default(): 
            self._lr = tf.train.exponential_decay(self._HParam['LearningRate'], \
                                                  global_step=self._step, \
                                                  decay_steps=self._HParam['DecayAfter'], \
                                                  decay_rate=0.90) + self._HParam['MinLearningRate']
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
                
                data, label20, label100 = next(genTrain)
                
                loss20, loss100, lossMultilet, loss, accu20, accu100, accutri, accumulti, \
                    lossTrash20, lossTrash100, accuTrash20, accuTrash100, step, _ = \
                    self._sess.run([self._lossClass20, self._lossClass100, self._lossMultilet, self._loss, \
                                    self._accuracyClass20, self._accuracyClass100, self._accuTriplet, self._accuMultilet, \
                                    self._lossTrashClass20, self._lossTrashClass100, \
                                    self._accuracyTrashClass20, self._accuracyTrashClass100, \
                                    self._step, self._optimizer], \
                                   feed_dict={self._images: data, \
                                              self._labelsClass20: label20, \
                                              self._labelsClass100: label100})
                self._sess.run(self._updateOps)
                print('\rStep: ', step, '; L20: %.3f'% loss20, '; L100: %.3f'% loss100, '; LMul:%.3f'% lossMultilet, \
                      '; L: %.3f'% loss, \
                      '; A20: %.3f'% accu20, '; A100: %.3f'% accu100, \
                      '; ATri: %.3f'% accutri, '; AMul: %.3f'% accumulti, \
                      '; lT20: %.3f'% lossTrash20, '; lT100: %.3f'% lossTrash100, \
                      '; AT20: %.3f'% accuTrash20, '; AT100: %.3f'% accuTrash100, \
                      end='')
                
                if step % self._HParam['ValidateAfter'] == 0: 
                    self.evaluate(genTest)
                    if pathSave is not None:
                        self.save(pathSave)
                    self._sess.run([self._phaseTrain])
                    weight = self._sess.run([self._weightMultiletLoss])[0]
                    if weight < 1.0: 
                        self._sess.run([self._changeWeightMultiletLoss])
#                     if step % 6000 == 0: 
#                         self.sample(genTest)
            
    def evaluate(self, genTest, path=None):
        if path is not None:
            self.load(path)
        
        totalLoss20       = 0.0
        totalLoss100      = 0.0
        totalLossMultilet = 0.0
        totalLoss         = 0.0
        totalAccu20       = 0.0
        totalAccu100      = 0.0
        totalTriplet      = 0.0
        totalMultilet     = 0.0
        totalLossTrash20       = 0.0
        totalLossTrash100      = 0.0
        totalAccuTrash20       = 0.0
        totalAccuTrash100      = 0.0
        self._sess.run([self._phaseTest])  
        for _ in range(self._HParam['TestSteps']): 
            data, label20, label100 = next(genTest)
            loss20, loss100, lossMultilet, loss, accu20, accu100, accutri, accumulti, \
                lossTrash20, lossTrash100, accuTrash20, accuTrash100 = \
                self._sess.run([self._lossClass20, self._lossClass100, self._lossMultilet, self._loss, \
                                self._accuracyClass20, self._accuracyClass100, \
                                self._accuTriplet, self._accuMultilet, \
                                self._lossTrashClass20, self._lossTrashClass100, 
                                self._accuracyTrashClass20, self._accuracyTrashClass100], \
                               feed_dict={self._images: data, \
                                          self._labelsClass20: label20, \
                                          self._labelsClass100: label100})
            totalLoss20       += loss20
            totalLoss100      += loss100
            totalLossMultilet += lossMultilet
            totalLoss         += loss
            totalAccu20       += accu20
            totalAccu100      += accu100
            totalTriplet      += accutri
            totalMultilet     += accumulti
            totalLossTrash20       += lossTrash20
            totalLossTrash100      += lossTrash100
            totalAccuTrash20      += accuTrash20
            totalAccuTrash100     += accuTrash100
        totalLoss20       /= self._HParam['TestSteps']
        totalLoss100      /= self._HParam['TestSteps']
        totalLossMultilet /= self._HParam['TestSteps']
        totalLoss         /= self._HParam['TestSteps']
        totalAccu20       /= self._HParam['TestSteps']
        totalAccu100      /= self._HParam['TestSteps']
        totalTriplet      /= self._HParam['TestSteps']
        totalMultilet     /= self._HParam['TestSteps']
        totalLossTrash20       /= self._HParam['TestSteps']
        totalLossTrash100      /= self._HParam['TestSteps']
        totalAccuTrash20      /= self._HParam['TestSteps']
        totalAccuTrash100     /= self._HParam['TestSteps']
        print('\nTest: Loss20: ', totalLoss20, '; Loss100: ', totalLoss100, '; LossMultilet', totalLossMultilet, \
              '; Loss: ', totalLoss, \
              '; Accu20: ', totalAccu20, '; Accu100: ', totalAccu100, \
              '; AccuTriple: ', totalTriplet, '; AccuMultilet: ', totalMultilet, \
              '; LossTrash20: ', totalLossTrash20, '; LossTrash100: ', totalLossTrash100, \
              '; AccuTrash20: ', totalAccuTrash20, '; AccuTrash100: ', totalAccuTrash100)
        
    def infer(self, data): 
        self._sess.run([self._phaseTest])  
        feature = \
            self._sess.run(self._embedding, \
                           feed_dict={self._images: data})
        
        return feature
        
    def save(self, path):
        self._saver.save(self._sess, path, global_step=self._step)
    
    def load(self, path):
        self._saver.restore(self._sess, path)
    
    def sample(self, genTest):
        import numpy as np
        data = []
        label20 = []
        label100 = []
        for _ in range(100): 
            tmpdata, tmplabel20, tmplabel100 = next(genTest)
            tmpdata = self._sess.run(self._embedding, feed_dict={self._images: tmpdata, \
                                                               self._labelsClass20: tmplabel20, \
                                                               self._labelsClass100: tmplabel100})
            data.append(tmpdata)
            label20.append(tmplabel20)
            label100.append(tmplabel100)
        data = np.concatenate(data, axis=0)
        label20 = np.concatenate(label20, axis=0)
        label100 = np.concatenate(label100, axis=0)
        dist100 = np.zeros([100, 100])
        count100 = np.zeros([100, 100])
        dist20 = np.zeros([20, 20])
        count20 = np.zeros([20, 20])
        idx = 0
        print('Samples: ', data.shape)
        while idx < data.shape[0]: 
            dist20[label20[idx], label20[idx+1]] += np.linalg.norm(data[idx]-data[idx+1])
            count20[label20[idx], label20[idx+1]] += 1
            dist100[label100[idx], label100[idx+1]] += np.linalg.norm(data[idx]-data[idx+1])
            count100[label100[idx], label100[idx+1]] += 1
            dist20[label20[idx], label20[idx+2]] += np.linalg.norm(data[idx]-data[idx+2])
            count20[label20[idx], label20[idx+2]] += 1
            dist100[label100[idx], label100[idx+2]] += np.linalg.norm(data[idx]-data[idx+2])
            count100[label100[idx], label100[idx+2]] += 1
            dist20[label20[idx], label20[idx+3]] += np.linalg.norm(data[idx]-data[idx+3])
            count20[label20[idx], label20[idx+3]] += 1
            dist100[label100[idx], label100[idx+3]] += np.linalg.norm(data[idx]-data[idx+3])
            count100[label100[idx], label100[idx+3]] += 1
            idx += 4
        for idx in range(100): 
            for jdx in range(100):
                if count100[idx, jdx] > 0: 
                    dist100[idx, jdx] /= count100[idx, jdx]
        for idx in range(20): 
            for jdx in range(20):
                if count20[idx, jdx] > 0: 
                    dist20[idx, jdx] /= count20[idx, jdx]
        print('Coarse:')
        for idx in range(20):
            print('Class ', idx, ': ', dist20[idx])
        print('Fine:')
        for idx in range(100):
            print('Class ', idx, ': ', dist100[idx])
            
if __name__ == '__main__':
    net = NetCIFAR100([28, 28, 3], 2)
    batchTrain, batchTest = Data.generators(BatchSize=HParamCIFAR100['BatchSize'], preprocSize=[28, 28, 3], numDiff=1)
#     net.load('./ModelsSimpleV7/netcifar100.ckpt-39600')
#     net.sample(batchTest)
    net.train(batchTrain, batchTest, pathSave='./ModelsSimpleV11/netcifar100.ckpt')

# The best configuration is 64 features and 8 middle layers


