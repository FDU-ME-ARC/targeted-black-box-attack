import tensorflow as tf

import Layers
import Data
import Nets

HParamCIFAR100 = {'BatchSize': 100, 
                  'LearningRate': 1e-3, 
                  'MinLearningRate': 1e-5, 
                  'DecayAfter': 600,
                  'ValidateAfter': 600,
                  'TestSteps': 100,
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
            self._bodyClass20, self._bodyClass100 = self.body(self._images)
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
        net = Nets.SimpleV1(standardized, self._step, self._ifTest, self._layers)
        #net = Nets.Xcpetion(standardized, self._step, self._ifTest, self._layers, numMiddle=self._numMiddle)
        
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
        
        return class20.output, class100.output 
        
    def inference(self, logits):
        return tf.argmax(logits, axis=-1, name='inference')
    
    def lossMultilet(self, logits, name='multilet'): 
        net = Layers.TruncatedMultiletLoss(logits, numSame=1, numDiff=1, weightSame=1.0, weightDiff=3.0, name=name)
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
                                                  decay_rate=0.95) + self._HParam['MinLearningRate']
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
                
                loss20, loss100, loss, accu20, accu100, step, _ = \
                    self._sess.run([self._lossClass20, self._lossClass100, self._loss, \
                                    self._accuracyClass20, self._accuracyClass100, self._step, self._optimizer], \
                                   feed_dict={self._images: data, \
                                              self._labelsClass20: label20, \
                                              self._labelsClass100: label100})
                self._sess.run(self._updateOps)
                print('\rStep: ', step, '; L20: %.3f'% loss20, '; L100: %.3f'% loss100, \
                      '; L: %.3f'% loss, \
                      '; A20: %.3f'% accu20, '; A100: %.3f'% accu100, \
                      end='')
                
                if step % self._HParam['ValidateAfter'] == 0: 
                    self.evaluate(genTest)
                    if pathSave is not None:
                        self.save(pathSave)
                    self._sess.run([self._phaseTrain])
            
    def evaluate(self, genTest, path=None):
        if path is not None:
            self.load(path)
        
        totalLoss20       = 0.0
        totalLoss100      = 0.0
        totalLoss         = 0.0
        totalAccu20       = 0.0
        totalAccu100      = 0.0
        self._sess.run([self._phaseTest])  
        for _ in range(self._HParam['TestSteps']): 
            data, label20, label100 = next(genTest)
            loss20, loss100, loss, accu20, accu100 = \
                self._sess.run([self._lossClass20, self._lossClass100, self._loss, \
                                self._accuracyClass20, self._accuracyClass100], \
                               feed_dict={self._images: data, \
                                          self._labelsClass20: label20, \
                                          self._labelsClass100: label100})
            totalLoss20       += loss20
            totalLoss100      += loss100
            totalLoss         += loss
            totalAccu20       += accu20
            totalAccu100      += accu100
        totalLoss20       /= self._HParam['TestSteps']
        totalLoss100      /= self._HParam['TestSteps']
        totalLoss         /= self._HParam['TestSteps']
        totalAccu20       /= self._HParam['TestSteps']
        totalAccu100      /= self._HParam['TestSteps']
        print('\nTest: Loss20: ', totalLoss20, '; Loss100: ', totalLoss100, \
              '; Loss: ', totalLoss, \
              '; Accu20: ', totalAccu20, '; Accu100: ', totalAccu100, )
        
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
    net = NetCIFAR100([32, 32, 3], 2) 
    batchTrain, batchTest = Data.generators(BatchSize=HParamCIFAR100['BatchSize'], preprocSize=[32, 32, 3], numSame=0, numDiff=0)
    net.train(batchTrain, batchTest, pathSave='./Classify/netcifar100.ckpt')
# The best configuration is 64 features and 8 middle layers

