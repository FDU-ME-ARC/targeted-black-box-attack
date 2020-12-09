import tensorflow as tf
import Layers

from Protocols import *

wd = 4e-5

def LogisticRegression(standardized, step, ifTest, layers): 
    net = Layers.Flatten(standardized)
    layers.append(net)
    
    return net

def VanillaNN(standardized, step, ifTest, layers): 
    net = Layers.Flatten(standardized)
    layers.append(net)
    net = Layers.FullyConnected(net.output, outputSize=256, weightInit=Layers.XavierInit, wd=1e-4, \
                                biasInit=Layers.ConstInit(0.0), \
                                activation=Layers.Tanh, \
                                name='FC1', dtype=tf.float32)
    layers.append(net)
    
    return net

def MNIST1(standardized, step, ifTest, layers): 
    net = Layers.Conv2D(standardized, convChannels=16, \
                        convKernel=[5, 5], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv1', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=32, \
                        convKernel=[5, 5], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv2', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=64, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv7b', dtype=tf.float32)
    layers.append(net)
    flattened = tf.reshape(net.output, [-1, net.output.shape[1]*net.output.shape[2]*net.output.shape[3]])
    net = Layers.FullyConnected(flattened, outputSize=256, weightInit=Layers.XavierInit, wd=wd, \
                                biasInit=Layers.ConstInit(0.0), \
                                activation=Layers.ReLU, \
                                name='FC1', dtype=tf.float32)
    layers.append(net)
    net = Layers.BatchNorm(net.output, step, ifTest, epsilon=1e-5, name='BatchNormFC1', dtype=tf.float32)
    layers.append(net)
    
    return net

def SimpleV1(standardized, step, ifTest, layers): 
    net = Layers.Conv2D(standardized, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv1', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='Conv2', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv3', dtype=tf.float32)
    layers.append(net)
    toadd = net.output
    net = Layers.Conv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv4', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv5', dtype=tf.float32)
    layers.append(net)
    added = toadd + net.output
    net = Layers.Conv2D(added, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='Conv6', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='Conv7', dtype=tf.float32)
    layers.append(net)
    flattened = tf.reshape(net.output, [-1, net.output.shape[1]*net.output.shape[2]*net.output.shape[3]])
    net = Layers.FullyConnected(flattened, outputSize=1024, weightInit=Layers.XavierInit, wd=wd, \
                                biasInit=Layers.ConstInit(0.0), \
                                activation=Layers.ReLU, \
                                name='FC1', dtype=tf.float32)
    layers.append(net)
    net = Layers.BatchNorm(net.output, step, ifTest, epsilon=1e-5, name='BatchNormFC1', dtype=tf.float32)
    layers.append(net)
    net = Layers.FullyConnected(net.output, outputSize=1024, weightInit=Layers.XavierInit, wd=wd, \
                                biasInit=Layers.ConstInit(0.0), \
                                activation=Layers.ReLU, \
                                name='FC2', dtype=tf.float32)
    layers.append(net)
    net = Layers.BatchNorm(net.output, step, ifTest, epsilon=1e-5, name='BatchNormFC2', dtype=tf.float32)
    layers.append(net)
    
    return net

def SimpleV1C(standardized, step, ifTest, layers): 
    net = Layers.Conv2D(standardized, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv1', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv2', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv3', dtype=tf.float32)
    layers.append(net)
    toadd = net.output
    net = Layers.Conv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv4a', dtype=tf.float32)
    layers.append(net)
    added = toadd + net.output
    net = Layers.Conv2D(added, convChannels=128, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv5', dtype=tf.float32)
    layers.append(net)
    toadd = net.output
    net = Layers.Conv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv6a', dtype=tf.float32)
    layers.append(net)
    added = toadd + net.output
    net = Layers.Conv2D(added, convChannels=128, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv7a', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv7b', dtype=tf.float32)
    layers.append(net)
    flattened = tf.reshape(net.output, [-1, net.output.shape[1]*net.output.shape[2]*net.output.shape[3]])
    net = Layers.FullyConnected(flattened, outputSize=1024, weightInit=Layers.XavierInit, wd=wd, \
                                biasInit=Layers.ConstInit(0.0), \
                                activation=Layers.ReLU, \
                                name='FC1', dtype=tf.float32)
    layers.append(net)
    net = Layers.BatchNorm(net.output, step, ifTest, epsilon=1e-5, name='BatchNormFC1', dtype=tf.float32)
    layers.append(net)
    
    return net

def SimpleV1CC(standardized, step, ifTest, layers): 
    net = Layers.Conv2D(standardized, convChannels=16, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv1', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=32, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv2', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=32, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv3', dtype=tf.float32)
    layers.append(net)
    toadd = net.output
    net = Layers.Conv2D(net.output, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv4a', dtype=tf.float32)
    layers.append(net)
    added = toadd + net.output
    net = Layers.Conv2D(added, convChannels=64, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv5', dtype=tf.float32)
    layers.append(net)
    toadd = net.output
    net = Layers.Conv2D(net.output, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv6a', dtype=tf.float32)
    layers.append(net)
    added = toadd + net.output
    net = Layers.Conv2D(added, convChannels=64, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv7a', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=32, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv7b', dtype=tf.float32)
    layers.append(net)
    flattened = tf.reshape(net.output, [-1, net.output.shape[1]*net.output.shape[2]*net.output.shape[3]])
    net = Layers.FullyConnected(flattened, outputSize=256, weightInit=Layers.XavierInit, wd=wd, \
                                biasInit=Layers.ConstInit(0.0), \
                                activation=Layers.ReLU, \
                                name='FC1', dtype=tf.float32)
    layers.append(net)
    net = Layers.BatchNorm(net.output, step, ifTest, epsilon=1e-5, name='BatchNormFC1', dtype=tf.float32)
    layers.append(net)
    
    return net

def SimpleV2(standardized, step, ifTest, layers): 
    
    net = Layers.Conv2D(standardized, convChannels=32, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv32_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=32, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv32_2', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=64, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv64_1', dtype=tf.float32)
    layers.append(net)
    toadd = net.output
    net = Layers.Conv2D(net.output, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv64_2', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv64_3', dtype=tf.float32)
    layers.append(net)
    
    added = toadd + net.output
    toadd = added
    
    net = Layers.Conv2D(toadd, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv64_4', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv64_5', dtype=tf.float32)
    layers.append(net)
    
    added = toadd + net.output
    toadd = added
    
    net = Layers.Conv2D(toadd, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv64_6', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv64_7', dtype=tf.float32)
    layers.append(net)
    
    added = toadd + net.output
    
    net = Layers.Conv2D(added, convChannels=128, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv128_1', dtype=tf.float32)
    layers.append(net)
    toadd = net.output
    
    net = Layers.Conv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv128_2', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv128_3', dtype=tf.float32)
    layers.append(net)
    
    added = toadd + net.output
    toadd = added
    
    net = Layers.Conv2D(toadd, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv128_4', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv128_5', dtype=tf.float32)
    layers.append(net)
    
    added = toadd + net.output
    toadd = added
    
    net = Layers.Conv2D(toadd, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv128_6', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv128_7', dtype=tf.float32)
    layers.append(net)
    
    added = toadd + net.output
    
    net = Layers.Conv2D(added, convChannels=256, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv256_1', dtype=tf.float32)
    layers.append(net)
    
    toadd = net.output
    
    net = Layers.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv256_2', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv256_3', dtype=tf.float32)
    layers.append(net)
    
    added = toadd + net.output
    toadd = added
    
    net = Layers.Conv2D(added, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv256_4', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv256_5', dtype=tf.float32)
    layers.append(net)
    
    added = toadd + net.output
    toadd = added
    
    net = Layers.Conv2D(added, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv256_6', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv256_7', dtype=tf.float32)
    layers.append(net)
    
    added = toadd + net.output
    
    net = Layers.Flatten(added)
    
    net = Layers.FullyConnected(net.output, outputSize=1024, weightInit=Layers.XavierInit, wd=wd, \
                                biasInit=Layers.ConstInit(0.0), \
                                activation=Layers.ReLU, \
                                name='FC1', dtype=tf.float32)
    layers.append(net)
    net = Layers.BatchNorm(net.output, step, ifTest, epsilon=1e-5, name='BatchNormFC1', dtype=tf.float32)
    layers.append(net)
    net = Layers.FullyConnected(net.output, outputSize=1024, weightInit=Layers.XavierInit, wd=wd, \
                                biasInit=Layers.ConstInit(0.0), \
                                activation=Layers.ReLU, \
                                name='FC2', dtype=tf.float32)
    layers.append(net)
    net = Layers.BatchNorm(net.output, step, ifTest, epsilon=1e-5, name='BatchNormFC2', dtype=tf.float32)
    layers.append(net)

    return net

def SimpleV3(standardized, step, ifTest, layers):
    net = Layers.DepthwiseConv2D(standardized, convChannels=3*16, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='DepthwiseConv3x16', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv96', dtype=tf.float32)
    layers.append(net)
    
    toadd = Layers.Conv2D(net.output, convChannels=192, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='SepConv192Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.SepConv2D(net.output, convChannels=192, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv192a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=192, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        name='SepConv192b', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=384, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='SepConv384Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU384')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv384a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv384b', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=768, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='SepConv768Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU768')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=768, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv768a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=768, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv768b', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU11024')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=1024, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='SepConv1024', dtype=tf.float32)
    layers.append(net)
#     net = Layers.SepConv2D(net.output, convChannels=1536, \
#                            convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
#                            convInit=Layers.XavierInit, convPadding='SAME', \
#                            biasInit=Layers.ConstInit(0.0), \
#                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
#                            activation=Layers.ReLU, \
#                            name='SepConv1536', dtype=tf.float32)
#     layers.append(net)
    net = Layers.GlobalAvgPool(net.output, name='GlobalAvgPool')
    layers.append(net)
    return net

def SimpleV4(standardized, step, ifTest, layers):
    '''A bad version'''
    net = Layers.DepthwiseConv2D(standardized, convChannels=3*16, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='DepthwiseConv3x16', dtype=tf.float32)
    layers.append(net)
    
    group = tf.split(net.output, 3, axis=3, name='Group')
    
    branch1 = Layers.SepConv2D(group[0], convChannels=32, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv96Branch1', dtype=tf.float32)
    layers.append(branch1)
    branch2 = Layers.SepConv2D(group[1], convChannels=32, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv96Branch2', dtype=tf.float32)
    layers.append(branch2)
    branch3 = Layers.SepConv2D(group[2], convChannels=32, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv96Branch3', dtype=tf.float32)
    layers.append(branch3)
    
    concated = tf.concat([branch1.output, branch2.output, branch3.output], axis=3, name='Concat')
    
    toadd = Layers.Conv2D(concated, convChannels=192, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='SepConv192Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.SepConv2D(concated, convChannels=192, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv192a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=192, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        name='SepConv192b', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=384, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='SepConv384Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU384')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv384a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv384b', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=768, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='SepConv768Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU768')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=768, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv768a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=768, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv768b', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU11024')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=1024, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='SepConv1024', dtype=tf.float32)
    layers.append(net)
#     net = Layers.SepConv2D(net.output, convChannels=1536, \
#                            convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
#                            convInit=Layers.XavierInit, convPadding='SAME', \
#                            biasInit=Layers.ConstInit(0.0), \
#                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
#                            activation=Layers.ReLU, \
#                            name='SepConv1536', dtype=tf.float32)
#     layers.append(net)
    net = Layers.GlobalAvgPool(net.output, name='GlobalAvgPool')
    layers.append(net)
    return net

def SimpleV5(standardized, step, ifTest, layers):
    net = Layers.DepthwiseConv2D(standardized, convChannels=3*16, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='DepthwiseConv3x16', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv96', dtype=tf.float32)
    layers.append(net)
    
    toadd = Layers.Conv2D(net.output, convChannels=192, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='SepConv192Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.SepConv2D(net.output, convChannels=192, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv192a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=192, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        name='SepConv192b', dtype=tf.float32)
    layers.append(net)
    
#     added = toadd.output + net.output
    added = tf.concat([toadd.output, net.output], axis=3)
    
    toadd = Layers.Conv2D(added, convChannels=384, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='SepConv384Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU384')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv384a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv384b', dtype=tf.float32)
    layers.append(net)
    
#     added = toadd.output + net.output
    added = tf.concat([toadd.output, net.output], axis=3)
    
    toadd = Layers.Conv2D(added, convChannels=768, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='SepConv768Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU768')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=768, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv768a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=768, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv768b', dtype=tf.float32)
    layers.append(net)
    
#     added = toadd.output + net.output
    added = tf.concat([toadd.output, net.output], axis=3)
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU1024')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=1024, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='SepConv1024', dtype=tf.float32)
    layers.append(net)
    net = Layers.GlobalAvgPool(net.output, name='GlobalAvgPool')
    layers.append(net)
    return net

def SimpleV6(standardized, step, ifTest, layers):
    net = Layers.DepthwiseConv2D(standardized, convChannels=3*16, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        name='DepthwiseConv3x16', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=48, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv48', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv96', dtype=tf.float32)
    layers.append(net)
    
    toadd = Layers.Conv2D(net.output, convChannels=192, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='SepConv192Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.SepConv2D(net.output, convChannels=192, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv192a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=192, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        name='SepConv192b', dtype=tf.float32)
    layers.append(net)
    
#     added = toadd.output + net.output
    added = tf.concat([toadd.output, net.output], axis=3)
    
    toadd = Layers.Conv2D(added, convChannels=384, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='SepConv384Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU384')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv384a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv384b', dtype=tf.float32)
    layers.append(net)
    
#     added = toadd.output + net.output
    added = tf.concat([toadd.output, net.output], axis=3)
    
    toadd = Layers.Conv2D(added, convChannels=768, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='SepConv768Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU768')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=768, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv768a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=768, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='SepConv768b', dtype=tf.float32)
    layers.append(net)
    
#     added = toadd.output + net.output
    added = tf.concat([toadd.output, net.output], axis=3)
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU1024')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=1024, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='SepConv1024', dtype=tf.float32)
    layers.append(net)
    net = Layers.GlobalAvgPool(net.output, name='GlobalAvgPool')
    layers.append(net)
    return net

def SimpleV7(standardized, step, ifTest, layers, numMiddle=2):
    net = Layers.DepthwiseConv2D(standardized, convChannels=3*16, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        name='DepthwiseConv3x16', dtype=tf.float32)
    layers.append(net)
    
    toconcat = Layers.Conv2D(net.output, convChannels=48, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage1_Conv_48a', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layers.Conv2D(toconcat.output, convChannels=96, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage1_Conv1x1_96', dtype=tf.float32)
    layers.append(net)
    net = Layers.DepthwiseConv2D(net.output, convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage1_DepthwiseConv96', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=48, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.Linear, \
                        name='Stage1_Conv1x1_48b', dtype=tf.float32)
    layers.append(net)
    
    concated = tf.concat([toconcat.output, net.output], axis=3)
    
    toconcat = Layers.Conv2D(concated, convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage2_Conv_96a', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layers.Conv2D(toconcat.output, convChannels=192, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage2_Conv1x1_192', dtype=tf.float32)
    layers.append(net)
    net = Layers.DepthwiseConv2D(net.output, convChannels=192, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage2_DepthwiseConv192', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=96, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.Linear, \
                        name='Stage2_Conv1x1_96b', dtype=tf.float32)
    layers.append(net)
    
    concated = tf.concat([toconcat.output, net.output], axis=3)
    
    toconcat = Layers.Conv2D(concated, convChannels=192, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='Stage3_Conv_192a', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layers.Conv2D(toconcat.output, convChannels=384, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage3_Conv1x1_384', dtype=tf.float32)
    layers.append(net)
    net = Layers.DepthwiseConv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage3_DepthwiseConv384', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=192, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.Linear, \
                        name='Stage3_Conv1x1_192b', dtype=tf.float32)
    layers.append(net)
    
    concated = tf.concat([toconcat.output, net.output], axis=3)
    
    toconcat = Layers.Conv2D(concated, convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='Stage4_Conv_384a', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layers.Conv2D(toconcat.output, convChannels=768, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage4_Conv1x1_768', dtype=tf.float32)
    layers.append(net)
    net = Layers.DepthwiseConv2D(net.output, convChannels=768, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage4_DepthwiseConv768', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=384, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.Linear, \
                        name='Stage4_Conv1x1_384b', dtype=tf.float32)
    layers.append(net)
    
    concated = tf.concat([toconcat.output, net.output], axis=3)
    toadd = Layers.Conv2D(concated, convChannels=768, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.Linear, \
                        name='SepConv768Toadd', dtype=tf.float32)
    layers.append(toadd)
    conved = toadd.output
    
    for idx in range(numMiddle):
        net = Layers.Activation(conved, Layers.ReLU, name='ActMiddle'+str(idx)+'_1')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=768, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_1', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_2')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=768, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_2', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_3')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=768, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_3', dtype=tf.float32)
        layers.append(net)
        conved = net.output + conved
    
    toadd = Layers.Conv2D(conved, convChannels=1536, \
                          convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                          poolType=Layers.MaxPool, poolPadding='SAME', \
                          name='ConvExit1x1_1', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(conved, Layers.ReLU, name='ActExit768_1')
    layers.append(net)
    
    toconcat = Layers.Conv2D(net.output, convChannels=768, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                           poolType=Layers.MaxPool, poolPadding='SAME', \
                           name='ConvExit768_1', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layers.Conv2D(toconcat.output, convChannels=1536, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Exit_Conv1x1_1536', dtype=tf.float32)
    layers.append(net)
    net = Layers.DepthwiseConv2D(net.output, convChannels=1536, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Exit_DepthwiseConv1536', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=768, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.Linear, \
                        name='Exit_Conv1x1_768b', dtype=tf.float32)
    layers.append(net)
   
    concated = tf.concat([toconcat.output, net.output], axis=3)
    added = concated + toadd.output
    
    net = Layers.SepConv2D(added, convChannels=2048, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvExit2048_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.GlobalAvgPool(net.output, name='GlobalAvgPool')
    layers.append(net)
    
    return net


def SimpleV8(standardized, step, ifTest, layers, numMiddle=2):
    net = Layers.DepthwiseConv2D(standardized, convChannels=3*16, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        name='DepthwiseConv3x16', dtype=tf.float32)
    layers.append(net)
    
    toconcat = Layers.Conv2D(net.output, convChannels=48, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage1_Conv1x1_48a', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layers.Conv2D(toconcat.output, convChannels=96, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage1_Conv1x1_96', dtype=tf.float32)
    layers.append(net)
    net = Layers.DepthwiseConv2D(net.output, convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage1_DepthwiseConv96', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=48, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.Linear, \
                        name='Stage1_Conv1x1_48b', dtype=tf.float32)
    layers.append(net)
    
    concated = tf.concat([toconcat.output, net.output], axis=3)
    
    toconcat = Layers.Conv2D(concated, convChannels=96, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='Stage2_Conv1x1_96a', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layers.Conv2D(toconcat.output, convChannels=192, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage2_Conv1x1_192', dtype=tf.float32)
    layers.append(net)
    net = Layers.DepthwiseConv2D(net.output, convChannels=192, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage2_DepthwiseConv192', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=96, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.Linear, \
                        name='Stage2_Conv1x1_96b', dtype=tf.float32)
    layers.append(net)
    
    concated = tf.concat([toconcat.output, net.output], axis=3)
    
    toconcat = Layers.Conv2D(concated, convChannels=192, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='Stage3_Conv1x1_192a', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layers.Conv2D(toconcat.output, convChannels=384, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage3_Conv1x1_384', dtype=tf.float32)
    layers.append(net)
    net = Layers.DepthwiseConv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage3_DepthwiseConv384', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=192, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.Linear, \
                        name='Stage3_Conv1x1_192b', dtype=tf.float32)
    layers.append(net)
    
    concated = tf.concat([toconcat.output, net.output], axis=3)
    
    toadd = Layers.Conv2D(concated, convChannels=384, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.Linear, \
                        name='SepConv384Toadd', dtype=tf.float32)
    layers.append(toadd)
    conved = toadd.output
    
    for idx in range(numMiddle):
        net = Layers.Activation(conved, Layers.ReLU, name='ActMiddle'+str(idx)+'_1')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=384, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_1', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_2')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=384, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_2', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_3')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=384, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_3', dtype=tf.float32)
        layers.append(net)
        conved = net.output + conved
    
    toadd = Layers.Conv2D(conved, convChannels=768, \
                          convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                          poolType=Layers.MaxPool, poolPadding='SAME', \
                          name='ConvExit1x1_1', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(conved, Layers.ReLU, name='ActExit384_1')
    layers.append(net)
    
    toconcat = Layers.SepConv2D(net.output, convChannels=384, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                           poolType=Layers.MaxPool, poolPadding='SAME', \
                           name='ConvExit384_1', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layers.Conv2D(toconcat.output, convChannels=768, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Exit_Conv1x1_768', dtype=tf.float32)
    layers.append(net)
    net = Layers.DepthwiseConv2D(net.output, convChannels=768, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Exit_DepthwiseConv768', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=384, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.Linear, \
                        name='Exit_Conv1x1_384b', dtype=tf.float32)
    layers.append(net)
   
    concated = tf.concat([toconcat.output, net.output], axis=3)
    added = concated + toadd.output
    
    net = Layers.SepConv2D(added, convChannels=1536, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvExit1536_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.GlobalAvgPool(net.output, name='GlobalAvgPool')
    layers.append(net)
    
    return net

def SimpleV9(standardized, step, ifTest, layers, numMiddle=2):
    net = Layers.DepthwiseConv2D(standardized, convChannels=3*12, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        name='DepthwiseConv3x12', dtype=tf.float32)
    layers.append(net)
    
    toconcat = Layers.SepConv2D(net.output, convChannels=36, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage1_Conv1x1_36a', dtype=tf.float32)
    layers.append(toconcat)
    concated = toconcat.output
    
    for idx in range(3):
        net = Layers.Conv2D(concated, convChannels=24, \
                            convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            name='Stage1_Conv1x1_24_'+str(idx), dtype=tf.float32)
        layers.append(net)
        net = Layers.DepthwiseConv2D(net.output, convChannels=24, \
                            convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            name='Stage1_DepthwiseConv24_'+str(idx), dtype=tf.float32)
        layers.append(net)
        net = Layers.Conv2D(net.output, convChannels=12, \
                            convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.Linear, \
                            name='Stage1_Conv1x1_12b_'+str(idx), dtype=tf.float32)
        layers.append(net)
    
        concated = tf.concat([toconcat.output, net.output], axis=3)
    
    toconcat = Layers.SepConv2D(net.output, convChannels=72, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage2_Conv1x1_72a', dtype=tf.float32)
    layers.append(toconcat)
    concated = toconcat.output
    
    for idx in range(3):
        net = Layers.Conv2D(concated, convChannels=48, \
                            convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            name='Stage2_Conv1x1_48_'+str(idx), dtype=tf.float32)
        layers.append(net)
        net = Layers.DepthwiseConv2D(net.output, convChannels=48, \
                            convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            name='Stage2_DepthwiseConv48_'+str(idx), dtype=tf.float32)
        layers.append(net)
        net = Layers.Conv2D(net.output, convChannels=24, \
                            convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.Linear, \
                            name='Stage2_Conv1x1_24b_'+str(idx), dtype=tf.float32)
        layers.append(net)
    
        concated = tf.concat([toconcat.output, net.output], axis=3)
    
    toconcat = Layers.SepConv2D(net.output, convChannels=144, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='Stage3_Conv1x1_72a', dtype=tf.float32)
    layers.append(toconcat)
    concated = toconcat.output
    
    for idx in range(3):
        net = Layers.Conv2D(concated, convChannels=96, \
                            convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            name='Stage3_Conv1x1_96_'+str(idx), dtype=tf.float32)
        layers.append(net)
        net = Layers.DepthwiseConv2D(net.output, convChannels=96, \
                            convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            name='Stage3_DepthwiseConv96_'+str(idx), dtype=tf.float32)
        layers.append(net)
        net = Layers.Conv2D(net.output, convChannels=48, \
                            convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.Linear, \
                            name='Stage3_Conv1x1_48b_'+str(idx), dtype=tf.float32)
        layers.append(net)
    
        concated = tf.concat([toconcat.output, net.output], axis=3)
    
    toadd = Layers.SepConv2D(concated, convChannels=576, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='SepConv576Toadd', dtype=tf.float32)
    layers.append(toadd)
    conved = toadd.output
    
    for idx in range(numMiddle):
        net = Layers.Activation(conved, Layers.ReLU, name='ActMiddle'+str(idx)+'_1')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=576, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_1', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_2')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=576, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_2', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_3')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=576, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_3', dtype=tf.float32)
        layers.append(net)
        conved = net.output + conved
    
    toadd = Layers.Conv2D(conved, convChannels=1152, \
                          convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                          poolType=Layers.MaxPool, poolPadding='SAME', \
                          name='ConvExit1x1_1', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(conved, Layers.ReLU, name='ActExit768_1')
    layers.append(net)
    
    toconcat = Layers.SepConv2D(net.output, convChannels=576, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                           poolType=Layers.MaxPool, poolPadding='SAME', \
                           name='ConvExit768_1', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layers.Conv2D(toconcat.output, convChannels=1152, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Exit_Conv1x1_1152', dtype=tf.float32)
    layers.append(net)
    net = Layers.DepthwiseConv2D(net.output, convChannels=1152, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Exit_DepthwiseConv1152', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=576, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.Linear, \
                        name='Exit_Conv1x1_576b', dtype=tf.float32)
    layers.append(net)
   
    concated = tf.concat([toconcat.output, net.output], axis=3)
    added = concated + toadd.output
    
    net = Layers.SepConv2D(added, convChannels=1152, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvExit1152_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.GlobalAvgPool(net.output, name='GlobalAvgPool')
    layers.append(net)
    
    return net

def SimpleV10(standardized, step, ifTest, layers, numMiddle=2):
    net = Layers.DepthwiseConv2D(standardized, convChannels=3*12, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        name='DepthwiseConv3x12', dtype=tf.float32)
    layers.append(net)
    
    toconcat = Layers.SepConv2D(net.output, convChannels=36, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage1_Conv1x1_36a', dtype=tf.float32)
    layers.append(toconcat)
    concated = toconcat.output
    
    for idx in range(3):
        net = Layers.SepConv2D(concated, convChannels=12, \
                            convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            name='Stage1_SepConv12_'+str(idx), dtype=tf.float32)
        layers.append(net)
        concated = tf.concat([toconcat.output, net.output], axis=3)
    
    toconcat = Layers.SepConv2D(net.output, convChannels=72, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage2_Conv1x1_72a', dtype=tf.float32)
    layers.append(toconcat)
    concated = toconcat.output
    
    for idx in range(3):
        net = Layers.SepConv2D(concated, convChannels=24, \
                            convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            name='Stage2_SepConv24_'+str(idx), dtype=tf.float32)
        layers.append(net)
        concated = tf.concat([toconcat.output, net.output], axis=3)
    
    toconcat = Layers.SepConv2D(net.output, convChannels=144, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='Stage3_Conv1x1_72a', dtype=tf.float32)
    layers.append(toconcat)
    concated = toconcat.output
    
    for idx in range(3):
        net = Layers.SepConv2D(concated, convChannels=48, \
                            convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            name='Stage3_SepConv48_'+str(idx), dtype=tf.float32)
        layers.append(net)
        concated = tf.concat([toconcat.output, net.output], axis=3)
    
    toadd = Layers.SepConv2D(concated, convChannels=576, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='SepConv576Toadd', dtype=tf.float32)
    layers.append(toadd)
    conved = toadd.output
    
    for idx in range(numMiddle):
        net = Layers.Activation(conved, Layers.ReLU, name='ActMiddle'+str(idx)+'_1')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=576, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_1', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_2')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=576, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_2', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_3')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=576, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_3', dtype=tf.float32)
        layers.append(net)
        conved = net.output + conved
    
    toadd = Layers.Conv2D(conved, convChannels=1152, \
                          convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                          poolType=Layers.MaxPool, poolPadding='SAME', \
                          name='ConvExit1x1_1', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(conved, Layers.ReLU, name='ActExit576_1')
    layers.append(net)
    
    toconcat = Layers.SepConv2D(net.output, convChannels=576, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                           poolType=Layers.MaxPool, poolPadding='SAME', \
                           name='ConvExit576_1', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layers.SepConv2D(toconcat.output, convChannels=576, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvExit576_2', dtype=tf.float32)
    layers.append(net)
   
    concated = tf.concat([toconcat.output, net.output], axis=3)
    added = concated + toadd.output
    
    net = Layers.SepConv2D(added, convChannels=1152, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvExit1152_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.GlobalAvgPool(net.output, name='GlobalAvgPool')
    layers.append(net)
    
    return net

def SimpleV11(standardized, step, ifTest, layers, numMiddle=2):
    net = Layers.DepthwiseConv2D(standardized, convChannels=3*16, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        name='DepthwiseConv3x12', dtype=tf.float32)
    layers.append(net)
    
    toconcat = Layers.Conv2D(net.output, convChannels=48, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage1_Conv1x1_48a', dtype=tf.float32)
    layers.append(toconcat)
    concated = toconcat.output
    
    for idx in range(3):
        net = Layers.Conv2D(concated, convChannels=32, \
                            convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            name='Stage1_Conv1x1_32_'+str(idx), dtype=tf.float32)
        layers.append(net)
        net = Layers.DepthwiseConv2D(net.output, convChannels=32, \
                            convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            name='Stage1_DepthwiseConv32_'+str(idx), dtype=tf.float32)
        layers.append(net)
        net = Layers.Conv2D(net.output, convChannels=16, \
                            convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.Linear, \
                            name='Stage1_Conv1x1_16b_'+str(idx), dtype=tf.float32)
        layers.append(net)
    
        concated = tf.concat([toconcat.output, net.output], axis=3)
    
    toconcat = Layers.Conv2D(net.output, convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage2_Conv1x1_72a', dtype=tf.float32)
    layers.append(toconcat)
    concated = toconcat.output
    
    for idx in range(3):
        net = Layers.Conv2D(concated, convChannels=64, \
                            convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            name='Stage2_Conv1x1_64_'+str(idx), dtype=tf.float32)
        layers.append(net)
        net = Layers.DepthwiseConv2D(net.output, convChannels=64, \
                            convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            name='Stage2_DepthwiseConv64_'+str(idx), dtype=tf.float32)
        layers.append(net)
        net = Layers.Conv2D(net.output, convChannels=32, \
                            convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.Linear, \
                            name='Stage2_Conv1x1_32b_'+str(idx), dtype=tf.float32)
        layers.append(net)
    
        concated = tf.concat([toconcat.output, net.output], axis=3)
    
    toconcat = Layers.Conv2D(net.output, convChannels=192, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='Stage3_Conv1x1_192a', dtype=tf.float32)
    layers.append(toconcat)
    concated = toconcat.output
    
    for idx in range(3):
        net = Layers.Conv2D(concated, convChannels=128, \
                            convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            name='Stage3_Conv1x1_128_'+str(idx), dtype=tf.float32)
        layers.append(net)
        net = Layers.DepthwiseConv2D(net.output, convChannels=128, \
                            convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            name='Stage3_DepthwiseConv128_'+str(idx), dtype=tf.float32)
        layers.append(net)
        net = Layers.Conv2D(net.output, convChannels=64, \
                            convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.Linear, \
                            name='Stage3_Conv1x1_64b_'+str(idx), dtype=tf.float32)
        layers.append(net)
    
        concated = tf.concat([toconcat.output, net.output], axis=3)
    
    toconcat = Layers.Conv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage4_Conv1x1_384a', dtype=tf.float32)
    layers.append(toconcat)
    concated = toconcat.output
    
    for idx in range(3):
        net = Layers.Conv2D(concated, convChannels=256, \
                            convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            name='Stage4_Conv1x1_256_'+str(idx), dtype=tf.float32)
        layers.append(net)
        net = Layers.DepthwiseConv2D(net.output, convChannels=256, \
                            convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            name='Stage4_DepthwiseConv256_'+str(idx), dtype=tf.float32)
        layers.append(net)
        net = Layers.Conv2D(net.output, convChannels=128, \
                            convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.Linear, \
                            name='Stage4_Conv1x1_128b_'+str(idx), dtype=tf.float32)
        layers.append(net)
    
        concated = tf.concat([toconcat.output, net.output], axis=3)
    
    toadd = Layers.Conv2D(concated, convChannels=768, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='SepConv768Toadd', dtype=tf.float32)
    layers.append(toadd)
    conved = toadd.output
    
    for idx in range(numMiddle):
        net = Layers.Activation(conved, Layers.ReLU, name='ActMiddle'+str(idx)+'_1')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=768, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_1', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_2')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=768, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_2', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_3')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=768, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_3', dtype=tf.float32)
        layers.append(net)
        conved = net.output + conved
    
    toadd = Layers.Conv2D(conved, convChannels=1536, \
                          convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                          poolType=Layers.MaxPool, poolPadding='SAME', \
                          name='ConvExit1x1_1', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(conved, Layers.ReLU, name='ActExit768_1')
    layers.append(net)
    
    toconcat = Layers.SepConv2D(net.output, convChannels=768, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                           poolType=Layers.MaxPool, poolPadding='SAME', \
                           name='ConvExit768_1', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layers.Conv2D(toconcat.output, convChannels=1536, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Exit_Conv1x1_1536', dtype=tf.float32)
    layers.append(net)
    net = Layers.DepthwiseConv2D(net.output, convChannels=1536, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Exit_DepthwiseConv1536', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=768, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.Linear, \
                        name='Exit_Conv1x1_768b', dtype=tf.float32)
    layers.append(net)
   
    concated = tf.concat([toconcat.output, net.output], axis=3)
    added = concated + toadd.output
    
    net = Layers.SepConv2D(added, convChannels=2048, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvExit2048_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.GlobalAvgPool(net.output, name='GlobalAvgPool')
    layers.append(net)
    
    return net


def SimpleV7X(standardized, step, ifTest, layers, numMiddle=2):
    net = Layers.DepthwiseConv2D(standardized, convChannels=3*16, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        name='DepthwiseConv3x16', dtype=tf.float32)
    layers.append(net)
    
    toconcat = Layers.Conv2D(net.output, convChannels=48, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage1_Conv_48a', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layers.Conv2D(toconcat.output, convChannels=96, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        activation=Layers.ReLU, \
                        name='Stage1_Conv1x1_96', dtype=tf.float32)
    layers.append(net)
    net = Layers.DepthwiseConv2D(net.output, convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        activation=Layers.ReLU, \
                        name='Stage1_DepthwiseConv96', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=48, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.Linear, \
                        name='Stage1_Conv1x1_48b', dtype=tf.float32)
    layers.append(net)
    
    concated = tf.concat([toconcat.output, net.output], axis=3)
    
    toconcat = Layers.SepConv2D(concated, convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage2_Conv_96a', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layers.Conv2D(toconcat.output, convChannels=192, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        activation=Layers.ReLU, \
                        name='Stage2_Conv1x1_192', dtype=tf.float32)
    layers.append(net)
    net = Layers.DepthwiseConv2D(net.output, convChannels=192, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        activation=Layers.ReLU, \
                        name='Stage2_DepthwiseConv192', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=96, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.Linear, \
                        name='Stage2_Conv1x1_96b', dtype=tf.float32)
    layers.append(net)
    
    concated = tf.concat([toconcat.output, net.output], axis=3)
    
    toconcat = Layers.SepConv2D(concated, convChannels=192, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage3_Conv_192a', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layers.Conv2D(toconcat.output, convChannels=384, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        activation=Layers.ReLU, \
                        name='Stage3_Conv1x1_384', dtype=tf.float32)
    layers.append(net)
    net = Layers.DepthwiseConv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        activation=Layers.ReLU, \
                        name='Stage3_DepthwiseConv384', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=192, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.Linear, \
                        name='Stage3_Conv1x1_192b', dtype=tf.float32)
    layers.append(net)
    
    concated = tf.concat([toconcat.output, net.output], axis=3)
    
    toconcat = Layers.SepConv2D(concated, convChannels=384, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage4_Conv_384a', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layers.Conv2D(toconcat.output, convChannels=768, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        activation=Layers.ReLU, \
                        name='Stage4_Conv1x1_768', dtype=tf.float32)
    layers.append(net)
    net = Layers.DepthwiseConv2D(net.output, convChannels=768, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        activation=Layers.ReLU, \
                        name='Stage4_DepthwiseConv768', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=384, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.Linear, \
                        name='Stage4_Conv1x1_384b', dtype=tf.float32)
    layers.append(net)
    
    concated = tf.concat([toconcat.output, net.output], axis=3)
    toadd = Layers.SepConv2D(concated, convChannels=768, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.Linear, \
                        name='SepConv768Toadd', dtype=tf.float32)
    layers.append(toadd)
    conved = toadd.output
    
    for idx in range(numMiddle):
        net = Layers.Activation(conved, Layers.ReLU, name='ActMiddle'+str(idx)+'_1')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=768, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_1', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_2')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=768, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_2', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_3')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=768, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_3', dtype=tf.float32)
        layers.append(net)
        conved = net.output + conved
    
    toadd = Layers.Conv2D(conved, convChannels=1536, \
                          convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                          poolType=Layers.MaxPool, poolPadding='SAME', \
                          name='ConvExit1x1_1', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(conved, Layers.ReLU, name='ActExit768_1')
    layers.append(net)
    
    toconcat = Layers.SepConv2D(net.output, convChannels=768, \
                           convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvExit768_1', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layers.Conv2D(toconcat.output, convChannels=1536, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        activation=Layers.ReLU, \
                        name='Exit_Conv1x1_1536', dtype=tf.float32)
    layers.append(net)
    net = Layers.DepthwiseConv2D(net.output, convChannels=1536, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        activation=Layers.ReLU, \
                        name='Exit_DepthwiseConv1536', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=768, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.Linear, \
                        name='Exit_Conv1x1_768b', dtype=tf.float32)
    layers.append(net)
   
    concated = tf.concat([toconcat.output, net.output], axis=3)
    added = concated + toadd.output
    
    net = Layers.SepConv2D(added, convChannels=2048, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvExit2048_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.GlobalAvgPool(net.output, name='GlobalAvgPool')
    layers.append(net)
    
    return net

def SimpleV7Slim(standardized, step, ifTest, layers, numMiddle=2):
    net = Layers.DepthwiseConv2D(standardized, convChannels=3*8, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        name='DepthwiseConv3x8', dtype=tf.float32)
    layers.append(net)
    
    toconcat = Layers.Conv2D(net.output, convChannels=24, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage1_Conv_24a', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layers.Conv2D(toconcat.output, convChannels=48, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage1_Conv1x1_48', dtype=tf.float32)
    layers.append(net)
    net = Layers.DepthwiseConv2D(net.output, convChannels=48, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage1_DepthwiseConv48', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=24, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.Linear, \
                        name='Stage1_Conv1x1_24b', dtype=tf.float32)
    layers.append(net)
    
    concated = tf.concat([toconcat.output, net.output], axis=3)
    
    toconcat = Layers.Conv2D(concated, convChannels=48, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage2_Conv_48a', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layers.Conv2D(toconcat.output, convChannels=96, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage2_Conv1x1_96', dtype=tf.float32)
    layers.append(net)
    net = Layers.DepthwiseConv2D(net.output, convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage2_DepthwiseConv96', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=48, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.Linear, \
                        name='Stage2_Conv1x1_48b', dtype=tf.float32)
    layers.append(net)
    
    concated = tf.concat([toconcat.output, net.output], axis=3)
    
    toconcat = Layers.Conv2D(concated, convChannels=96, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage3_Conv_96a', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layers.Conv2D(toconcat.output, convChannels=192, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage3_Conv1x1_192', dtype=tf.float32)
    layers.append(net)
    net = Layers.DepthwiseConv2D(net.output, convChannels=192, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage3_DepthwiseConv192', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=96, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.Linear, \
                        name='Stage3_Conv1x1_96b', dtype=tf.float32)
    layers.append(net)
    
    concated = tf.concat([toconcat.output, net.output], axis=3)
    
    toconcat = Layers.Conv2D(concated, convChannels=192, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage4_Conv_192a', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layers.Conv2D(toconcat.output, convChannels=384, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage4_Conv1x1_384', dtype=tf.float32)
    layers.append(net)
    net = Layers.DepthwiseConv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Stage4_DepthwiseConv384', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=192, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.Linear, \
                        name='Stage4_Conv1x1_192b', dtype=tf.float32)
    layers.append(net)
    
    concated = tf.concat([toconcat.output, net.output], axis=3)
    toadd = Layers.Conv2D(concated, convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.Linear, \
                        name='SepConv384Toadd', dtype=tf.float32)
    layers.append(toadd)
    conved = toadd.output
    
    for idx in range(numMiddle):
        net = Layers.Activation(conved, Layers.ReLU, name='ActMiddle'+str(idx)+'_1')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=384, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_1', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_2')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=384, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_2', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_3')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=384, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_3', dtype=tf.float32)
        layers.append(net)
        conved = net.output + conved
    
    toadd = Layers.Conv2D(conved, convChannels=768, \
                          convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                          poolType=Layers.MaxPool, poolPadding='SAME', \
                          name='ConvExit1x1_1', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(conved, Layers.ReLU, name='ActExit768_1')
    layers.append(net)
    
    toconcat = Layers.Conv2D(net.output, convChannels=384, \
                           convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvExit384_1', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layers.Conv2D(toconcat.output, convChannels=768, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Exit_Conv1x1_768', dtype=tf.float32)
    layers.append(net)
    net = Layers.DepthwiseConv2D(net.output, convChannels=768, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Exit_DepthwiseConv768', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=384, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.Linear, \
                        name='Exit_Conv1x1_384b', dtype=tf.float32)
    layers.append(net)
   
    concated = tf.concat([toconcat.output, net.output], axis=3)
    added = concated + toadd.output
    
    net = Layers.SepConv2D(added, convChannels=1024, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvExit1024_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.GlobalAvgPool(net.output, name='GlobalAvgPool')
    layers.append(net)
    
    return net

def Xcpetion(standardized, step, ifTest, layers, numMiddle=8): 
    
    net = Layers.Conv2D(standardized, convChannels=32, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='ConvEntry32_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='ConvEntry64_1', dtype=tf.float32)
    layers.append(net)
    
    toadd = Layers.Conv2D(net.output, convChannels=128, \
                          convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          name='ConvEntry1x1_1', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.SepConv2D(net.output, convChannels=128, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvEntry128_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=128, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           name='ConvEntry128_2', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=256, \
                          convKernel=[1, 1], convStride=[2, 2], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          name='ConvEntry1x1_2', dtype=tf.float32)
    layers.append(toadd)
    
    acted = Layers.Activation(added, Layers.ReLU, name='ReLUEntry256_0')
    layers.append(acted)
    net = Layers.SepConv2D(acted.output, convChannels=256, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvEntry256_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=256, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                           poolType=Layers.MaxPool, poolPadding='SAME', \
                           name='ConvEntry256_2', dtype=tf.float32)
    layers.append(net)
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=728, \
                          convKernel=[1, 1], convStride=[2, 2], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          name='ConvEntry1x1_3', dtype=tf.float32)
    layers.append(toadd)
    
    acted = Layers.Activation(added, Layers.ReLU, name='ReLUEntry728_0')
    layers.append(acted)
    net = Layers.SepConv2D(acted.output, convChannels=728, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvEntry728_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=728, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                           poolType=Layers.MaxPool, poolPadding='SAME', \
                           name='ConvEntry728_2', dtype=tf.float32)
    layers.append(net)
    added = toadd.output + net.output
    conved = added
    
    for idx in range(numMiddle): 
        net = Layers.Activation(conved, Layers.ReLU, name='ActMiddle'+str(idx)+'_1')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=728, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_1', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_2')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=728, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_2', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_3')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=728, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_3', dtype=tf.float32)
        layers.append(net)
        conved = net.output + conved
    
    toadd = Layers.Conv2D(conved, convChannels=1024, \
                          convKernel=[1, 1], convStride=[2, 2], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          name='ConvExit1x1_1', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(conved, Layers.ReLU, name='ActExit728_1')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=728, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvExit728_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=1024, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                           poolType=Layers.MaxPool, poolPadding='SAME', \
                           name='ConvExit1024_1', dtype=tf.float32)
    layers.append(net)
    added = toadd.output + net.output
    
    net = Layers.SepConv2D(added, convChannels=1536, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvExit1536_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=2048, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvExit2048_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.GlobalAvgPool(net.output, name='GlobalAvgPool')
    layers.append(net)
    
    return net

def XcpetionM(standardized, step, ifTest, layers, numMiddle=8): 
    '''Xception M is better than Xcpetion'''
    net = Layers.Conv2D(standardized, convChannels=32, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='ConvEntry32_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='ConvEntry64_1', dtype=tf.float32)
    layers.append(net)
    
    toadd = Layers.Conv2D(net.output, convChannels=128, \
                          convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          name='ConvEntry1x1_1', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.SepConv2D(net.output, convChannels=128, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvEntry128_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=128, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           name='ConvEntry128_2', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=256, \
                          convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                          poolType=Layers.MaxPool, poolPadding='SAME', \
                          name='ConvEntry1x1_2', dtype=tf.float32)
    layers.append(toadd)
    
    acted = Layers.Activation(added, Layers.ReLU, name='ReLUEntry256_0')
    layers.append(acted)
    net = Layers.SepConv2D(acted.output, convChannels=256, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvEntry256_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=256, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                           poolType=Layers.MaxPool, poolPadding='SAME', \
                           name='ConvEntry256_2', dtype=tf.float32)
    layers.append(net)
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=728, \
                          convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                          poolType=Layers.MaxPool, poolPadding='SAME', \
                          name='ConvEntry1x1_3', dtype=tf.float32)
    layers.append(toadd)
    
    acted = Layers.Activation(added, Layers.ReLU, name='ReLUEntry728_0')
    layers.append(acted)
    net = Layers.SepConv2D(acted.output, convChannels=728, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvEntry728_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=728, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                           poolType=Layers.MaxPool, poolPadding='SAME', \
                           name='ConvEntry728_2', dtype=tf.float32)
    layers.append(net)
    added = toadd.output + net.output
    conved = added
    
    for idx in range(numMiddle): 
        net = Layers.Activation(conved, Layers.ReLU, name='ActMiddle'+str(idx)+'_1')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=728, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_1', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_2')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=728, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_2', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_3')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=728, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_3', dtype=tf.float32)
        layers.append(net)
        conved = net.output + conved
    
    toadd = Layers.Conv2D(conved, convChannels=1024, \
                          convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                          poolType=Layers.MaxPool, poolPadding='SAME', \
                          name='ConvExit1x1_1', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(conved, Layers.ReLU, name='ActExit728_1')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=728, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvExit728_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=1024, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                           poolType=Layers.MaxPool, poolPadding='SAME', \
                           name='ConvExit1024_1', dtype=tf.float32)
    layers.append(net)
    added = toadd.output + net.output
    
    net = Layers.SepConv2D(added, convChannels=1536, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvExit1536_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=2048, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvExit2048_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.GlobalAvgPool(net.output, name='GlobalAvgPool')
    layers.append(net)
    
    return net

def XcpetionM2(standardized, step, ifTest, layers, numMiddle=8): 
    '''Xception M2 is worse than Xcpetion'''
    net = Layers.Conv2D(standardized, convChannels=32, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='ConvEntry32_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='ConvEntry64_1', dtype=tf.float32)
    layers.append(net)
    
    toadd = Layers.Conv2D(net.output, convChannels=128, \
                          convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          name='ConvEntry1x1_1', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.SepConv2D(net.output, convChannels=128, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvEntry128_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=128, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           name='ConvEntry128_2', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=256, \
                          convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          name='ConvEntry1x1_2', dtype=tf.float32)
    layers.append(toadd)
    
    acted = Layers.Activation(added, Layers.ReLU, name='ReLUEntry256_0')
    layers.append(acted)
    net = Layers.SepConv2D(acted.output, convChannels=256, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvEntry256_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=256, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                           poolType=Layers.MaxPool, poolPadding='SAME', \
                           name='ConvEntry256_2', dtype=tf.float32)
    layers.append(net)
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=728, \
                          convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          name='ConvEntry1x1_3', dtype=tf.float32)
    layers.append(toadd)
    
    acted = Layers.Activation(added, Layers.ReLU, name='ReLUEntry728_0')
    layers.append(acted)
    net = Layers.SepConv2D(acted.output, convChannels=728, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvEntry728_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=728, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                           poolType=Layers.MaxPool, poolPadding='SAME', \
                           name='ConvEntry728_2', dtype=tf.float32)
    layers.append(net)
    added = toadd.output + net.output
    conved = added
    
    for idx in range(numMiddle): 
        net = Layers.Activation(conved, Layers.ReLU, name='ActMiddle'+str(idx)+'_1')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=728, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_1', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_2')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=728, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_2', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_3')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=728, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_3', dtype=tf.float32)
        layers.append(net)
        conved = net.output + conved
    
    toadd = Layers.Conv2D(conved, convChannels=1024, \
                          convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          name='ConvExit1x1_1', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(conved, Layers.ReLU, name='ActExit728_1')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=728, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvExit728_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=1024, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                           poolType=Layers.MaxPool, poolPadding='SAME', \
                           name='ConvExit1024_1', dtype=tf.float32)
    layers.append(net)
    added = toadd.output + net.output
    
    net = Layers.SepConv2D(added, convChannels=1536, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvExit1536_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=2048, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvExit2048_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.GlobalAvgPool(net.output, name='GlobalAvgPool')
    layers.append(net)
    
    return net
  
def XcpetionM3(standardized, step, ifTest, layers, numMiddle=8): 
    
    net = Layers.DepthwiseConv2D(standardized, convChannels=3*16, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='DepthwiseConv3x16', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=96, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvEntry', dtype=tf.float32)
    layers.append(net)
    
    toadd = Layers.Conv2D(net.output, convChannels=128, \
                          convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          name='ConvEntry1x1_1', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.SepConv2D(net.output, convChannels=128, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvEntry128_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=128, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           name='ConvEntry128_2', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=256, \
                          convKernel=[1, 1], convStride=[2, 2], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          name='ConvEntry1x1_2', dtype=tf.float32)
    layers.append(toadd)
    
    acted = Layers.Activation(added, Layers.ReLU, name='ReLUEntry256_0')
    layers.append(acted)
    net = Layers.SepConv2D(acted.output, convChannels=256, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvEntry256_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=256, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                           poolType=Layers.MaxPool, poolPadding='SAME', \
                           name='ConvEntry256_2', dtype=tf.float32)
    layers.append(net)
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=728, \
                          convKernel=[1, 1], convStride=[2, 2], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          name='ConvEntry1x1_3', dtype=tf.float32)
    layers.append(toadd)
    
    acted = Layers.Activation(added, Layers.ReLU, name='ReLUEntry728_0')
    layers.append(acted)
    net = Layers.SepConv2D(acted.output, convChannels=728, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvEntry728_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=728, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                           poolType=Layers.MaxPool, poolPadding='SAME', \
                           name='ConvEntry728_2', dtype=tf.float32)
    layers.append(net)
    added = toadd.output + net.output
    conved = added
    
    for idx in range(numMiddle): 
        net = Layers.Activation(conved, Layers.ReLU, name='ActMiddle'+str(idx)+'_1')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=728, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_1', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_2')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=728, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_2', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_3')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=728, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_3', dtype=tf.float32)
        layers.append(net)
        conved = net.output + conved
    
    toadd = Layers.Conv2D(conved, convChannels=1024, \
                          convKernel=[1, 1], convStride=[2, 2], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          name='ConvExit1x1_1', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(conved, Layers.ReLU, name='ActExit728_1')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=728, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvExit728_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=1024, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                           poolType=Layers.MaxPool, poolPadding='SAME', \
                           name='ConvExit1024_1', dtype=tf.float32)
    layers.append(net)
    added = toadd.output + net.output
    
    net = Layers.SepConv2D(added, convChannels=1536, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvExit1536_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=2048, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvExit2048_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.GlobalAvgPool(net.output, name='GlobalAvgPool')
    layers.append(net)
    
    return net
  
def XcpetionM3X(standardized, step, ifTest, layers, numMiddle=2): 
    
    net = Layers.DepthwiseConv2D(standardized, convChannels=3*16, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='DepthwiseConv3x16', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=96, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvEntry', dtype=tf.float32)
    layers.append(net)
    
    toadd = Layers.Conv2D(net.output, convChannels=128, \
                          convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          name='ConvEntry1x1_1', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.SepConv2D(net.output, convChannels=128, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvEntry128_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=128, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           name='ConvEntry128_2', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=256, \
                          convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                          poolType=Layers.MaxPool, poolPadding='SAME', \
                          name='ConvEntry1x1_2', dtype=tf.float32)
    layers.append(toadd)
    
    acted = Layers.Activation(added, Layers.ReLU, name='ReLUEntry256_0')
    layers.append(acted)
    net = Layers.SepConv2D(acted.output, convChannels=256, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvEntry256_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=256, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                           poolType=Layers.MaxPool, poolPadding='SAME', \
                           name='ConvEntry256_2', dtype=tf.float32)
    layers.append(net)
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=728, \
                          convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                          poolType=Layers.MaxPool, poolPadding='SAME', \
                          name='ConvEntry1x1_3', dtype=tf.float32)
    layers.append(toadd)
    
    acted = Layers.Activation(added, Layers.ReLU, name='ReLUEntry728_0')
    layers.append(acted)
    net = Layers.SepConv2D(acted.output, convChannels=728, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvEntry728_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=728, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                           poolType=Layers.MaxPool, poolPadding='SAME', \
                           name='ConvEntry728_2', dtype=tf.float32)
    layers.append(net)
    added = toadd.output + net.output
    conved = added
    
    for idx in range(numMiddle): 
        net = Layers.Activation(conved, Layers.ReLU, name='ActMiddle'+str(idx)+'_1')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=728, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_1', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_2')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=728, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_2', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_3')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=728, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_3', dtype=tf.float32)
        layers.append(net)
        conved = net.output + conved
    
    toadd = Layers.Conv2D(conved, convChannels=1024, \
                          convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                          poolType=Layers.MaxPool, poolPadding='SAME', \
                          name='ConvExit1x1_1', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(conved, Layers.ReLU, name='ActExit728_1')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=728, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvExit728_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=1024, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                           poolType=Layers.MaxPool, poolPadding='SAME', \
                           name='ConvExit1024_1', dtype=tf.float32)
    layers.append(net)
    added = toadd.output + net.output
    
    net = Layers.SepConv2D(added, convChannels=1536, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvExit1536_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=2048, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvExit2048_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.GlobalAvgPool(net.output, name='GlobalAvgPool')
    layers.append(net)
    
    return net
  
def XcpetionM3C(standardized, step, ifTest, layers, numMiddle=8): 
    
    net = Layers.DepthwiseConv2D(standardized, convChannels=3*16, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='DepthwiseConv3x16', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=96, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvEntry', dtype=tf.float32)
    layers.append(net)
    
    toadd = Layers.Conv2D(net.output, convChannels=128, \
                          convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          name='ConvEntry1x1_1', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.SepConv2D(net.output, convChannels=128, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvEntry128_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=128, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           name='ConvEntry128_2', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output
    added = tf.concat([added, toadd.output], axis=-1)
    
    toadd = Layers.Conv2D(added, convChannels=256, \
                          convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                          poolType=Layers.MaxPool, poolPadding='SAME', \
                          name='ConvEntry1x1_2', dtype=tf.float32)
    layers.append(toadd)
    
    acted = Layers.Activation(added, Layers.ReLU, name='ReLUEntry256_0')
    layers.append(acted)
    net = Layers.SepConv2D(acted.output, convChannels=256, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvEntry256_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=256, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                           poolType=Layers.MaxPool, poolPadding='SAME', \
                           name='ConvEntry256_2', dtype=tf.float32)
    layers.append(net)
    added = toadd.output + net.output
    added = tf.concat([added, toadd.output], axis=-1)
    
    toadd = Layers.Conv2D(added, convChannels=728, \
                          convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                          poolType=Layers.MaxPool, poolPadding='SAME', \
                          name='ConvEntry1x1_3', dtype=tf.float32)
    layers.append(toadd)
    
    acted = Layers.Activation(added, Layers.ReLU, name='ReLUEntry728_0')
    layers.append(acted)
    net = Layers.SepConv2D(acted.output, convChannels=728, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvEntry728_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=728, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                           poolType=Layers.MaxPool, poolPadding='SAME', \
                           name='ConvEntry728_2', dtype=tf.float32)
    layers.append(net)
    added = toadd.output + net.output
    conved = added
    
    for idx in range(numMiddle): 
        net = Layers.Activation(conved, Layers.ReLU, name='ActMiddle'+str(idx)+'_1')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=728, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_1', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_2')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=728, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_2', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_3')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=728, \
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                               convInit=Layers.XavierInit, convPadding='SAME', \
                               biasInit=Layers.ConstInit(0.0), \
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                               name='ConvMiddle'+str(idx)+'_3', dtype=tf.float32)
        layers.append(net)
        conved = net.output + conved
    
    toadd = Layers.Conv2D(conved, convChannels=1024, \
                          convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                          convInit=Layers.XavierInit, convPadding='SAME', \
                          biasInit=Layers.ConstInit(0.0), \
                          bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                          pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                          poolType=Layers.MaxPool, poolPadding='SAME', \
                          name='ConvExit1x1_1', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(conved, Layers.ReLU, name='ActExit728_1')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=728, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvExit728_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=1024, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                           poolType=Layers.MaxPool, poolPadding='SAME', \
                           name='ConvExit1024_1', dtype=tf.float32)
    layers.append(net)
    added = toadd.output + net.output
    added = tf.concat([added, toadd.output], axis=-1)
    
    net = Layers.SepConv2D(added, convChannels=1536, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvExit1536_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=2048, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='ConvExit2048_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.GlobalAvgPool(net.output, name='GlobalAvgPool')
    layers.append(net)
    
    return net

def ResNet(standardized, step, ifTest, layers): 
    
    def identity(x, filters, name):
        filters1, filters2, filters3 = filters
        kernelSize = [3, 3]
    
        net = Layers.Conv2D(x, convChannels=filters1, \
                            convKernel=kernelSize, convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            activation=Layers.Linear, \
                            name=name+'Conv1', dtype=tf.float32)
        layers.append(net)
        net = Layers.BatchNorm(net.output, step, ifTest, epsilon=1e-5, name=name+'BN1', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name=name+'ReLU1')
        layers.append(net)
        
        net = Layers.Conv2D(net.output, convChannels=filters2, \
                            convKernel=kernelSize, convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            activation=Layers.Linear, \
                            name=name+'Conv2', dtype=tf.float32)
        layers.append(net)
        net = Layers.BatchNorm(net.output, step, ifTest, epsilon=1e-5, name=name+'BN2', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name=name+'ReLU2')
        layers.append(net)
        
        net = Layers.Conv2D(net.output, convChannels=filters3, \
                            convKernel=kernelSize, convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            activation=Layers.Linear, \
                            name=name+'Conv3', dtype=tf.float32)
        layers.append(net)
        net = Layers.BatchNorm(net.output, step, ifTest, epsilon=1e-5, name=name+'BN3', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name=name+'ReLU3')
        layers.append(net)
        
        return net.output + x
    
    def conv(x, filters, name, stride=[2, 2]):
        filters1, filters2, filters3 = filters
        kernelSize = [3, 3]
        
        net = Layers.Conv2D(x, convChannels=filters1, \
                            convKernel=[1, 1], convStride=stride, convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            activation=Layers.Linear, \
                            name=name+'Conv1', dtype=tf.float32)
        layers.append(net)
        net = Layers.BatchNorm(net.output, step, ifTest, epsilon=1e-5, name=name+'BN1', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name=name+'ReLU1')
        layers.append(net)
        
        net = Layers.Conv2D(net.output, convChannels=filters2, \
                            convKernel=kernelSize, convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            activation=Layers.Linear, \
                            name=name+'Conv2', dtype=tf.float32)
        layers.append(net)
        net = Layers.BatchNorm(net.output, step, ifTest, epsilon=1e-5, name=name+'BN2', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name=name+'ReLU2')
        layers.append(net)
        
        net = Layers.Conv2D(net.output, convChannels=filters3, \
                            convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            activation=Layers.Linear, \
                            name=name+'Conv3', dtype=tf.float32)
        layers.append(net)
        net = Layers.BatchNorm(net.output, step, ifTest, epsilon=1e-5, name=name+'BN3', dtype=tf.float32)
        layers.append(net)
        
        shortcut = Layers.Conv2D(x, convChannels=filters3, \
                            convKernel=[1, 1], convStride=stride, convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            activation=Layers.Linear, \
                            name=name+'ConvShortcut', dtype=tf.float32)
        layers.append(shortcut)
        shortcut = Layers.BatchNorm(shortcut.output, step, ifTest, epsilon=1e-5, name=name+'BNShortcut', dtype=tf.float32)
        layers.append(shortcut)
    
        added = net.output + shortcut.output
        net = Layers.Activation(added, Layers.ReLU, name=name+'ReLUFinal')
        layers.append(net)
        
        return net.output
    
    net = Layers.Conv2D(standardized, convChannels=64, \
                            convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            activation=Layers.Linear, \
                            name='ConvEntry1', dtype=tf.float32)
    layers.append(net)
    net = Layers.BatchNorm(net.output, step, ifTest, epsilon=1e-5, name='BNEntry1', dtype=tf.float32)
    layers.append(net)
    net = Layers.Activation(net.output, Layers.ReLU, name='ReLUEntry1')
    layers.append(net)
    net = Layers.Pooling(net.output, Layers.MaxPool, size=[3, 3], stride=[2, 2], padding='SAME', name='PoolEntry1')
    layers.append(net)
    
    net = conv(net.output, [64, 64, 256], name='Stage2a', stride=[1, 1])
    net = identity(net, [64, 64, 256], name='Stage2b')
    net = identity(net, [64, 64, 256], name='Stage2c')
    
    net = conv(net, [128, 128, 512], name='Stage3a', stride=[2, 2])
    net = identity(net, [128, 128, 512], name='Stage3b')
    net = identity(net, [128, 128, 512], name='Stage3c')
    net = identity(net, [128, 128, 512], name='Stage3d')
    
    net = conv(net, [256, 256, 1024], name='Stage4a', stride=[2, 2])
    net = identity(net, [256, 256, 1024], name='Stage4b')
    net = identity(net, [256, 256, 1024], name='Stage4c')
    net = identity(net, [256, 256, 1024], name='Stage4d')
    net = identity(net, [256, 256, 1024], name='Stage4e')
    net = identity(net, [256, 256, 1024], name='Stage4f')
    
    net = conv(net, [512, 512, 2048], name='Stage5a', stride=[2, 2])
    net = identity(net, [512, 512, 2048], name='Stage5b')
    net = identity(net, [512, 512, 2048], name='Stage5c')

    net = Layers.GlobalAvgPool(net, name='GlobalAvgPool')
    layers.append(net)
    
    return net

def VGG(standardized, step, ifTest, layers): 
    '''It is hard to train VGG'''
    net = Layers.Conv2D(standardized, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        activation=Layers.ReLU, \
                        name='Conv64a', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        activation=Layers.ReLU, \
                        name='Conv64b', dtype=tf.float32)
    layers.append(net)
    
    net = Layers.Conv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        activation=Layers.ReLU, \
                        name='Conv128a', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='Conv128b', dtype=tf.float32)
    layers.append(net)
    
    net = Layers.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        activation=Layers.ReLU, \
                        name='Conv256a', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        activation=Layers.ReLU, \
                        name='Conv256b', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        activation=Layers.ReLU, \
                        name='Conv256c', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='Conv256d', dtype=tf.float32)
    layers.append(net)
    
    net = Layers.Conv2D(net.output, convChannels=512, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        activation=Layers.ReLU, \
                        name='Conv512a', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=512, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        activation=Layers.ReLU, \
                        name='Conv512b', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=512, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        activation=Layers.ReLU, \
                        name='Conv512c', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=512, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='Conv512d', dtype=tf.float32)
    layers.append(net)
    
    net = Layers.Flatten(net.output)
    
    net = Layers.FullyConnected(net.output, outputSize=4096, weightInit=Layers.XavierInit, wd=wd, \
                                biasInit=Layers.ConstInit(0.0), \
                                activation=Layers.ReLU, \
                                name='FC1', dtype=tf.float32)
    layers.append(net)
    net = Layers.FullyConnected(net.output, outputSize=4096, weightInit=Layers.XavierInit, wd=wd, \
                                biasInit=Layers.ConstInit(0.0), \
                                activation=Layers.ReLU, \
                                name='FC2', dtype=tf.float32)
    layers.append(net)
    
    return net
    
def VGGM(standardized, step, ifTest, layers): 
    
    net = Layers.Conv2D(standardized, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv64a', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv64b', dtype=tf.float32)
    layers.append(net)
    
    net = Layers.Conv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv128a', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='Conv128b', dtype=tf.float32)
    layers.append(net)
    
    net = Layers.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv256a', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        activation=Layers.ReLU, \
                        name='Conv256b', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv256c', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='Conv256d', dtype=tf.float32)
    layers.append(net)
    
    net = Layers.Conv2D(net.output, convChannels=512, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv512a', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=512, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv512b', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=512, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='Conv512c', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=512, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[2, 2], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='Conv512d', dtype=tf.float32)
    layers.append(net)
    
    net = Layers.Flatten(net.output)
    
    net = Layers.FullyConnected(net.output, outputSize=4096, weightInit=Layers.XavierInit, wd=wd, \
                                biasInit=Layers.ConstInit(0.0), \
                                activation=Layers.ReLU, \
                                name='FC1', dtype=tf.float32)
    layers.append(net)
    net = Layers.BatchNorm(net.output, step, ifTest, epsilon=1e-5, name='BatchNormFC1', dtype=tf.float32)
    layers.append(net)
    net = Layers.FullyConnected(net.output, outputSize=4096, weightInit=Layers.XavierInit, wd=wd, \
                                biasInit=Layers.ConstInit(0.0), \
                                activation=Layers.ReLU, \
                                name='FC2', dtype=tf.float32)
    layers.append(net)
    net = Layers.BatchNorm(net.output, step, ifTest, epsilon=1e-5, name='BatchNormFC2', dtype=tf.float32)
    layers.append(net)
    
    return net
    




