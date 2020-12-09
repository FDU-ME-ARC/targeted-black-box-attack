import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import Layers
import Nets
import FashionMNIST as MNIST

wd = 1e-4
NoiseRange = 60.0

HParamMNIST = {'BatchSize': 200, 
                 'NumSubnets': 10, 
                 'NumPredictor': 1, 
                 'NumGenerator': 1, 
                 'NoiseDecay': 1e-4, 
                 'LearningRate': 1e-3, 
                 'MinLearningRate': 1e-5, 
                 'DecayAfter': 300,
                 'ValidateAfter': 300,
                 'TestSteps': 50,
                 'TotalSteps': 40000}

def preproc(images):
    # Preprocessings
    casted        = tf.cast(images, tf.float32)
    standardized  = tf.identity(casted / 127.5 - 1.0)
        
    return standardized

def Generator(images, targets, numSubnets, step, ifTest, layers):   
    with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE) as scope: 
        net = Layers.DepthwiseConv2D(preproc(images), convChannels=16, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='G_DepthwiseConv3x16', dtype=tf.float32)
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=32, \
                            convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            name='G_SepConv96', dtype=tf.float32)
        layers.append(net)
        
        toadd = Layers.Conv2D(net.output, convChannels=64, \
                            convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                            poolType=Layers.MaxPool, poolPadding='SAME', \
                            name='G_SepConv192Shortcut', dtype=tf.float32)
        layers.append(toadd)
        
        net = Layers.SepConv2D(net.output, convChannels=64, \
                            convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            name='G_SepConv192a', dtype=tf.float32)
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=64, \
                            convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            name='G_SepConv192b', dtype=tf.float32)
        layers.append(net)
        
        added = toadd.output + net.output
        
        toadd = Layers.Conv2D(added, convChannels=128, \
                            convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                            poolType=Layers.MaxPool, poolPadding='SAME', \
                            name='G_SepConv384Shortcut', dtype=tf.float32)
        layers.append(toadd)
        
        net = Layers.Activation(added, activation=Layers.ReLU, name='G_ReLU384')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=128, \
                            convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            name='G_SepConv384a', dtype=tf.float32)
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=128, \
                            convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            name='G_SepConv384b', dtype=tf.float32)
        layers.append(net)
        
        added = toadd.output + net.output
        
        toadd = Layers.Conv2D(added, convChannels=256, \
                            convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                            poolType=Layers.MaxPool, poolPadding='SAME', \
                            name='G_SepConv768Shortcut', dtype=tf.float32)
        layers.append(toadd)
        
        net = Layers.Activation(added, activation=Layers.ReLU, name='G_ReLU768')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=256, \
                            convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            name='G_SepConv768a', dtype=tf.float32)
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=256, \
                            convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            name='G_SepConv768b', dtype=tf.float32)
        layers.append(net)
        
        added = toadd.output + net.output
        
        net = Layers.Activation(added, activation=Layers.ReLU, name='G_ReLU11024')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=512, \
                            convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            name='G_SepConv1024', dtype=tf.float32)
        layers.append(net)
        net = Layers.DeConv2D(net.output, convChannels=64, shapeOutput=[7, 7], \
                            convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            reuse=tf.AUTO_REUSE, name='G_DeConv192', dtype=tf.float32)
        layers.append(net)
        
    with tf.variable_scope('Decoder', reuse=tf.AUTO_REUSE) as scope: 
        subnets = []
        for idx in range(numSubnets): 
            subnet = Layers.DeConv2D(net.output, convChannels=32, shapeOutput=[14, 14], \
                                convKernel=[5, 5], convStride=[2, 2], convWD=wd, \
                                convInit=Layers.XavierInit, convPadding='SAME', \
                                biasInit=Layers.ConstInit(0.0), \
                                bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                                activation=Layers.ReLU, \
                                reuse=tf.AUTO_REUSE, name='G_DeConv96_'+str(idx), dtype=tf.float32)
            layers.append(subnet)
            subnet = Layers.DeConv2D(subnet.output, convChannels=16, shapeOutput=[28, 28], \
                                convKernel=[5, 5], convStride=[2, 2], convWD=wd, \
                                convInit=Layers.XavierInit, convPadding='SAME', \
                                biasInit=Layers.ConstInit(0.0), \
                                bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                                activation=Layers.ReLU, \
                                reuse=tf.AUTO_REUSE, name='G_DeConv48_'+str(idx), dtype=tf.float32)
            layers.append(subnet)
            subnet = Layers.Conv2D(subnet.output, convChannels=1, \
                                convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                                convInit=Layers.NormalInit(0.01), convPadding='SAME', \
                                biasInit=Layers.ConstInit(0.0), \
                                bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                                activation=Layers.Linear, \
                                reuse=tf.AUTO_REUSE, name='G_SepConv3_'+str(idx), dtype=tf.float32)
            layers.append(subnet)
            subnets.append(tf.expand_dims(subnet.output, axis=-1))
        subnets = tf.concat(subnets, axis=-1)
        #weights = tf.one_hot(targets, 10)
        weights = Layers.FullyConnected(tf.one_hot(targets, 10), outputSize=numSubnets, weightInit=Layers.XavierInit, wd=wd, \
                                    biasInit=Layers.ConstInit(0.0), \
                                    activation=Layers.Softmax, \
                                    reuse=tf.AUTO_REUSE, name='G_WeightsMoE', dtype=tf.float32)
        layers.append(weights)
        #moe = tf.transpose(tf.transpose(subnets, [1, 2, 3, 0, 4]) * weights, [3, 0, 1, 2, 4])
        moe = tf.transpose(tf.transpose(subnets, [1, 2, 3, 0, 4]) * weights.output, [3, 0, 1, 2, 4])
        noises = tf.nn.tanh(tf.reduce_mean(moe, -1)) * NoiseRange
        print('Shape of Noises: ', noises.shape)
    
    varGE = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Encoder')
    varGD = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Decoder')
    
    return noises, varGE, varGD

def Predictor(images, step, ifTest, layers): 
    net = Layers.DepthwiseConv2D(preproc(tf.clip_by_value(images, 0, 255)), convChannels=16, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='P_DepthwiseConv3x16', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=32, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='P_SepConv96', dtype=tf.float32)
    layers.append(net)
    
    toadd = Layers.Conv2D(net.output, convChannels=64, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='P_SepConv192Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.SepConv2D(net.output, convChannels=64, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='P_SepConv192a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        name='P_SepConv192b', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=128, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='P_SepConv384Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU384')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='P_SepConv384a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='P_SepConv384b', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=256, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='P_SepConv768Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='P_ReLU768')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='P_SepConv768a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='P_SepConv768b', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='P_ReLU11024')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=512, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='P_SepConv1024', dtype=tf.float32)
    layers.append(net)
    net = Layers.GlobalAvgPool(net.output, name='P_GlobalAvgPool')
    layers.append(net)
    logits = Layers.FullyConnected(net.output, outputSize=10, weightInit=Layers.XavierInit, wd=wd, \
                                biasInit=Layers.ConstInit(0.0), \
                                activation=Layers.Linear, \
                                reuse=tf.AUTO_REUSE, name='P_FC_classes', dtype=tf.float32)
    layers.append(logits)
    
    return logits.output

def PredictorG(images, step, ifTest, layers): 
    net = Layers.DepthwiseConv2D(preproc(tf.clip_by_value(images, 0, 255)), convChannels=16, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='P_DepthwiseConv3x16', dtype=tf.float32)
    net = Layers.SepConv2D(net.output, convChannels=32, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='P_SepConv96', dtype=tf.float32)
    
    toadd = Layers.Conv2D(net.output, convChannels=64, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='P_SepConv192Shortcut', dtype=tf.float32)
    
    net = Layers.SepConv2D(net.output, convChannels=64, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='P_SepConv192a', dtype=tf.float32)
    net = Layers.SepConv2D(net.output, convChannels=64, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        name='P_SepConv192b', dtype=tf.float32)
    
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=128, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='P_SepConv384Shortcut', dtype=tf.float32)
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU384')
    net = Layers.SepConv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='P_SepConv384a', dtype=tf.float32)
    net = Layers.SepConv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='P_SepConv384b', dtype=tf.float32)
    
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=256, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='P_SepConv768Shortcut', dtype=tf.float32)
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='P_ReLU768')
    net = Layers.SepConv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='P_SepConv768a', dtype=tf.float32)
    net = Layers.SepConv2D(net.output, convChannels=256, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='P_SepConv768b', dtype=tf.float32)
    
    added = toadd.output + net.output
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='P_ReLU11024')
    net = Layers.SepConv2D(net.output, convChannels=512, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='P_SepConv1024', dtype=tf.float32)
    net = Layers.GlobalAvgPool(net.output, name='P_GlobalAvgPool')
    logits = Layers.FullyConnected(net.output, outputSize=10, weightInit=Layers.XavierInit, wd=wd, \
                                biasInit=Layers.ConstInit(0.0), \
                                activation=Layers.Linear, \
                                reuse=tf.AUTO_REUSE, name='P_FC_classes', dtype=tf.float32)
    
    return logits.output


class NetMNIST(Nets.Net):
    
    def __init__(self, shapeImages, enemy, numMiddle=2, HParam=HParamMNIST):
        Nets.Net.__init__(self)
        
        self._init = False
        self._numMiddle    = numMiddle
        self._HParam       = HParam
        self._graph        = tf.Graph()
        self._sess         = tf.Session(graph=self._graph)
        self._enemy        = enemy
        
        with self._graph.as_default(): 
            self._ifTest        = tf.Variable(False, name='ifTest', trainable=False, dtype=tf.bool)
            self._step          = tf.Variable(0, name='step', trainable=False, dtype=tf.int32)
            self._phaseTrain    = tf.assign(self._ifTest, False)
            self._phaseTest     = tf.assign(self._ifTest, True)
            
            # Inputs
            self._images = tf.placeholder(dtype=tf.float32, shape=[self._HParam['BatchSize']]+shapeImages, \
                                          name='CIFAR10_images')
            self._labels = tf.placeholder(dtype=tf.int64, shape=[self._HParam['BatchSize']], \
                                          name='CIFAR10_labels')
            self._targets = tf.placeholder(dtype=tf.int64, shape=[self._HParam['BatchSize']], \
                                          name='CIFAR10_targets')
            
            # Net
            with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE) as scope: 
                self._generator, self._varsGE, self._varsGD = Generator(self._images, self._targets, self._HParam['NumSubnets'], self._step, self._ifTest, self._layers)
            self._noises = self._generator
            self._adversary = self._noises + self._images
            with tf.variable_scope('Predictor', reuse=tf.AUTO_REUSE) as scope: 
                self._predictor = Predictor(self._images, self._step, self._ifTest, self._layers)
                self._predictorG = PredictorG(self._adversary, self._step, self._ifTest, self._layers)
            self._inference = self.inference(self._predictor)
            self._accuracy = tf.reduce_mean(tf.cast(tf.equal(self._inference, self._labels), tf.float32))
            self._loss = 0
            self._updateOps = []
            for elem in self._layers: 
                if len(elem.losses) > 0: 
                    for tmp in elem.losses: 
                        self._loss += tmp
            for elem in self._layers: 
                if len(elem.updateOps) > 0: 
                    for tmp in elem.updateOps: 
                        self._updateOps.append(tmp)
            self._lossPredictor = self.lossClassify(self._predictor, self._labels, name='lossP') + self._loss
            self._lossGenerator = self.lossClassify(self._predictorG, self._targets, name='lossG') + self._HParam['NoiseDecay'] * tf.reduce_mean(tf.norm(self._noises)) + self._loss
            print(self.summary)
            print("\n Begin Training: \n")
                    
            # Saver
            self._saver = tf.train.Saver(max_to_keep=5)
        
    def preproc(self, images):
        # Preprocessings
        casted        = tf.cast(images, tf.float32)
        standardized  = tf.identity(casted / 127.5 - 1.0, name='training_standardized')
            
        return standardized
        
    def inference(self, logits):
        return tf.argmax(logits, axis=-1, name='inference')
    
    def lossClassify(self, logits, labels, name='cross_entropy'):
        net = Layers.CrossEntropy(logits, labels, name=name)
        self._layers.append(net)
        return net.output
    
    def train(self, genTrain, genTest, pathLoad=None, pathSave=None):
        with self._graph.as_default(): 
            #self._lr = tf.train.exponential_decay(self._HParam['LearningRate'], \
                                                 #global_step=self._step, \
                                                 #decay_steps=self._HParam['DecayAfter']*5, \
                                                 #decay_rate=0.2) + self._HParam['MinLearningRate']
            self._lr = tf.Variable(self._HParam['LearningRate'], trainable=False)
            self._lr2 = tf.Variable(self._HParam['LearningRate'], trainable=False)
            #self._lr2 = tf.train.exponential_decay(self._HParam['LearningRate'], \
                                                 #global_step=self._step, \
                                                 #decay_steps=self._HParam['DecayAfter'], \
                                                 #decay_rate=0.5) + self._HParam['MinLearningRate']
            self._lrDecay1 = tf.assign(self._lr, self._lr * 0.1)
            self._lrDecay2 = tf.assign(self._lr2, self._lr2 * 0.1)
            self._stepInc = tf.assign(self._step, self._step+1)
            self._varsG = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
            self._varsP = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Predictor')
            self._optimizerG = tf.train.AdamOptimizer(self._lr, epsilon=1e-8)
            self._optimizerP = tf.train.AdamOptimizer(self._lr, epsilon=1e-8).minimize(self._lossPredictor, var_list=self._varsP)
            gradientsG = self._optimizerG.compute_gradients(self._lossGenerator, var_list=self._varsG)
            capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gradientsG]
            self._optimizerG = self._optimizerG.apply_gradients(capped_gvs)
            #self._lr = tf.Variable(self._HParam['LearningRate'], trainable=False)
            #self._lrDecay1 = tf.assign(self._lr, self._lr * 0.1)
            #self._stepInc = tf.assign(self._step, self._step+1)
            #self._varsP = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Predictor')
            #self._optimizerGE = tf.train.AdamOptimizer(self._lr, epsilon=1e-8)
            #self._optimizerGD = tf.train.AdamOptimizer(self._lr, epsilon=1e-8)
            #self._optimizerP = tf.train.AdamOptimizer(self._lr, epsilon=1e-8).minimize(self._lossPredictor, var_list=self._varsP)
            #gradientsGE = self._optimizerGE.compute_gradients(self._lossGenerator, var_list=self._varsGE)
            #gradientsGD = self._optimizerGD.compute_gradients(self._lossGenerator, var_list=self._varsGD)
            #capped_GE = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gradientsGE]
            #capped_GD = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gradientsGD]
            ## capped_GE = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gradientsGE]
            ## capped_GD = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gradientsGD]
            #self._optimizerGE = self._optimizerGE.apply_gradients(capped_GE)
            #self._optimizerGD = self._optimizerGD.apply_gradients(capped_GD)
            #with tf.control_dependencies([self._optimizerGE, self._optimizerGD]): 
                #self._optimizerG = tf.no_op()
            #self._optimizerG = tf.train.AdamOptimizer(self._lr*10, epsilon=1e-8).minimize(self._lossGenerator, var_list=self._varsG)
            #self._optimizerP = tf.train.AdamOptimizer(self._lr, epsilon=1e-8).minimize(self._lossPredictor, var_list=self._varsP)
            
            # Initialize all
            self._sess.run(tf.global_variables_initializer())
            
            if pathLoad is not None:
                self.load(pathLoad)
                
            self.evaluate(genTest)
#             self.sample(genTest)
            
            self._sess.run([self._phaseTrain])
            if pathSave is not None:
                self.save(pathSave)
            
            globalStep = 0
            
            #print('Warming up. ')
            #for idx in range(300): 
                #data, label, target = next(genTrain)
                #adversary = data
                #label = self._enemy.infer(adversary)
                #loss, accu, _ = \
                    #self._sess.run([self._lossPredictor, \
                                    #self._accuracy, self._optimizerP], \
                                    #feed_dict={self._images: adversary, \
                                                #self._labels: label, \
                                            #self._targets: target})
                #print('\rPredictor => Step: ', idx-300, \
                        #'; Loss: %.3f'% loss, \
                        #'; Accuracy: %.3f'% accu, \
                        #end='')
            
            #warmupAccu = 0.0
            #for idx in range(50): 
                #data, label, target = next(genTest)
                #adversary = data
                #label = self._enemy.infer(adversary)
                #loss, accu, _ = \
                    #self._sess.run([self._lossPredictor, \
                                    #self._accuracy, self._optimizerP], \
                                    #feed_dict={self._images: adversary, \
                                                #self._labels: label, \
                                            #self._targets: target})
                #warmupAccu += accu / 50
            #print('\nWarmup Accuracy: ', warmupAccu)
            #self._sess.run(self._lrDecay2)
            
            while globalStep < self._HParam['TotalSteps']: 
                
                self._sess.run(self._stepInc)
                
                for _ in range(self._HParam['NumPredictor']): 
                    data, label, target = next(genTrain)
                    adversary = self._sess.run(self._adversary, \
                                               feed_dict={self._images: data, \
                                                          self._labels: label, \
                                                          self._targets: target})
                    #label = self._enemy.infer(data)
                    #loss, accu, globalStep, _ = \
                        #self._sess.run([self._lossPredictor, \
                                        #self._accuracy, self._step, self._optimizerP], \
                                        #feed_dict={self._images: data, \
                                                   #self._labels: label, \
                                                   #self._targets: target})
                    data = data + (np.random.rand(self._HParam['BatchSize'], 28, 28, 1) - 0.5) * NoiseRange * 2
                    label = self._enemy.infer(data)
                    loss, accu, globalStep, _ = \
                        self._sess.run([self._lossPredictor, \
                                        self._accuracy, self._step, self._optimizerP], \
                                        feed_dict={self._images: data, \
                                                   self._labels: label, \
                                                   self._targets: target})
                    #data = np.random.rand(self._HParam['BatchSize'], 28, 28, 1) * 255
                    #label = self._enemy.infer(data)
                    #loss, accu, globalStep, _ = \
                        #self._sess.run([self._lossPredictor, \
                                        #self._accuracy, self._step, self._optimizerP], \
                                        #feed_dict={self._images: data, \
                                                   #self._labels: label, \
                                                   #self._targets: target})
                    label = self._enemy.infer(adversary)
                    loss, accu, globalStep, _ = \
                        self._sess.run([self._lossPredictor, \
                                        self._accuracy, self._step, self._optimizerP], \
                                        feed_dict={self._images: adversary, \
                                                   self._labels: label, \
                                                   self._targets: target})
                    print('\rPredictor => Step: ', globalStep, \
                                '; Loss: %.3f'% loss, \
                                '; Accuracy: %.3f'% accu, \
                                end='')
                    
                for _ in range(self._HParam['NumGenerator']): 
                    data, label, target = next(genTrain)
                    refs = self._enemy.infer(data)
                    for idx in range(data.shape[0]):
                        if refs[idx] == target[idx]: 
                            tmp = random.randint(0, 9)
                            while tmp == refs[idx]: 
                                tmp = random.randint(0, 9)
                            target[idx] = tmp
                    loss, adversary, globalStep, _ = \
                        self._sess.run([self._lossGenerator, \
                                        self._adversary, self._step, self._optimizerG], \
                                        feed_dict={self._images: data, \
                                                self._labels: refs, \
                                                self._targets: target})
                    results = self._enemy.infer(adversary)
                    accu = np.mean(target==results)
                    fullrate = np.mean(refs!=results)
                    
                    print('\rGenerator => Step: ', globalStep, \
                            '; Loss: %.3f'% loss, \
                            '; Accuracy: %.3f'% accu, \
                            '; FoolRate: %.3f'% fullrate, \
                            end='')
                
                if globalStep % self._HParam['ValidateAfter'] == 0: 
                    self.evaluate(genTest)
                    data, label, target = next(genTest)
                    adversary = \
                        self._sess.run(self._adversary, \
                                        feed_dict={self._images: data, \
                                                self._labels: label, \
                                                self._targets: target})
                    refs = self._enemy.infer(data)
                    results = self._enemy.infer(adversary)
                    print((adversary-data)[1, 10:15, 10:15])
                    print((adversary-data).max())
                    print((adversary-data).min())
                    # print(np.max(adversary-data))
                    # print(np.min(adversary-data))
                    # print((adversary-data)[1])
                    print(list(zip(label, refs, results, target)))
                    if pathSave is not None:
                        self.save(pathSave)
                    self._sess.run([self._phaseTrain])
                
                #if globalStep == 1501: 
                    #self._sess.run(self._lrDecay1)
                
                if (globalStep == 5701 or globalStep == 7501): 
                    self._sess.run(self._lrDecay1)
                    print('Learning rate decayed. ')
                
    def evaluate(self, genTest, path=None):
        if path is not None:
            self.load(path)
        
        totalLoss  = 0.0
        totalAccu  = 0.0
        totalFullRate  = 0.0
        self._sess.run([self._phaseTest])  
        for _ in range(self._HParam['TestSteps']): 
            data, label, target = next(genTest)
            refs = self._enemy.infer(data)
            for idx in range(data.shape[0]):
                if refs[idx] == target[idx]: 
                    tmp = random.randint(0, 9)
                    while tmp == refs[idx]: 
                        tmp = random.randint(0, 9)
                    target[idx] = tmp
            loss, adversary = \
                self._sess.run([self._lossGenerator, \
                                self._adversary], \
                                feed_dict={self._images: data, \
                                           self._labels: refs, \
                                           self._targets: target})
            adversary = adversary.clip(0, 255).astype(np.uint8)
            results = self._enemy.infer(adversary)
            accu = np.mean(target==results)
            fullrate = np.mean(refs!=results)
            totalLoss += loss
            totalAccu += accu
            totalFullRate += fullrate
        totalLoss /= self._HParam['TestSteps']
        totalAccu /= self._HParam['TestSteps']
        totalFullRate /= self._HParam['TestSteps']
        print('\nTest: Loss: ', totalLoss, \
              '; Accu: ', totalAccu, 
              '; FullRate: ', totalFullRate)
    
    def sample(self, genTest, path=None):
        if path is not None:
            self.load(path)
            
        self._sess.run([self._phaseTest])  
        data, label, target = next(genTest)
        refs = self._enemy.infer(data)
        for idx in range(data.shape[0]):
            if refs[idx] == target[idx]: 
                tmp = random.randint(0, 9)
                while tmp == refs[idx]: 
                    tmp = random.randint(0, 9)
                target[idx] = tmp
        loss, adversary = \
            self._sess.run([self._lossGenerator, \
                            self._adversary], \
                            feed_dict={self._images: data, \
                                        self._labels: refs, \
                                        self._targets: target})
        adversary = adversary.clip(0, 255).astype(np.uint8)
        results = self._enemy.infer(adversary)
        
        
        for idx in range(10): 
            for jdx in range(3): 
                plt.subplot(10, 6, idx*6+jdx*2+1)
                plt.imshow(data[idx*3+jdx, :, :, 0], cmap='gray')
                plt.subplot(10, 6, idx*6+jdx*2+2)
                plt.imshow(adversary[idx*3+jdx, :, :, 0], cmap='gray')
                print([refs[idx*3+jdx], results[idx*3+jdx], target[idx*3+jdx]])
        plt.show()
    
    def save(self, path):
        self._saver.save(self._sess, path, global_step=self._step)
    
    def load(self, path):
        self._saver.restore(self._sess, path)

if __name__ == '__main__':
    enemy = MNIST.NetMNIST([28, 28, 1], 2)
    enemy.load('./ClassifyFashionMNIST/netcifar10.ckpt-5400')
    net = NetMNIST([28, 28, 1], enemy=enemy, numMiddle=2) 
    batchTrain, batchTest = MNIST.generatorsAdv(BatchSize=HParamMNIST['BatchSize'], preprocSize=[28, 28, 1])
    net.train(batchTrain, batchTest, pathLoad='./AttackFashionMNIST/netcifar10.ckpt-7500', pathSave='./AttackFashionMNIST/netcifar10.ckpt')
    #net.sample(batchTest, './AttackFashionMNIST/netcifar10.ckpt-3600')
