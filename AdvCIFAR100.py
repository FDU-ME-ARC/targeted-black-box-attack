import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import Layers
import Nets
import CIFAR100
import Preproc

wd = 1e-4
NoiseRange = 10.0


HParamCIFAR10 = {'BatchSize': 200, 
                 'NumSubnets': 30, 
                 'NumPredictor': 1, 
                 'NumGenerator': 1, 
                 'NoiseDecay': 1e-5, 
                 'LearningRate': 1e-3, 
                 'MinLearningRate': 1e-5, 
                 'DecayAfter': 300,
                 'ValidateAfter': 300,
                 'TestSteps': 50,
                 'TotalSteps': 60000}

def preproc(images):
    # Preprocessings
    casted        = tf.cast(images, tf.float32)
    standardized  = tf.identity(casted / 127.5 - 1.0)
        
    return standardized

def Generator(images, targets, numSubnets, step, ifTest, layers):   
    net = Layers.DepthwiseConv2D(preproc(images), convChannels=3*16, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='G_DepthwiseConv3x16', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='G_SepConv96', dtype=tf.float32)
    layers.append(net)
    
    toadd = Layers.Conv2D(net.output, convChannels=192, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='G_SepConv192Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.SepConv2D(net.output, convChannels=192, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='G_SepConv192a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=192, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        name='G_SepConv192b', dtype=tf.float32)
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
                        name='G_SepConv384Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='G_ReLU384')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='G_SepConv384a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='G_SepConv384b', dtype=tf.float32)
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
                        name='G_SepConv768Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='G_ReLU768')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=768, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='G_SepConv768a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=768, \
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
    net = Layers.SepConv2D(net.output, convChannels=1024, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='G_SepConv1024', dtype=tf.float32)
    layers.append(net)
    net = Layers.DeConv2D(net.output, convChannels=128, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        reuse=tf.AUTO_REUSE, name='G_DeConv192_', dtype=tf.float32)
    layers.append(net)
    subnets = []
    for idx in range(numSubnets): 
        subnet = Layers.DeConv2D(net.output, convChannels=64, \
                            convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            reuse=tf.AUTO_REUSE, name='G_DeConv96_'+str(idx), dtype=tf.float32)
        layers.append(subnet)
        subnet = Layers.DeConv2D(subnet.output, convChannels=32, \
                            convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.ReLU, \
                            reuse=tf.AUTO_REUSE, name='G_DeConv48_'+str(idx), dtype=tf.float32)
        layers.append(subnet)
        subnet = Layers.Conv2D(subnet.output, convChannels=3, \
                            convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                            convInit=Layers.XavierInit, convPadding='SAME', \
                            biasInit=Layers.ConstInit(0.0), \
                            bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                            activation=Layers.Linear, \
                            reuse=tf.AUTO_REUSE, name='G_SepConv3_'+str(idx), dtype=tf.float32)
        layers.append(subnet)
        subnets.append(tf.expand_dims(subnet.output, axis=-1))
    subnets = tf.concat(subnets, axis=-1)
    weights = Layers.FullyConnected(tf.one_hot(targets, 100), outputSize=numSubnets, weightInit=Layers.XavierInit, wd=0.0, \
                                biasInit=Layers.ConstInit(0.0), \
                                activation=Layers.Softmax, \
                                reuse=tf.AUTO_REUSE, name='G_WeightsMoE', dtype=tf.float32)
    layers.append(weights)
    #weights = tf.one_hot(targets, 100)
    moe = tf.transpose(tf.transpose(subnets, [1, 2, 3, 0, 4]) * weights.output, [3, 0, 1, 2, 4])
    #moe = tf.transpose(tf.transpose(subnets, [1, 2, 3, 0, 4]) * weights, [3, 0, 1, 2, 4])
    noises = (tf.nn.tanh(tf.reduce_sum(moe, -1)) - 0.5) * NoiseRange * 2
    print('Shape of Noises: ', noises.shape)
    
    return noises

def Predictor(images, step, ifTest, layers): 
    net = Layers.DepthwiseConv2D(preproc(tf.clip_by_value(images, 0, 255)), convChannels=3*16, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='P_DepthwiseConv3x16', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='P_SepConv96', dtype=tf.float32)
    layers.append(net)
    
    toadd = Layers.Conv2D(net.output, convChannels=192, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='P_SepConv192Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.SepConv2D(net.output, convChannels=192, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='P_SepConv192a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=192, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        name='P_SepConv192b', dtype=tf.float32)
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
                        name='P_SepConv384Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU384')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='P_SepConv384a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='P_SepConv384b', dtype=tf.float32)
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
                        name='P_SepConv768Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='P_ReLU768')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=768, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='P_SepConv768a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=768, \
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
    net = Layers.SepConv2D(net.output, convChannels=1024, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='P_SepConv1024', dtype=tf.float32)
    layers.append(net)
    net = Layers.GlobalAvgPool(net.output, name='P_GlobalAvgPool')
    layers.append(net)
    logits = Layers.FullyConnected(net.output, outputSize=100, weightInit=Layers.XavierInit, wd=wd, \
                                biasInit=Layers.ConstInit(0.0), \
                                activation=Layers.Linear, \
                                reuse=tf.AUTO_REUSE, name='P_FC_classes', dtype=tf.float32)
    layers.append(logits)
    
    return logits.output

def PredictorG(images, step, ifTest, layers): 
    net = Layers.DepthwiseConv2D(preproc(tf.clip_by_value(images, 0, 255)), convChannels=3*16, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='P_DepthwiseConv3x16', dtype=tf.float32)
    net = Layers.SepConv2D(net.output, convChannels=96, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='P_SepConv96', dtype=tf.float32)
    
    toadd = Layers.Conv2D(net.output, convChannels=192, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='P_SepConv192Shortcut', dtype=tf.float32)
    
    net = Layers.SepConv2D(net.output, convChannels=192, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='P_SepConv192a', dtype=tf.float32)
    net = Layers.SepConv2D(net.output, convChannels=192, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        name='P_SepConv192b', dtype=tf.float32)
    
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=384, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='P_SepConv384Shortcut', dtype=tf.float32)
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU384')
    net = Layers.SepConv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='P_SepConv384a', dtype=tf.float32)
    net = Layers.SepConv2D(net.output, convChannels=384, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='P_SepConv384b', dtype=tf.float32)
    
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=768, \
                        convKernel=[1, 1], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        pool=True, poolSize=[3, 3], poolStride=[2, 2], \
                        poolType=Layers.MaxPool, poolPadding='SAME', \
                        name='P_SepConv768Shortcut', dtype=tf.float32)
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='P_ReLU768')
    net = Layers.SepConv2D(net.output, convChannels=768, \
                        convKernel=[3, 3], convStride=[2, 2], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='P_SepConv768a', dtype=tf.float32)
    net = Layers.SepConv2D(net.output, convChannels=768, \
                        convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                        convInit=Layers.XavierInit, convPadding='SAME', \
                        biasInit=Layers.ConstInit(0.0), \
                        bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                        activation=Layers.ReLU, \
                        name='P_SepConv768b', dtype=tf.float32)
    
    added = toadd.output + net.output
    
    net = Layers.Activation(added, activation=Layers.ReLU, name='P_ReLU11024')
    net = Layers.SepConv2D(net.output, convChannels=1024, \
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd, \
                           convInit=Layers.XavierInit, convPadding='SAME', \
                           biasInit=Layers.ConstInit(0.0), \
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5, \
                           activation=Layers.ReLU, \
                           name='P_SepConv1024', dtype=tf.float32)
    net = Layers.GlobalAvgPool(net.output, name='P_GlobalAvgPool')
    logits = Layers.FullyConnected(net.output, outputSize=100, weightInit=Layers.XavierInit, wd=wd, \
                                biasInit=Layers.ConstInit(0.0), \
                                activation=Layers.Linear, \
                                reuse=tf.AUTO_REUSE, name='P_FC_classes', dtype=tf.float32)
    
    return logits.output


class NetCIFAR10(Nets.Net):
    
    def __init__(self, shapeImages, enemy, numMiddle=2, HParam=HParamCIFAR10):
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
                self._generator = Generator(self._images, self._targets, self._HParam['NumSubnets'], self._step, self._ifTest, self._layers)
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
            # self._lr = tf.train.exponential_decay(self._HParam['LearningRate'], \
            #                                      global_step=self._step, \
            #                                      decay_steps=self._HParam['DecayAfter'], \
            #                                      decay_rate=1.0) + self._HParam['MinLearningRate']
            self._lr = tf.Variable(self._HParam['LearningRate'], trainable=False)
            self._lrDecay1 = tf.assign(self._lr, self._lr * 0.1)
            self._stepInc = tf.assign(self._step, self._step+1)
            self._varsG = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
            self._varsP = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Predictor')
            self._optimizerG = tf.train.AdamOptimizer(self._lr, epsilon=1e-8)#tf.train.GradientDescentOptimizer(self._lr*100)
            self._optimizerP = tf.train.AdamOptimizer(self._lr, epsilon=1e-8).minimize(self._lossPredictor, var_list=self._varsP)
            gradientsG = self._optimizerG.compute_gradients(self._lossGenerator, var_list=self._varsG)
            capped_gvs = [(grad, var) for grad, var in gradientsG]
            #capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gradientsG]
            self._optimizerG = self._optimizerG.apply_gradients(capped_gvs)
            #self._optimizerG = tf.train.AdamOptimizer(self._lr*10, epsilon=1e-8).minimize(self._lossGenerator, var_list=self._varsG)
            #self._optimizerP = tf.train.AdamOptimizer(self._lr, epsilon=1e-8).minimize(self._lossPredictor, var_list=self._varsP)
            
            # Initialize all
            self._sess.run(tf.global_variables_initializer())
            
            if pathLoad is not None:
                self.load(pathLoad)
            else:
                print('Warming up. ')
                for idx in range(300): 
                    data, label, target = next(genTrain)
                    label = np.array(self._enemy.infer(data))
                    loss, accu, _ = \
                        self._sess.run([self._lossPredictor, \
                                        self._accuracy, self._optimizerP], \
                                        feed_dict={self._images: data, \
                                                    self._labels: label, \
                                                self._targets: target})
                    print('\rPredictor => Step: ', idx-300, \
                            '; Loss: %.3f'% loss, \
                            '; Accuracy: %.3f'% accu, \
                            end='')
                
                warmupAccu = 0.0
                for idx in range(50): 
                    data, label, target = next(genTest)
                    label = np.array(self._enemy.infer(data))
                    loss, accu, _ = \
                        self._sess.run([self._lossPredictor, \
                                        self._accuracy, self._optimizerP], \
                                        feed_dict={self._images: data, \
                                                    self._labels: label, \
                                                self._targets: target})
                    warmupAccu += accu / 50
                print('\nWarmup Accuracy: ', warmupAccu)
#            
#            data, label, target = next(genTrain)
#            noise, adversary = self._sess.run([self._noises, self._adversary], \
#                                        feed_dict={self._images: data, \
#                                                    self._labels: label, \
#                                                    self._targets: target})
#            loss, accu, results = \
#                self._sess.run([self._lossPredictor, \
#                                self._accuracy, self._inference], \
#                                feed_dict={self._images: adversary, \
#                                            self._labels: label, \
#                                            self._targets: target})
            #print(label)
            #print(results)
            #print(loss)
            #print(noise[0, 2, 4:10, 0])
            #plt.subplot(131)
            #plt.imshow(np.clip(data[1], 0, 255)/255)
            #plt.subplot(132)
            #plt.imshow(np.clip(adversary[1], 0, 255)/255)
            #plt.subplot(133)
            #plt.imshow(np.clip(noise[1], 0, 255)/255)
            #plt.show()
            
            
            self.evaluate(genTest)
#             self.sample(genTest)
            
            self._sess.run([self._phaseTrain])
            if pathSave is not None:
                self.save(pathSave)
            
            globalStep = 0
            
            while globalStep < self._HParam['TotalSteps']: 
                
                self._sess.run(self._stepInc)
                
                for _ in range(self._HParam['NumPredictor']): 
                    #data = np.random.rand(self._HParam['BatchSize'], 32, 32, 3) * 255
                    #label = self._enemy.infer(data)
                    #loss, accu, globalStep, _ = \
                        #self._sess.run([self._lossPredictor, \
                                        #self._accuracy, self._step, self._optimizerP], \
                                        #feed_dict={self._images: data, \
                                                   #self._labels: label})
                    #print('\rPredictor => Step: ', globalStep, \
                                #'; Loss: %.3f'% loss, \
                                #'; Accu: %.3f'% accu, \
                                #end='')
                    data, label, target = next(genTrain)
                    adversary = self._sess.run(self._adversary, \
                                               feed_dict={self._images: data, \
                                                          self._labels: label, \
                                                          self._targets: target})
                    randNoise = (np.random.rand(self._HParam['BatchSize'], 32, 32, 3) - 0.5) * 2 * NoiseRange
                    data = data + randNoise
                    label = self._enemy.infer(data)
                    loss, accu, globalStep, _ = \
                        self._sess.run([self._lossPredictor, \
                                        self._accuracy, self._step, self._optimizerP], \
                                        feed_dict={self._images: data, \
                                                   self._labels: label})
                    print('\rPredictor => Step: ', globalStep, \
                                '; Loss: %.3f'% loss, \
                                '; Accuracy: %.3f'% accu, \
                                end='')
                    #label = self._enemy.infer(data)
                    #loss, accu, globalStep, _ = \
                        #self._sess.run([self._lossPredictor, \
                                        #self._accuracy, self._step, self._optimizerP], \
                                        #feed_dict={self._images: data, \
                                                   #self._labels: label, \
                                                   #self._targets: target})
                    #print('\rPredictor => Step: ', globalStep, \
                                #'; Loss: %.3f'% loss, \
                                #'; Accuracy: %.3f'% accu, \
                                #end='')
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
                            tmp = random.randint(0, 99)
                            while tmp == refs[idx]: 
                                tmp = random.randint(0, 99)
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
                    # print(np.max(adversary-data))
                    # print(np.min(adversary-data))
                    # print((adversary-data)[1])
                    # print(list(zip(label, refs, results, target)))
                    if pathSave is not None:
                        self.save(pathSave)
                    self._sess.run([self._phaseTrain])
                
                if (globalStep % 10500 == 0 or globalStep % 12000 == 0): 
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
                    tmp = random.randint(0, 99)
                    while tmp == refs[idx]: 
                        tmp = random.randint(0, 99)
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
        
    def plot(self, genTest, path=None): 
        if path is not None:
            self.load(path)
        
        accu = np.zeros([380])
        for time in range(1000): 
            data, label, target = next(genTest)
            
            tmpdata = []
            tmptarget = []
            
            for idx in range(20):
                while True: 
                    jdx = 0
                    while jdx < data.shape[0]:
                        if label[jdx] == idx: 
                            break
                        jdx += 1
                    if jdx < data.shape[0]: 
                        break
                    else: 
                        data, label, target = next(genTest)
                for ldx in range(20): 
                    if ldx != idx: 
                        tmpdata.append(data[jdx][np.newaxis, :, :, :])
                        tmptarget.append(ldx)
            tmpdata = np.concatenate(tmpdata, axis=0)
            tmptarget = np.array(tmptarget)
            
            adversary = \
            self._sess.run(self._adversary, \
                            feed_dict={self._images: tmpdata, \
                                        self._targets: tmptarget})
            adversary = adversary.clip(0, 255).astype(np.uint8)
            results = self._enemy.infer(adversary)
            
            accu += (results == tmptarget)
            print("\rround: ", time, end='')
        print("\n", accu / 1000)
                
        
#        kdx = 0
#        for idx in range(10):
#            jdx = 0
#            while jdx < 10: 
#                if jdx == idx: 
#                    jdx += 1
#                    continue
#                plt.subplot(10, 10, idx*10+jdx+1)
#                plt.imshow(adversary[kdx], cmap='gray')
#                plt.axis('off')
#                jdx += 1
#                kdx += 1
#                
#        plt.show()
    
    def sample(self, genTest, path=None):
        if path is not None:
            self.load(path)
            
        self._sess.run([self._phaseTest])  
        data, label, target = next(genTest)
        data, label, target = next(genTest)
        refs = self._enemy.infer(data)
        for idx in range(data.shape[0]):
            if refs[idx] == target[idx]: 
                tmp = random.randint(0, 99)
                while tmp == refs[idx]: 
                    tmp = random.randint(0, 99)
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
                plt.imshow(data[idx*3+jdx])
                plt.subplot(10, 6, idx*6+jdx*2+2)
                plt.imshow(adversary[idx*3+jdx])
                print([refs[idx*3+jdx], results[idx*3+jdx], target[idx*3+jdx]])
        plt.show()
        
    
    def save(self, path):
        self._saver.save(self._sess, path, global_step=self._step)
    
    def load(self, path):
        self._saver.restore(self._sess, path)

if __name__ == '__main__':
    enemy = CIFAR100.NetCIFAR100([32, 32, 3], 2)
    enemy.load('./ClassifyCIFAR100/netcifar100.ckpt-5400')
    net = NetCIFAR10([32, 32, 3], enemy=enemy, numMiddle=2) 
    batchTrain, batchTest = CIFAR100.generatorsAdv(BatchSize=HParamCIFAR10['BatchSize'], preprocSize=[32, 32, 3])
    # print(enemy.infer(next(batchTest)[0]))
    #net.load('./AttackCIFAR100/netcifar100.ckpt-1800')
    net.train(batchTrain, batchTest, pathSave='./AttackCIFAR100/netcifar100.ckpt')
    #net.plot(batchTest, './AttackCIFAR100/netcifar100.ckpt-16800')
