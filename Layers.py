import functools
import tensorflow as tf

from tensorflow.python.training.moving_averages import assign_moving_average

# Initializers
XavierInit = tf.contrib.layers.xavier_initializer()
Norm01Init = tf.truncated_normal_initializer(0.0, stddev=0.1)
def NormalInit(stddev, dtype=tf.float32):
    return tf.truncated_normal_initializer(0.0, stddev=stddev, dtype=dtype)
def ConstInit(const, dtype=tf.float32):
    return tf.constant_initializer(const, dtype=dtype)

# Activations
Linear  = tf.identity
Sigmoid = tf.nn.sigmoid
Tanh    = tf.nn.tanh
ReLU    = tf.nn.relu
ELU     = tf.nn.elu
Softmax = tf.nn.softmax
def LeakyReLU(alpha=0.2):
    return functools.partial(tf.nn.leaky_relu, alpha=alpha)

#Poolings
AvgPool        = tf.nn.avg_pool
MaxPool        = tf.nn.max_pool

class UnsupportedParam:
    pass

class Layer(object):
    
    def __init__(self):
        self._output    = None
        self._variables = []
        self._updateOps = []
        self._losses    = []
    
    @property
    def type(self):
        return 'Layer'
    
    @property
    def output(self):
        return self._output
    
    @property
    def variables(self):
        return self._variables
    
    @property
    def updateOps(self):
        return self._updateOps
    
    @property
    def losses(self):
        return self._losses
    
    @property
    def summary(self):
        return 'Layer: the parent class of all layers'

# Convolution

class Conv2D(Layer):
    
    def __init__(self, feature, convChannels, \
                 convKernel=[3, 3], convStride=[1, 1], convWD=None, convInit=XavierInit, convPadding='SAME', \
                 bias=True, biasInit=ConstInit(0.0), \
                 bn=False, step=None, ifTest=None, epsilon=1e-5, \
                 activation=Linear, \
                 pool=False, poolSize=[3, 3], poolStride=[2, 2], poolType=MaxPool, poolPadding='SAME', \
                 reuse=False, name=None, dtype=tf.float32): 
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'
        
        Layer.__init__(self); 
        self._name = name
        self._losses = []
        with tf.variable_scope(self._name, reuse=reuse) as scope: 
            self._sizeKernel      = convKernel + [feature.get_shape().as_list()[3], convChannels]
            self._strideConv      = [1]+convStride+[1]
            self._typeConvPadding = convPadding
            self._weights = tf.get_variable(scope.name+'_weights', \
                                            self._sizeKernel, initializer=convInit, dtype=dtype)
            conv = tf.nn.conv2d(feature, self._weights, self._strideConv, padding=self._typeConvPadding, \
                               name=scope.name+'_conv2d')
            self._variables.append(self._weights)
            if convWD is not None:
                decay = tf.multiply(tf.nn.l2_loss(self._weights), convWD, name=scope.name+'l2_wd')
                self._losses.append(decay)
        
            #tf.nn.conv2d(pooling, kernel, [1, 1, 1, 1], padding=Padding)
            if bias: 
                self._bias = tf.get_variable(scope.name+'_bias', [convChannels], \
                                                   initializer=biasInit, dtype=dtype)
                conv = conv + self._bias
                self._variables.append(self._bias)
            
            self._bn = bn
            if bn:
                assert (step is not None), "step parameter must not be None. "
                assert (ifTest is not None), "ifTest parameter must not be None. "
                shapeParams   = [conv.shape[-1]]
                self._offset  = tf.get_variable(scope.name+'_offset', \
                                                shapeParams, initializer=ConstInit(0.0), dtype=dtype)
                self._scale   = tf.get_variable(scope.name+'_scale', \
                                                shapeParams, initializer=ConstInit(1.0), dtype=dtype)
                self._movMean = tf.get_variable(scope.name+'_movMean', \
                                                shapeParams, trainable=False, initializer=ConstInit(0.0), dtype=dtype)
                self._movVar  = tf.get_variable(scope.name+'_movVar', \
                                                shapeParams, trainable=False, initializer=ConstInit(1.0), dtype=dtype)
                self._variables.append(self._scale)
                self._variables.append(self._offset)
                self._epsilon   = epsilon
                def trainMeanVar(): 
                    mean, var = tf.nn.moments(conv, list(range(len(conv.shape)-1)))
                    with tf.control_dependencies([assign_moving_average(self._movMean, mean, 0.9), \
                                                  assign_moving_average(self._movVar, var, 0.9)]): 
                        self._trainMean = tf.identity(mean)
                        self._trainVar  = tf.identity(var)
                    return self._trainMean, self._trainVar
                    
                self._actualMean, self._actualVar = tf.cond(ifTest, lambda: (self._movMean, self._movVar), trainMeanVar)
                conv = tf.nn.batch_normalization(conv, self._actualMean, self._actualVar, \
                                                         self._offset, self._scale, self._epsilon, \
                                                         name=scope.name+'_batch_normalization')
                
            self._activation = activation
            if activation is not None: 
                activated = activation(conv, name=scope.name+'_activation')
            else:
                activated = conv
            if pool:
                self._sizePooling     = [1]+poolSize+[1]
                self._stridePooling   = [1]+poolStride+[1]
                self._typePoolPadding = poolPadding
                pooled = poolType(activated, ksize=self._sizePooling, strides=self._stridePooling, padding=self._typePoolPadding, \
                                  name=scope.name+'_pooling')
            else:
                self._sizePooling     = [0]
                self._stridePooling   = [0]
                self._typePoolPadding = 'NONE'
                pooled = activated
        self._output = pooled
            
    @property
    def type(self):
        return 'Conv2D'
    
    @property
    def summary(self):
        if isinstance(self._activation, functools.partial): 
            activation = self._activation.func.__name__
        else:
            activation = self._activation.__name__
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'Kernel Size: ' + str(self._sizeKernel) + '; ' + \
                'Conv Stride: ' + str(self._strideConv) + '; '  + 'Conv Padding: ' + self._typeConvPadding + '; '  + \
                'Batch Normalization: ' + str(self._bn) + '; ' + \
                'Pooling Size: ' + str(self._sizePooling) + '; '  + 'Pooling Size: ' + str(self._stridePooling) + '; '  + \
                'Pooling Padding: ' + self._typePoolPadding + '; '  + 'Activation: ' + activation + ']')

class DeConv2D(Layer):
    
    def __init__(self, feature, convChannels, shapeOutput=None, \
                 convKernel=[3, 3], convStride=[1, 1], convWD=None, convInit=XavierInit, convPadding='SAME', \
                 bias=True, biasInit=ConstInit(0.0), \
                 bn=False, step=None, ifTest=None, epsilon=1e-5, \
                 activation=Linear, \
                 pool=False, poolSize=[3, 3], poolStride=[2, 2], poolType=MaxPool, poolPadding='SAME', \
                 reuse=False, name=None, dtype=tf.float32): 
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'
        
        Layer.__init__(self); 
        self._name = name
        self._losses = []
        with tf.variable_scope(self._name, reuse=reuse) as scope: 
            self._sizeKernel      = convKernel + [convChannels, feature.get_shape().as_list()[3]]
            self._strideConv      = [1]+convStride+[1]
            if shapeOutput is None: 
                self._shapeOutput = tf.TensorShape([feature.get_shape().as_list()[0], feature.get_shape().as_list()[1]*convStride[0], feature.get_shape().as_list()[2]*convStride[1], convChannels])
            else:
                self._shapeOutput = tf.TensorShape([feature.shape[0]] + shapeOutput + [convChannels])
            self._typeConvPadding = convPadding
            self._weights = tf.get_variable(scope.name+'_weights', \
                                            self._sizeKernel, initializer=convInit, dtype=dtype)
            conv = tf.nn.conv2d_transpose(feature, self._weights, self._shapeOutput, self._strideConv, padding=self._typeConvPadding, \
                               name=scope.name+'_conv2d_transpose')
            self._variables.append(self._weights)
            if convWD is not None:
                decay = tf.multiply(tf.nn.l2_loss(self._weights), convWD, name=scope.name+'l2_wd')
                self._losses.append(decay)
        
            #tf.nn.conv2d(pooling, kernel, [1, 1, 1, 1], padding=Padding)
            if bias: 
                self._bias = tf.get_variable(scope.name+'_bias', [convChannels], \
                                                   initializer=biasInit, dtype=dtype)
                conv = conv + self._bias
                self._variables.append(self._bias)
            
            self._bn = bn
            if bn:
                assert (step is not None), "step parameter must not be None. "
                assert (ifTest is not None), "ifTest parameter must not be None. "
                shapeParams   = [conv.shape[-1]]
                self._offset  = tf.get_variable(scope.name+'_offset', \
                                                shapeParams, initializer=ConstInit(0.0), dtype=dtype)
                self._scale   = tf.get_variable(scope.name+'_scale', \
                                                shapeParams, initializer=ConstInit(1.0), dtype=dtype)
                self._movMean = tf.get_variable(scope.name+'_movMean', \
                                                shapeParams, trainable=False, initializer=ConstInit(0.0), dtype=dtype)
                self._movVar  = tf.get_variable(scope.name+'_movVar', \
                                                shapeParams, trainable=False, initializer=ConstInit(1.0), dtype=dtype)
                self._variables.append(self._scale)
                self._variables.append(self._offset)
                self._epsilon   = epsilon
                def trainMeanVar(): 
                    mean, var = tf.nn.moments(conv, list(range(len(conv.shape)-1)))
                    with tf.control_dependencies([assign_moving_average(self._movMean, mean, 0.9), \
                                                  assign_moving_average(self._movVar, var, 0.9)]): 
                        self._trainMean = tf.identity(mean)
                        self._trainVar  = tf.identity(var)
                    return self._trainMean, self._trainVar
                    
                self._actualMean, self._actualVar = tf.cond(ifTest, lambda: (self._movMean, self._movVar), trainMeanVar)
                conv = tf.nn.batch_normalization(conv, self._actualMean, self._actualVar, \
                                                         self._offset, self._scale, self._epsilon, \
                                                         name=scope.name+'_batch_normalization')
                
            self._activation = activation
            if activation is not None: 
                activated = activation(conv, name=scope.name+'_activation')
            else:
                activated = conv
            if pool:
                self._sizePooling     = [1]+poolSize+[1]
                self._stridePooling   = [1]+poolStride+[1]
                self._typePoolPadding = poolPadding
                pooled = poolType(activated, ksize=self._sizePooling, strides=self._stridePooling, padding=self._typePoolPadding, \
                                  name=scope.name+'_pooling')
            else:
                self._sizePooling     = [0]
                self._stridePooling   = [0]
                self._typePoolPadding = 'NONE'
                pooled = activated
        self._output = pooled
            
    @property
    def type(self):
        return 'DeConv2D'
    
    @property
    def summary(self):
        if isinstance(self._activation, functools.partial): 
            activation = self._activation.func.__name__
        else:
            activation = self._activation.__name__
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'Kernel Size: ' + str(self._sizeKernel) + '; ' + \
                'Conv Stride: ' + str(self._strideConv) + '; '  + 'Conv Padding: ' + self._typeConvPadding + '; '  + \
                'Batch Normalization: ' + str(self._bn) + '; ' + \
                'Pooling Size: ' + str(self._sizePooling) + '; '  + 'Pooling Size: ' + str(self._stridePooling) + '; '  + \
                'Pooling Padding: ' + self._typePoolPadding + '; '  + 'Activation: ' + activation + ']')
    

class SepConv2D(Layer):
    
    def __init__(self, feature, convChannels, \
                 convKernel=[3, 3], convStride=[1, 1], convWD=None, convInit=XavierInit, convPadding='SAME', \
                 bias=True, biasInit=ConstInit(0.0), \
                 bn=False, step=None, ifTest=None, epsilon=1e-5, \
                 activation=Linear, \
                 pool=False, poolSize=[3, 3], poolStride=[2, 2], poolType=MaxPool, poolPadding='SAME', \
                 reuse=False, name=None, dtype=tf.float32): 
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'
        
        Layer.__init__(self); 
        self._name = name
        self._losses = []
        with tf.variable_scope(self._name, reuse=reuse) as scope: 
            self._sizeDepthKernel = convKernel + [feature.get_shape().as_list()[3], 1]
            self._sizePointKernel = [1, 1] + [feature.get_shape().as_list()[3], convChannels]
            self._strideConv      = [1]+convStride+[1]
            self._typeConvPadding = convPadding
            self._weightsDepth = tf.get_variable(scope.name+'_weightsDepth', \
                                                 self._sizeDepthKernel, initializer=convInit, dtype=dtype)
            self._weightsPoint = tf.get_variable(scope.name+'_weightsPoint', \
                                                 self._sizePointKernel, initializer=convInit, dtype=dtype)
            conv = tf.nn.separable_conv2d(feature, self._weightsDepth, self._weightsPoint, \
                                          strides=self._strideConv, padding=self._typeConvPadding, \
                                          name=scope.name+'_sep_conv')
            self._variables.append(self._weightsDepth)
            self._variables.append(self._weightsPoint)
            if convWD is not None:
                decay = tf.multiply(tf.nn.l2_loss(self._weightsDepth), convWD, name=scope.name+'l2_wd_depth')
                self._losses.append(decay)
                decay = tf.multiply(tf.nn.l2_loss(self._weightsPoint), convWD, name=scope.name+'l2_wd_point')
                self._losses.append(decay)
        
            #tf.nn.conv2d(pooling, kernel, [1, 1, 1, 1], padding=Padding)
            if bias: 
                self._bias = tf.get_variable(scope.name+'_bias', [convChannels], \
                                                   initializer=biasInit, dtype=dtype)
                conv = conv + self._bias
                self._variables.append(self._bias)
            
            self._bn = bn
            if bn:
                assert (step is not None), "step parameter must not be None. "
                assert (ifTest is not None), "ifTest parameter must not be None. "
                shapeParams   = [conv.shape[-1]]
                self._offset  = tf.get_variable(scope.name+'_offset', \
                                                shapeParams, initializer=ConstInit(0.0), dtype=dtype)
                self._scale   = tf.get_variable(scope.name+'_scale', \
                                                shapeParams, initializer=ConstInit(1.0), dtype=dtype)
                self._movMean = tf.get_variable(scope.name+'_movMean', \
                                                shapeParams, trainable=False, initializer=ConstInit(0.0), dtype=dtype)
                self._movVar  = tf.get_variable(scope.name+'_movVar', \
                                                shapeParams, trainable=False, initializer=ConstInit(1.0), dtype=dtype)
                self._variables.append(self._scale)
                self._variables.append(self._offset)
                self._epsilon   = epsilon
                def trainMeanVar(): 
                    mean, var = tf.nn.moments(conv, list(range(len(conv.shape)-1)))
                    with tf.control_dependencies([assign_moving_average(self._movMean, mean, 0.9), \
                                                  assign_moving_average(self._movVar, var, 0.9)]): 
                        self._trainMean = tf.identity(mean)
                        self._trainVar  = tf.identity(var)
                    return self._trainMean, self._trainVar
                    
                self._actualMean, self._actualVar = tf.cond(ifTest, lambda: (self._movMean, self._movVar), trainMeanVar)
                conv = tf.nn.batch_normalization(conv, self._actualMean, self._actualVar, \
                                                         self._offset, self._scale, self._epsilon, \
                                                         name=scope.name+'_batch_normalization')
                
            self._activation = activation
            if activation is not None: 
                activated = activation(conv, name=scope.name+'_activation')
            else:
                activated = conv
            if pool:
                self._sizePooling     = [1]+poolSize+[1]
                self._stridePooling   = [1]+poolStride+[1]
                self._typePoolPadding = poolPadding
                pooled = poolType(activated, ksize=self._sizePooling, strides=self._stridePooling, padding=self._typePoolPadding, \
                                  name=scope.name+'_pooling')
            else:
                self._sizePooling     = [0]
                self._stridePooling   = [0]
                self._typePoolPadding = 'NONE'
                pooled = activated
        self._output = pooled
            
    @property
    def type(self):
        return 'SepConv2D'
    
    @property
    def summary(self):
        if isinstance(self._activation, functools.partial): 
            activation = self._activation.func.__name__
        else:
            activation = self._activation.__name__
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'Kernel Size: ' + str(self._sizeDepthKernel) + ', ' + str(self._sizePointKernel) + \
                'Conv Stride: ' + str(self._strideConv) + '; '  + 'Conv Padding: ' + self._typeConvPadding + '; '  + \
                'Batch Normalization: ' + str(self._bn) + '; ' + \
                'Pooling Size: ' + str(self._sizePooling) + '; '  + 'Pooling Size: ' + str(self._stridePooling) + '; '  + \
                'Pooling Padding: ' + self._typePoolPadding + '; '  + 'Activation: ' + activation + ']')

class DepthwiseConv2D(Layer):
    
    def __init__(self, feature, convChannels, \
                 convKernel=[3, 3], convStride=[1, 1], convWD=None, convInit=XavierInit, convPadding='SAME', \
                 bias=True, biasInit=ConstInit(0.0), \
                 bn=False, step=None, ifTest=None, epsilon=1e-5, \
                 activation=Linear, \
                 pool=False, poolSize=[3, 3], poolStride=[2, 2], poolType=MaxPool, poolPadding='SAME', \
                 reuse=False, name=None, dtype=tf.float32): 
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'
        
        Layer.__init__(self); 
        self._name = name
        self._losses = []
        with tf.variable_scope(self._name, reuse=reuse) as scope: 
            self._sizeDepthKernel = convKernel + [feature.get_shape().as_list()[3], int(convChannels/feature.get_shape().as_list()[3])]
            self._strideConv      = [1]+convStride+[1]
            self._typeConvPadding = convPadding
            self._weightsDepth = tf.get_variable(scope.name+'_weightsDepth', \
                                                 self._sizeDepthKernel, initializer=convInit, dtype=dtype)
            conv = tf.nn.depthwise_conv2d(feature, self._weightsDepth, strides=self._strideConv, padding=self._typeConvPadding, 
                                          name=scope.name+'_depthwise_conv')
            self._variables.append(self._weightsDepth)
            if convWD is not None:
                decay = tf.multiply(tf.nn.l2_loss(self._weightsDepth), convWD, name=scope.name+'l2_wd_depth')
                self._losses.append(decay)
        
            #tf.nn.conv2d(pooling, kernel, [1, 1, 1, 1], padding=Padding)
            if bias: 
                self._bias = tf.get_variable(scope.name+'_bias', [convChannels], \
                                                   initializer=biasInit, dtype=dtype)
                conv = conv + self._bias
                self._variables.append(self._bias)
            
            self._bn = bn
            if bn:
                assert (step is not None), "step parameter must not be None. "
                assert (ifTest is not None), "ifTest parameter must not be None. "
                shapeParams   = [conv.shape[-1]]
                self._offset  = tf.get_variable(scope.name+'_offset', \
                                                shapeParams, initializer=ConstInit(0.0), dtype=dtype)
                self._scale   = tf.get_variable(scope.name+'_scale', \
                                                shapeParams, initializer=ConstInit(1.0), dtype=dtype)
                self._movMean = tf.get_variable(scope.name+'_movMean', \
                                                shapeParams, trainable=False, initializer=ConstInit(0.0), dtype=dtype)
                self._movVar  = tf.get_variable(scope.name+'_movVar', \
                                                shapeParams, trainable=False, initializer=ConstInit(1.0), dtype=dtype)
                self._variables.append(self._scale)
                self._variables.append(self._offset)
                self._epsilon   = epsilon
                def trainMeanVar(): 
                    mean, var = tf.nn.moments(conv, list(range(len(conv.shape)-1)))
                    with tf.control_dependencies([assign_moving_average(self._movMean, mean, 0.9), \
                                                  assign_moving_average(self._movVar, var, 0.9)]): 
                        self._trainMean = tf.identity(mean)
                        self._trainVar  = tf.identity(var)
                    return self._trainMean, self._trainVar
                    
                self._actualMean, self._actualVar = tf.cond(ifTest, lambda: (self._movMean, self._movVar), trainMeanVar)
                conv = tf.nn.batch_normalization(conv, self._actualMean, self._actualVar, \
                                                         self._offset, self._scale, self._epsilon, \
                                                         name=scope.name+'_batch_normalization')
                
            self._activation = activation
            if activation is not None: 
                activated = activation(conv, name=scope.name+'_activation')
            else:
                activated = conv
            if pool:
                self._sizePooling     = [1]+poolSize+[1]
                self._stridePooling   = [1]+poolStride+[1]
                self._typePoolPadding = poolPadding
                pooled = poolType(activated, ksize=self._sizePooling, strides=self._stridePooling, padding=self._typePoolPadding, \
                                  name=scope.name+'_pooling')
            else:
                self._sizePooling     = [0]
                self._stridePooling   = [0]
                self._typePoolPadding = 'NONE'
                pooled = activated
        self._output = pooled
            
    @property
    def type(self):
        return 'DepthwiseConv2D'
    
    @property
    def summary(self):
        if isinstance(self._activation, functools.partial): 
            activation = self._activation.func.__name__
        else:
            activation = self._activation.__name__
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'Kernel Size: ' + str(self._sizeDepthKernel) + ',; ' + \
                'Conv Stride: ' + str(self._strideConv) + '; '  + 'Conv Padding: ' + self._typeConvPadding + '; '  + \
                'Batch Normalization: ' + str(self._bn) + '; ' + \
                'Pooling Size: ' + str(self._sizePooling) + '; '  + 'Pooling Size: ' + str(self._stridePooling) + '; '  + \
                'Pooling Padding: ' + self._typePoolPadding + '; '  + 'Activation: ' + activation + ']')
    
# Normalizations

class LRNorm(Layer):
    
    def __init__(self, feature, depth_radius=5, bias=1, alpha=1, beta=0.5, reuse=False, name=None): 
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'
        
        Layer.__init__(self); 
        self._name        = name
        self._depthRadius = depth_radius
        self._bias        = bias
        self._alpha       = alpha
        self._beta        = beta
        with tf.variable_scope(self._name, reuse=reuse) as scope: 
            self._output = tf.nn.lrn(feature, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta, \
                                     name=scope.name)
            
    @property
    def type(self):
        return 'LRNorm'
    
    @property
    def summary(self):
        return (self.type + ': [Name: ' + self._name + 'Output Size: ' + str(self._output.shape) + '; '  + \
                '; ' + 'Depth Radius: ' + self._depthRadius + '; ' + \
                'Bias: ' + self._bias + '; ' + 'Alpha: ' + self._alpha + '; ' + 'Beta: ' + self._beta + ']')
        
class BatchNorm(Layer):
    
    def __init__(self, feature, step, ifTest, epsilon=1e-5, reuse=False, name=None, dtype=tf.float32): 
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'
        
        Layer.__init__(self); 
        self._name = name
        with tf.variable_scope(self._name, reuse=reuse) as scope: 
            shapeParams   = [feature.shape[-1]]
            self._offset  = tf.get_variable(scope.name+'_offset', \
                                            shapeParams, initializer=ConstInit(0.0), dtype=dtype)
            self._scale   = tf.get_variable(scope.name+'_scale', \
                                            shapeParams, initializer=ConstInit(1.0), dtype=dtype)
            self._movMean = tf.get_variable(scope.name+'_movMean', \
                                            shapeParams, trainable=False, initializer=ConstInit(0.0), dtype=dtype)
            self._movVar  = tf.get_variable(scope.name+'_movVar', \
                                            shapeParams, trainable=False, initializer=ConstInit(1.0), dtype=dtype)
            self._variables.append(self._scale)
            self._variables.append(self._offset)
            self._epsilon   = epsilon
            def trainMeanVar(): 
                mean, var = tf.nn.moments(feature, list(range(len(feature.shape)-1)))
                with tf.control_dependencies([assign_moving_average(self._movMean, mean, 0.9), \
                                              assign_moving_average(self._movVar, var, 0.9)]): 
                    self._trainMean = tf.identity(mean)
                    self._trainVar  = tf.identity(var)
                return self._trainMean, self._trainVar
                
            self._actualMean, self._actualVar = tf.cond(ifTest, lambda: (self._movMean, self._movVar), trainMeanVar)
            self._output = tf.nn.batch_normalization(feature, self._actualMean, self._actualVar, \
                                                     self._offset, self._scale, self._epsilon, \
                                                     name=scope.name+'_batch_normalization')
            
    @property
    def type(self):
        return 'BatchNorm'
    
    @property
    def summary(self):
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'Epsilon: ' + str(self._epsilon) + ']')
     
class Dropout(Layer): 
    
    def __init__(self, feature, ifTest, rateKeep=0.5, reuse=False, name=None): 
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'
        
        Layer.__init__(self); 
        self._name = name
        self._rateKeep = rateKeep
        with tf.variable_scope(self._name, reuse=reuse) as scope:
            self._keepProb = tf.Variable(rateKeep, trainable=False)
            def phaseTest(): 
                return tf.assign(self._keepProb, 1.0)
            def phaseTrain(): 
                return tf.assign(self._keepProb, rateKeep)
            with tf.control_dependencies([tf.cond(ifTest, phaseTest, phaseTrain)]): 
                self._output = tf.nn.dropout(feature, self._keepProb, name=scope.name+'_dropout')
    
    @property
    def type(self):
        return 'Dropout'
    
    @property
    def summary(self):
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'Keep Rate: ' + str(self._rateKeep) + ']')

# Fully Connected

class Flatten(Layer):
    def __init__(self, feature, name="Flatten"):
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'
        
        Layer.__init__(self); 
        self._name = name
        size = feature.shape[1]
        for elem in feature.shape[2:]: 
            size *= elem
        self._output = tf.reshape(feature, [-1, size])
        
    @property
    def type(self):
        return 'Flatten'
    
    @property
    def summary(self):
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + ']')
    
class FullyConnected(Layer):
    
    def __init__(self, feature, outputSize, weightInit=XavierInit, wd=None, \
                 bias=True, biasInit=ConstInit(0.0), \
                 activation=ReLU, \
                 reuse=False, name=None, dtype=tf.float32):
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'
        
        Layer.__init__(self); 
        self._name = name
        with tf.variable_scope(self._name, reuse=reuse) as scope: 
            self._sizeWeights = [feature.get_shape().as_list()[1], outputSize]
            self._weights = tf.get_variable(scope.name+'_weights', \
                                            self._sizeWeights, initializer=weightInit, dtype=dtype)
            self._variables.append(self._weights)
            if wd is not None:
                decay = tf.multiply(tf.nn.l2_loss(self._weights), wd, name=scope.name+'l2_wd')
                self._losses.append(decay)
            if bias: 
                self._bias = tf.get_variable(scope.name+'_bias', [outputSize], \
                                             initializer=biasInit, dtype=dtype)
                self._variables.append(self._bias)
            else:
                self._bias = tf.constant(0.0, dtype=dtype)
            
            self._output = tf.add(tf.matmul(feature, self._weights), self._bias, name=scope.name+'_fully_connected')
            self._activation = activation 
            if activation is not None:
                self._output = activation(self._output, name=scope.name+'_activation')
            
    @property
    def type(self):
        return 'FullyConnected'
    
    @property
    def summary(self):
        if isinstance(self._activation, functools.partial): 
            activation = self._activation.func.__name__
        else:
            activation = self._activation.__name__
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'Weight Size: ' + str(self._weights.shape) + '; ' + \
                'Bias Size: ' + str(self._bias.shape) + '; ' + \
                'Activation: ' + activation + ']')
        
class Activation(Layer):
    
    def __init__(self, feature, activation=Linear, reuse=False, name=None):
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'
        
        Layer.__init__(self); 
        self._name = name
        self._activation = activation
        with tf.variable_scope(self._name, reuse=reuse) as scope: 
            self._output = activation(feature, name=scope.name+'_activation')
            
    @property
    def type(self):
        return 'Activation'
    
    @property
    def summary(self): 
        if isinstance(self._activation, functools.partial): 
            activation = self._activation.func.__name__
        else:
            activation = self._activation.__name__
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'Activation: ' + activation + ']')
        
class Pooling(Layer):
    
    def __init__(self, feature, pool=MaxPool, size=[2, 2], stride=[2, 2], padding='SAME', reuse=False, name=None):
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'
        
        Layer.__init__(self); 
        self._name = name
        self._typePool = pool
        self._sizePooling     = [1]+size+[1]
        self._stridePooling   = [1]+stride+[1]
        self._typePoolPadding = padding
        with tf.variable_scope(self._name, reuse=reuse) as scope: 
            self._output = self._typePool(feature, ksize=self._sizePooling, strides=self._stridePooling, \
                                          padding=self._typePoolPadding, \
                                          name=scope.name+'_pooling')
            
    @property
    def type(self):
        return 'Pooling'
    
    @property
    def summary(self): 
        if isinstance(self._typePool, functools.partial): 
            pooltype = self._typePool.func.__name__
        else:
            pooltype = self._typePool.__name__
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'Type: ' + pooltype + ']')
        
class GlobalAvgPool(Layer):
    
    def __init__(self, feature, reuse=False, name=None):
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'
        
        Layer.__init__(self); 
        self._name = name
        with tf.variable_scope(self._name, reuse=reuse) as scope: 
            self._output = tf.reduce_mean(feature, [1, 2], name=scope.name+'_global_avg_pool')
        
    @property
    def type(self):
        return 'GlobalAvgPooling'
    
    @property
    def summary(self): 
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'Type: Global Average Pooling' + ']')
        
class CrossEntropy(Layer):
    
    def __init__(self, feature, labels, \
                 reuse=False, name=None):
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'
        
        Layer.__init__(self); 
        self._name = name
        with tf.variable_scope(self._name, reuse=reuse) as scope: 
            self._output = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=feature, \
                                                                          name=scope.name+'_cross_entropy') 
            self._output = tf.reduce_mean(self._output)
            self._losses.append(self._output)
            
    @property
    def type(self):
        return 'CrossEntropy'
    
    @property
    def summary(self): 
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'Activation: CrossEntropy' + ']')

class TripletLoss(Layer):
    def __init__(self, feature, numDiff=1, weightDiff=1.0, tao=-1.0,  reuse=False, name=None): 
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'
        
        Layer.__init__(self); 
        self._name = name
        self._numDiff = numDiff
        self._weightDiff = weightDiff
        with tf.variable_scope(self._name, reuse=reuse) as scope: 
            numPerGroup = 2 + numDiff
            reshaped = tf.reshape(feature, [-1, numPerGroup, feature.shape[1]])
            group = tf.split(reshaped, numPerGroup, axis=1)
            for idx in range(len(group)): 
                group[idx] = tf.squeeze(group[idx])
            self._lossPos = tf.norm(group[0] - group[1], axis=-1, name=scope.name+'_lossPos')
            lossDiff = 0.0
            for idx in range(2, 2+numDiff):  
                lossDiff += tf.norm(group[0] - group[idx], axis=-1)
            self._lossDiff = tf.identity(lossDiff, name=scope.name+'_lossDiff')
            self._output = tf.reduce_mean(tf.maximum(self._lossPos - self._weightDiff*self._lossDiff, tao), name=scope.name+'_multilet_loss')
            self._losses.append(self._output)
            
    @property
    def type(self):
        return 'TripletLoss'
    
    @property
    def summary(self): 
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'NumDiff: ' + str(self._numDiff) + '; ' + \
                'Activation: TripletLoss' + ']')

class MultiletLoss(Layer):
    def __init__(self, feature, numSame=1, numDiff=1, weightSame=1.0, weightDiff=3.0, reuse=False, name=None): 
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'
        
        Layer.__init__(self); 
        self._name = name
        self._numSame = numSame
        self._numDiff = numDiff
        self._weightSame = weightSame
        self._weightDiff = weightDiff
        with tf.variable_scope(self._name, reuse=reuse) as scope: 
            numPerGroup = 2 + numSame + numDiff
            reshaped = tf.reshape(feature, [-1, numPerGroup, feature.shape[1]])
            group  = tf.split(reshaped, numPerGroup, axis=1)
            for idx in range(len(group)): 
                group[idx] = tf.squeeze(group[idx])
            self._lossPos  = tf.norm(group[0] - group[1], axis=-1, name=scope.name+'_lossPos')
            lossSame = 0.0
            for idx in range(2, 2+numSame):  
                lossSame += tf.norm(group[0] - group[idx], axis=-1)
            self._lossSame = tf.identity(lossSame, name=scope.name+'_lossSame')
            lossDiff = 0.0
            for idx in range(2+numSame, 2+numSame+numDiff):  
                lossDiff += tf.norm(group[0] - group[idx], axis=-1)
            self._lossDiff = tf.identity(lossDiff, name=scope.name+'_lossDiff')
            self._output = tf.reduce_mean(self._lossPos - self._weightSame*self._lossSame - self._weightDiff*self._lossDiff, \
                                          name=scope.name+'_multilet_loss')
            self._losses.append(self._output)
            
    @property
    def type(self):
        return 'MultiletLoss'
    
    @property
    def summary(self): 
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'NumSame: ' + str(self._numSame) + '; ' + 'NumDiff: ' + str(self._numDiff) + '; ' + \
                'Activation: MultiletLoss' + ']')

class MultiletLossFinal(Layer):
    def __init__(self, feature, numSame=1, numDiff=1, weightSame=1.0, weightDiff=3.0, \
                tao1=-1.5, tao2=-4.0, tao3=-2.0, reuse=False, name=None): 
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'
        
        Layer.__init__(self); 
        self._name = name
        self._numSame = numSame
        self._numDiff = numDiff
        self._weightSame = weightSame
        self._weightDiff = weightDiff
        with tf.variable_scope(self._name, reuse=reuse) as scope: 
            numPerGroup = 2 + numSame + numDiff
            reshaped = tf.reshape(feature, [-1, numPerGroup, feature.shape[1]])
            group  = tf.split(reshaped, numPerGroup, axis=1)
            for idx in range(len(group)): 
                group[idx] = tf.squeeze(group[idx])
            self._lossPos  = tf.norm(group[0] - group[1], axis=-1, name=scope.name+'_lossPos')
            lossSame = 0.0
            for idx in range(2, 2+numSame):  
                lossSame += tf.norm(group[0] - group[idx], axis=-1)
            self._lossSame = tf.identity(lossSame, name=scope.name+'_lossSame')
            lossDiff = 0.0
            for idx in range(2+numSame, 2+numSame+numDiff):  
                lossDiff += tf.norm(group[0] - group[idx], axis=-1)
            self._lossDiff = tf.identity(lossDiff, name=scope.name+'_lossDiff')
            self._output = tf.reduce_mean(tf.maximum(0.5*self._lossPos - 1.5*self._lossSame, tao1) + \
                                          tf.maximum(0.5*self._lossPos - 2*self._lossDiff, tao2) + \
                                          tf.maximum(0.5*self._lossSame - self._lossDiff, tao3), \
                                          name=scope.name+'_multilet_loss')
            self._losses.append(self._output)
            
    @property
    def type(self):
        return 'MultiletLossFinal'
    
    @property
    def summary(self): 
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'NumSame: ' + str(self._numSame) + '; ' + 'NumDiff: ' + str(self._numDiff) + '; ' + \
                'Activation: MultiletLossFinal' + ']')

class MultiletLossTruncated2(Layer):
    def __init__(self, feature, numSame=1, numDiff=1, weightSame=1.0, weightDiff=3.0, \
                tao1=-1.0, tao2=-4.0, tao3=-5.0, reuse=False, name=None): 
#                 tao1=-2.0, tao2=-5.0, tao3=-2.0, name=None): 
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'
        
        Layer.__init__(self); 
        self._name = name
        self._numSame = numSame
        self._numDiff = numDiff
        self._weightSame = weightSame
        self._weightDiff = weightDiff
        with tf.variable_scope(self._name, reuse=reuse) as scope: 
            numPerGroup = 2 + numSame + numDiff
            reshaped = tf.reshape(feature, [-1, numPerGroup, feature.shape[1]])
            group  = tf.split(reshaped, numPerGroup, axis=1)
            for idx in range(len(group)): 
                group[idx] = tf.squeeze(group[idx])
            self._lossPos  = tf.norm(group[0] - group[1], axis=-1, name=scope.name+'_lossPos')
            lossSame = 0.0
            for idx in range(2, 2+numSame):  
                lossSame += tf.norm(group[0] - group[idx], axis=-1)
            self._lossSame = tf.identity(lossSame, name=scope.name+'_lossSame')
            lossDiff = 0.0
            for idx in range(2+numSame, 2+numSame+numDiff):  
                lossDiff += tf.norm(group[0] - group[idx], axis=-1)
            self._lossDiff = tf.identity(lossDiff, name=scope.name+'_lossDiff')
            self._output = tf.reduce_mean(tf.maximum(0.5*self._lossPos - 2.0*self._lossSame, -2.0) + \
                                          tf.maximum(self._lossSame - self._lossDiff, -1.0) + \
                                          tf.maximum(0.5*self._lossPos - 3.0*self._lossDiff, -5.0), \
                                          name=scope.name+'_multilet_loss')
#             self._output = tf.reduce_mean(tf.maximum(0.5*self._lossPos - 2*self._lossSame, tao1) + \
#                                           tf.maximum(0.5*self._lossPos - 3*self._lossDiff, tao2) + \
#                                           tf.maximum(self._lossSame - self._lossDiff, tao3), \
#                                           name=scope.name+'_multilet_loss')
            self._losses.append(self._output)
            
    @property
    def type(self):
        return 'MultiletLossFinal'
    
    @property
    def summary(self): 
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'NumSame: ' + str(self._numSame) + '; ' + 'NumDiff: ' + str(self._numDiff) + '; ' + \
                'Activation: MultiletLossFinal' + ']')

class TruncatedMultiletLoss(Layer):
    def __init__(self, feature, numSame=1, numDiff=1, weightSame=1.0, weightDiff=3.0, \
                 tao1=-1.0, tao2=-4.0, reuse=False, name=None): 
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'
        
        Layer.__init__(self); 
        self._name = name
        self._numSame = numSame
        self._numDiff = numDiff
        self._weightSame = weightSame
        self._weightDiff = weightDiff
        with tf.variable_scope(self._name, reuse=reuse) as scope: 
            numPerGroup = 2 + numSame + numDiff
            reshaped = tf.reshape(feature, [-1, numPerGroup, feature.shape[1]])
            group  = tf.split(reshaped, numPerGroup, axis=1)
            for idx in range(len(group)): 
                group[idx] = tf.squeeze(group[idx])
            self._lossPos  = tf.norm(group[0] - group[1], axis=-1, name=scope.name+'_lossPos')
            lossSame = 0.0
            for idx in range(2, 2+numSame):  
                lossSame += tf.norm(group[0] - group[idx], axis=-1)
            self._lossSame = tf.identity(lossSame, name=scope.name+'_lossSame')
            lossDiff = 0.0
            for idx in range(2+numSame, 2+numSame+numDiff):  
                lossDiff += tf.norm(group[0] - group[idx], axis=-1)
            self._lossDiff = tf.identity(lossDiff, name=scope.name+'_lossDiff')
            self._output = tf.reduce_mean(tf.maximum(0.5*self._lossPos - self._weightSame*self._lossSame, tao1) + \
                                          tf.maximum(0.5*self._lossPos - self._weightDiff*self._lossDiff, tao2), \
                                          name=scope.name+'_multilet_loss')
            self._losses.append(self._output)
            
    @property
    def type(self):
        return 'TruncatedMultiletLoss'
    
    @property
    def summary(self): 
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'NumSame: ' + str(self._numSame) + '; ' + 'NumDiff: ' + str(self._numDiff) + '; ' + \
                'Activation: MultiletLoss' + ']')

class MultiletAccu(Layer):
    def __init__(self, feature, numSame=1, numDiff=1, reuse=False, name=None): 
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'
        
        Layer.__init__(self); 
        self._name = name
        self._numSame = numSame
        self._numDiff = numDiff
        with tf.variable_scope(self._name, reuse=reuse) as scope: 
            numPerGroup = 2 + numSame + numDiff
            reshaped = tf.reshape(feature, [-1, numPerGroup, feature.shape[1]])
            group  = tf.split(reshaped, numPerGroup, axis=1)
            for idx in range(len(group)): 
                group[idx] = tf.squeeze(group[idx])
            self._lossPos  = tf.norm(group[0] - group[1], axis=-1)
            right = tf.greater_equal(self._lossPos, tf.zeros_like(self._lossPos))
            for idx in range(2, 2+numSame):  
                right = tf.logical_and(right, \
                                       tf.greater(tf.norm(group[0] - group[idx], axis=-1) - self._lossPos, \
                                                  tf.zeros_like(self._lossPos)))
            for idx in range(2+numSame, 2+numSame+numDiff):  
                right = tf.logical_and(right, \
                                       tf.greater(tf.norm(group[0] - group[idx], axis=-1) - self._lossPos, \
                                                  tf.zeros_like(self._lossPos)))
                for jdx in range(2, 2+numSame): 
                    right = tf.logical_and(right, \
                                           tf.greater(tf.norm(group[0] - group[idx], axis=-1), \
                                                      tf.norm(group[0] - group[jdx], axis=-1)))
            self._output = tf.reduce_mean(tf.cast(right, tf.float32), name=scope.name+'_accu')
            
    @property
    def type(self):
        return 'MultiletAccu'
    
    @property
    def summary(self): 
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'NumSame: ' + str(self._numSame) + '; ' + 'NumDiff: ' + str(self._numDiff) + '; ' + \
                'Activation: MultiletAccu' + ']')

class TripletAccu(Layer):
    def __init__(self, feature, numSame=1, numDiff=1, reuse=False, name=None): 
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'
        
        Layer.__init__(self); 
        self._name = name
        self._numSame = numSame
        self._numDiff = numDiff
        with tf.variable_scope(self._name, reuse=reuse) as scope: 
            numPerGroup = 2 + numSame + numDiff
            reshaped = tf.reshape(feature, [-1, numPerGroup, feature.shape[1]])
            group  = tf.split(reshaped, numPerGroup, axis=1)
            for idx in range(len(group)): 
                group[idx] = tf.squeeze(group[idx])
            self._lossPos  = tf.norm(group[0] - group[1], axis=-1)
            accum = 0.0
            for idx in range(2, 2+numSame):  
                accum = accum + tf.reduce_mean(tf.cast(tf.greater(tf.norm(group[0] - group[idx], axis=-1) - self._lossPos, \
                                                                  tf.zeros_like(self._lossPos)), tf.float32))
            for idx in range(2+numSame, 2+numSame+numDiff):  
                accum = accum + tf.reduce_mean(tf.cast(tf.greater(tf.norm(group[0] - group[idx], axis=-1) - self._lossPos, \
                                                                  tf.zeros_like(self._lossPos)), tf.float32))
            self._output = tf.identity(accum/(numSame + numDiff), name=scope.name+'_accu')
            
    @property
    def type(self):
        return 'TripletAccu'
    
    @property
    def summary(self): 
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' + \
                'NumSame: ' + str(self._numSame) + '; ' + 'NumDiff: ' + str(self._numDiff) + '; ' + \
                'Activation: TripletAccu' + ']')
