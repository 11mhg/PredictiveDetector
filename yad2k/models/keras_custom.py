from keras import backend as K
from keras.regularizers import l2
from keras.layers import Conv2D, Activation, BatchNormalization, LeakyReLU, add

def residual_block(y, nb_channels, strides=(1,1), proj_shortcut=False):
    shortcut = y

    #down-sample
    y = Conv2D(nb_channels, kernel_size = (3,3), strides=strides,padding='same')(y)
    y = BatchNormalization()(y)
    y = LeakyReLU(alpha=0.1)(y)

    y = Conv2D(nb_channels, kernel_size=(3,3), strides=(1,1),padding='same')(y)
    y = BatchNormalization()(y)

    if proj_shortcut:
        shortcut = Conv2D(nb_channels, kernel_size=(1,1), strides=strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    y = add([shortcut,y])
    y = LeakyReLU(alpha=0.1)(y)
    
    return y

def conv_block(x,filters,kernels,strides=1):
    x = Conv2D(filters,kernels,
               padding='same',
               strides=strides,
               activation='linear',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def yolo3_residual(y,filters):
    x = conv_block(y,filters,(1,1))
    x = conv_block(x,2*filters,(3,3))
    x = add([y,x])
    x = Activation('linear')(x)
    return x

def stack_residual_block(y, nb_channels, n=1, proj_shortcut=False):
    y = yolo3_residual(y,nb_channels)
    
    for i in range(n-1):
        y = yolo3_residual(y,nb_channels)
    
    return y



class Subpixel(Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 r,
                 padding='valid',
                 data_format=None,
                 strides=(1,1),
                 activation=None,
                 use_bias = True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Subpixel,self).__init__(
                filters=r*r*filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                **kwargs)
        self.r = r

    def _phase_shift(self, I):
        r = self.r
        bsize, a, b, c=I.get_shape().as_list()
        bsize=K.shape(I)[0] #Batch Dimension
        X = K.reshape(I, [bsize, a, b, c/(r*r),r, r]) # bsize, a, b, c/(r*r), r, r
        X = K.permute_dimensions(X, (0, 1, 2, 5, 4, 3))  # bsize, a, b, r, r, c/(r*r)
        #Keras backend does not support tf.split, so in future versions this could be nicer
        X = [X[:,i,:,:,:,:] for i in range(a)] # a, [bsize, b, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, b, a*r, r, c/(r*r)
        X = [X[:,i,:,:,:] for i in range(b)] # b, [bsize, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, a*r, b*r, c/(r*r)
        return X

    def call(self, inputs):
        return self._phase_shift(super(Subpixel, self).call(inputs))

    def compute_output_shape(self, input_shape):
        unshifted = super(Subpixel, self).compute_output_shape(input_shape)
        return (unshifted[0], self.r*unshifted[1],self.r*unshifted[2], unshifted[3]/(self.r*self.r))

    def get_config(self):
        config=super(Conv2D, self).get_config()
        config.pop('rank')
        config.pop('dilation_rate')
        config['filters']/=self.r*self.r
        config['r'] = self.r


