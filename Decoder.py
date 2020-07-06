import tensorflow as tf
import utils

class decoder:
    def __init__(self, input_shape):
        self.net = self.build(input_shape)
    
    def build(self, input_shape):
        
        '''
        build the network along the reverse direction of VGG19 from conv4_1.
        -> input
        -> conv4_1 (shape: 32 x 32 x 512)
        -> upsampling (NN)
        -> conv3_4, conv3_3, conv3_2, conv3_1 (shape: 64 x 64 x 256)
        -> upsampling (NN)
        -> conv2_2, conv2_1 (shape: 128 x 128 x 128)
        -> upsampling (NN)
        -> conv1_2 (shape: 256 x 256 x 64),  conv1_1 (shape:256 x 256 x 3)
        
        '''
        # change of channel numbers
        chs = [512, 256, 256, 256, 256, 128, 128, 64, 3]
        
        # upsampling right after conv4_1, conv3_1, conv2_1
        up_sample = [0, 4, 6] 
        inputs = tf.keras.layers.Input(shape=input_shape, dtype=tf.float32)
        x = inputs
        
        for i, ch in enumerate(chs):
            x = self.conv_block(x, ch, 3)
            
            if i in up_sample:
                #x = tf.keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last',interpolation='nearest')(x) 
                _, H, W,_ = tf.shape(x)
                x = tf.image.resize(x, size=[H*2, W*2], method='nearest')
    
            if i == len(chs)-1:
                x = self.conv_block(x, 3, 3, use_relu=False)
            
        outputs = x
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def conv_block(self, x, out_channel, kernel_size, use_bias=True, use_relu=True):        
        # reflect padding
        x_temp = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], mode='REFLECT')
        
        # 2D convolution
        x_temp = tf.keras.layers.Conv2D(out_channel, (kernel_size, kernel_size), (1, 1), padding='valid', use_bias=use_bias)(x_temp)
        
        # LeakyRelu if use relu
        '''
        When I used the Relu, the output was black image,
        because Relu deactivates all pixels for some reasons.
        So, I changed it to LeakyReLU.
        '''
        if use_relu:
            x_temp = tf.keras.layers.LeakyReLU()(x_temp)
           
        return x_temp
    
    def forward(self, inputs):
        '''
        This is for inference phase.
        '''
        outputs = self.net(inputs)
        deprocessed_outputs = utils.deprocessing(outputs)
        
        return deprocessed_outputs
            
            
    
        
        
        