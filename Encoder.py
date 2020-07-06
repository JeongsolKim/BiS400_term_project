import tensorflow as tf
import utils

class encoder:
    def __init__(self, input_size, style_layers, content_layer):
        '''
        :param input_size: tuple (H, W, C)
        :param style_layers: list (e.g. [relu1_1, relu2_1])
        :param content_layer: string (e.g. relu4_1)
        '''

        self.input_size = input_size
        self.layers = style_layers + [content_layer]
        self.net = self.build(self.layers)

    def build(self, layers):
        
        '''
        ref: https://www.tensorflow.org/tutorials/generative/style_transfer
        :param layers: list of layers' name for style and content features.
        '''

        # load pre-trained VGG19 network.
        VGG19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet')#,input_shape=self.input_size)
        # fix the VGG19 network. 
        VGG19.trainable = False
        
        # modify the output of the network.
        # outputs[:-2]: used for style features. outputs[-1]: used for content feature.
        outputs = [VGG19.get_layer(name).output for name in layers]

        return tf.keras.Model(inputs=[VGG19.input], outputs=outputs)
    
    
    def forward(self, inputs):
        '''
        This is for inference phase.
        '''
        processed_inputs = utils.preprocessing(inputs)
        
        return self.net(processed_inputs)
        