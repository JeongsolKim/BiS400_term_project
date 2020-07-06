import tensorflow as tf
from Encoder import encoder
from Decoder import decoder
import utils

class styletransfer:
    def __init__(self, args):
        self.img_size = args.img_size
        self.style_layers = args.style_layers
        self.content_layer = args.content_layer
        
        self.learning_rate = args.learning_rate 
        self.learning_rate_decay = args.learning_rate_decay
        self.decay_step = 2000
        
        self.batch_size = args.batch_size
        self.Ws = args.style_loss_weight
        
        self.continue_learn = args.continue_learn
        self.load_ckp_dir = args.load_ckp_dir
        
        self.build(self.style_layers, self.content_layer, self.continue_learn)
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        
    def __call__(self, content_img, style_img, alpha=1):
        '''
        # function for inference.
        Here, I used .forward function that contains preprocessing and deprocessing.
        
        :param content_img: input content iamge.
        :param style_img: input style_image.
        :output: synthesized image, shape=(batch_size, img_size, img_size, 3).
        '''
        # Get feature vectors of input images from VGG19 network (i.e. Encoding)
        content_feat = self.enc.forward(content_img)[-1]
        style_feat = self.enc.forward(style_img)[0:-1]

        # Adaptive Instance normalization with mean and var of style feature vector.
        aligned_content_feat = self.AdaIN(content_feat, style_feat[-1])
        
        # Content - Style tradeoff
        aligned_content_feat = (1-alpha)*content_feat + alpha*aligned_content_feat
        
        # Generate image from aligned feature (i.e. Decoding)
        synthesis_img = self.dec.forward(aligned_content_feat)

        # Clip [0..255]
        synthesis_img = tf.clip_by_value(synthesis_img, 0, 255)
        
        return synthesis_img
        
        
    def build(self, style_layers, content_layer, continue_learn):
        '''
        Initialize the style transfer.
        '''
        self.enc = encoder((self.img_size, self.img_size, 3), style_layers, content_layer)
        self.dec = decoder(self.enc.net.layers[-1].output.shape[1:])
        
        if continue_learn:
            try:
                self.dec.net.load_weights(self.load_ckp_dir + '/decoder/decoder_ckpt')
                print(self.load_ckp_dir+'/decoder/decoder_ckpt'+' is loaded.')
            except:
                print('No checkpoints found.. Start to train with randomly initialized model.')
        
        # Learning rate scheduler
        lr_scheduler = tf.keras.optimizers.schedules.InverseTimeDecay(self.learning_rate, self.decay_step, self.learning_rate_decay)
        self.optimizer = tf.keras.optimizers.Adam(lr_scheduler)
        
   
    def content_loss(self, target_feature, output_feature):
        '''
        :param target_feature: output feature from AdaIN
        :param output_feature: output of VGG19 conv4_1 layer with synthesized image (output of decoder) as an input.
        :output: content loss (scalar)
        :function: Enforce the decoder network to generate appropriate image from the given feature vector.
        '''
        return tf.reduce_sum(tf.reduce_mean(tf.square(target_feature - output_feature), [1, 2]))       
    
    
    def style_loss(self, target_styles, output_styles):
        '''
        :param target_styles: list of feature vectors of input style iamge, [from conv1_1, conv2_1, conv3_1, conv4_1]
        :param output_styles: list of feature vectors of synthesized image, [from conv1_1, conv2_1, conv3_1, conv4_1]
        :output: style loss (scalar)
        :function: Enforce the decoder network to generate the synthesized image that have the same statistics of style features with style input image.
        '''
        
        style_loss = 0
        for target, output in zip(target_styles, output_styles):
            target_mean, target_var = tf.nn.moments(target, axes=[1,2], keepdims=True)
            output_mean, output_var = tf.nn.moments(output, axes=[1,2], keepdims=True)
            
            target_sigma = tf.sqrt(target_var + 1e-6)
            output_sigma = tf.sqrt(output_var + 1e-6)
            
            
            loss = tf.reduce_sum(tf.reduce_mean(tf.square(target_mean - output_mean), [1, 2])) + \
                    tf.reduce_sum(tf.reduce_mean(tf.square(target_sigma - output_sigma), [1, 2]))
            
            style_loss += loss
        
        return style_loss
    
    
    def total_variation_loss(self, image):
        # get high frequency components along x and y direction
        x_deltas = image[:,:,1:,:] - image[:,:,:-1,:]
        y_deltas = image[:,1:,:,:] - image[:,:-1,:,:]
        
        return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))

    
    def AdaIN(self, content_feature, style_feature):
        '''
        :param style_feature: feature vector of style image, shape=(batch_size, 64, 64, 512)
        :param content_feature: feature vector of content image, shape=(batch_size, 64, 64, 512)
        :output: aligned content feature, shape=(batch_size, 64, 64, 512)
        '''
        
        # Calculate mean and variacne of style/content feature vectors.
        style_mean, style_var = tf.nn.moments(style_feature, axes=[1,2], keepdims=True)
        content_mean, content_var = tf.nn.moments(content_feature, axes=[1,2], keepdims=True)
        
        style_sigma = tf.sqrt(style_var+1e-6)
        content_sigma = tf.sqrt(content_var+1e-6)
        
        # Align content feature to style feature (RESCALE and SHIFT): using broadcasting.
        # to avoid zero-divide, add very small value to content variance.
        normalized_feature = (content_feature-content_mean)/(content_sigma)
        aligned_feature = normalized_feature*style_sigma + style_mean
        
        return aligned_feature
        
        
