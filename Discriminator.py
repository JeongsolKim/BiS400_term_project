import tensorflow as tf

class discriminator:
    def __init__(self, args):
        self.img_size = args.img_size
                
        self.learning_rate = args.learning_rate 
        self.learning_rate_decay = args.learning_rate_decay
        self.decay_step = 2000
        
        self.continue_learn = args.continue_learn
        self.load_ckp_dir = args.load_ckp_dir
        
        # make a network
        '''
        The network is defined as a Sequential model.
        If we call the network with the input, it will give the output.
        ex. output = net(input)
        '''
        self.net = self.build()
        
         # Learning rate scheduler
        lr_scheduler = tf.keras.optimizers.schedules.InverseTimeDecay(self.learning_rate, self.decay_step, self.learning_rate_decay)
        self.optimizer = tf.keras.optimizers.Adam(lr_scheduler)
        
        # If continue_learn = True, load saved weights.
        # If continue_learn = False, initialize with random weights.
        if args.continue_learn:
            try:
                self.dec.net.load_weights(self.load_ckp_dir + '/discriminator/discriminator_ckpt')
                print(self.load_ckp_dir+'/discriminator/discriminator_ckpt'+' is loaded.')
            except:
                print('No checkpoints found.. Start to train with randomly initialized model.')
        
    def build(self):
        d_model = tf.keras.Sequential()
        d_model.add(tf.keras.layers.Conv2D(64, (5, 5), (2, 2), padding='same', input_shape=[self.img_size, self.img_size, 3]))
        d_model.add(tf.keras.layers.BatchNormalization())
        d_model.add(tf.keras.layers.ReLU())

        d_model.add(tf.keras.layers.Conv2D(128, (5, 5), (2, 2), padding='same'))
        d_model.add(tf.keras.layers.BatchNormalization())
        d_model.add(tf.keras.layers.ReLU())
        
        d_model.add(tf.keras.layers.Conv2D(256, (5, 5), (2, 2), padding='same'))
        d_model.add(tf.keras.layers.BatchNormalization())
        d_model.add(tf.keras.layers.ReLU())
        
        d_model.add(tf.keras.layers.Flatten())
        d_model.add(tf.keras.layers.Dense(1))
        
        return d_model
