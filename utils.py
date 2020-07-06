import tensorflow as tf
import numpy as np

def MinMax_Scale(x, minv=0, maxv=1):
    # Prevent divide by zero
    x = x + 1e-5
    
    if len(x.get_shape()) == 3:
        min_x = tf.tile(tf.reduce_min(x, axis=[0,1], keepdims=True), tf.constant([x.get_shape()[0], x.get_shape()[1], 1]))
        max_x = tf.tile(tf.reduce_max(x, axis=[0,1], keepdims=True), tf.constant([x.get_shape()[0], x.get_shape()[1], 1]))

        nx = (x-min_x)/(max_x-min_x)

    elif len(x.get_shape()) == 4:
        min_x = tf.tile(tf.reduce_min(x, axis=[1,2], keepdims=True), tf.constant([1, x.get_shape()[1], x.get_shape()[2], 1]))
        max_x = tf.tile(tf.reduce_max(x, axis=[1,2], keepdims=True), tf.constant([1, x.get_shape()[1], x.get_shape()[2], 1]))

        nx = (x-min_x)/(max_x-min_x)
        
    return (maxv-minv)*nx+minv


'''
Below two processes are required because of the training stratagy of VGG19 net.
1. VGG19 was trained on images loaded as 'BGR' (channel order) from OpenCV.
2. Images used for VGG19 training are subtracted the pixel mean value of each channel across the ImageNet.
ref: https://github.com/elleryqueenhomels/arbitrary_style_transfer/issues/11
'''

def preprocessing(img):
    if len(img.get_shape()) == 3:
        # 1. Reverse channel order (RGB -> BGR)
        # 2. Normalize as the VGG19 trained on the ImageNet data.
        img = tf.keras.applications.vgg19.preprocess_input(img)

        return img
    
    elif len(img.get_shape()) == 4:
        for i,one_img in enumerate(img):
            pre_img = tf.expand_dims(preprocessing(one_img), axis=0)
            
            if i == 0: img_out = pre_img
            else: img_out = tf.concat([img_out, pre_img], axis=0)
        return img_out
    

def deprocessing(img):
    if len(img.get_shape()) == 3:
        # Reverse channel order (BGR -> RGB)
        img = tf.reverse(img, axis=[-1])
        img += tf.convert_to_tensor(np.array([103.939, 116.779, 123.68]), dtype=tf.float32)
        
        return img
    
    elif len(img.get_shape()) == 4:
        for i,one_img in enumerate(img):
            de_img = tf.expand_dims(deprocessing(one_img), axis=0)
            
            if i == 0: img_out = de_img
            else: img_out = tf.concat([img_out, de_img], axis=0)
        return img_out
    
    
    