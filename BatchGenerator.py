import tensorflow as tf
import random
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
import utils


class batchgenerator:
    def __init__(self, data_dir, batch_size, img_size, default_reshape_size=512):
        
        '''
        :param data_dir: data directory. It should contain 'train' and 'test' folder with appropriate images.
        :param batch_size: size of one batch set
        :param img_size: size of cropped image
        :param default_reshape_size: size for resizing the raw large image. default 512.
        '''
 
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.default_reshape_size = default_reshape_size
        
        # get all file names from the dataset dir
        self.content_img_names = [name for name in listdir(self.data_dir+'content') if isfile(join(data_dir+'content', name))]
        self.style_img_names = [name for name in listdir(self.data_dir+'style') if isfile(join(self.data_dir+'style', name))]
    
    def one_test_img(self, style_or_content, name='', crop=True):
        
        if name.endswith('.jpg') or name.endswith('.png'): file_name = name
        elif name != '': file_name =  name + '.jpg'
        else:
            if style_or_content == 'style':
                file_name = random.sample(self.style_img_names, 1)[0]
            elif style_or_content == 'content':
                file_name = random.sample(self.content_img_names, 1)[0]
            else:
                print("'style_or_content' should be either 'style' or 'content'.")
                raise NameError('Wrong parameter input for random_load_img.')
        
        data_dir = self.data_dir + '/' + style_or_content
        img_names = self.style_img_names if style_or_content=='style' else self.content_img_names
        
        # Image load
        try:
            # load
            img = np.array(Image.open(data_dir+'/'+file_name))
            img = tf.cast(tf.convert_to_tensor(img), tf.float32)
        # ignore wrong type of file.
        except ValueError:
            target_files.append(random.sample(img_names, 1)[0])

        # ignore 2D or 4D images.
        shape = img.get_shape()
        if len(shape) != 3 or shape[-1] != 3:
            target_files.append(random.sample(img_names, 1)[0])
        
        H,W,C = img.get_shape()
        # resize
        if min(H, W) != self.default_reshape_size:
            reH = self.default_reshape_size if H < W else int(H/W*self.default_reshape_size)
            reW = int(W/H*self.default_reshape_size) if H < W else self.default_reshape_size
            img = tf.image.resize(img, [reH, reW], preserve_aspect_ratio=True)
        
        # crop
        if crop:
            img = tf.image.random_crop(img, size=(self.img_size, self.img_size, C))

        return tf.convert_to_tensor(img)
        
    
    def next_batch(self, style_or_content):
        '''
        # load image files (.jpg) 'randomly' from the folder & return one batch. (for training)
        
        :param style_or_content: 'style' or 'content' (string) -> select image type
        :param output: (batch number, 256, 256, 3) tensor -> one batch
        
        1. Resize the image such that dimension of minimum size become 512 (preserve the aspect ratio).
        2. Crop 256x256 images from the resized image, randomly.
        3. Save to the list.
        4. Convert the list to tensor and return.
        '''
        
        if style_or_content == 'style':
            target_files = random.sample(self.style_img_names, self.batch_size)
        elif style_or_content == 'content':
            target_files = random.sample(self.content_img_names, self.batch_size)
        else:
            print("'style_or_content' should be either 'style' or 'content'.")
            raise NameError('Wrong parameter input for random_load_img.')
        
        data_dir = self.data_dir + '/' + style_or_content
        img_names = self.style_img_names if style_or_content=='style' else self.content_img_names
        
        # cropped image list
        crop_imgs = []

        for file_name in target_files:
            try:
                # load
                img = np.array(Image.open(data_dir+'/'+file_name))
                img = tf.cast(tf.convert_to_tensor(img), tf.float32)
                
            # ignore wrong type of file.
            except ValueError:
                target_files.append(random.sample(img_names, 1)[0])
                continue

            # ignore 2D or 4D images.
            shape = img.get_shape()
            if len(shape) != 3 or shape[-1] != 3:
                target_files.append(random.sample(img_names, 1)[0])
                continue
                
            H, W, C = shape

            # resize
            if min(H, W) != self.default_reshape_size:
                reH = self.default_reshape_size if H < W else int(H/W*self.default_reshape_size)
                reW = int(W/H*self.default_reshape_size) if H < W else self.default_reshape_size
                img = tf.image.resize(img, [reH, reW], preserve_aspect_ratio=True)
            
            # crop
            cropped = tf.image.random_crop(img, size=(self.img_size, self.img_size, C))
            
            #save to the list
            crop_imgs.append(cropped)
            
        # convert to tensor
        crop_imgs = tf.convert_to_tensor(crop_imgs)

        return crop_imgs