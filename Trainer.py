import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import os
import imageio
import glob
from tqdm import tqdm
import utils

class trainer():
    def __init__(self, styletransfer, args):
        
        self.max_iteration = args.max_iteration
        self.epochs = args.epochs
        
        self.Ws = args.style_loss_weight
        self.Wc = args.content_loss_weight
        self.Wtv = args.tv_loss_weight
        self.Wg = args.gradient_panelty_weight
        
        self.sf = styletransfer

        self.log_dir = args.log_dir
        self.save_ckp_dir = args.save_ckp_dir
        self.train_closs_writer = tf.summary.create_file_writer(self.log_dir + '/train/content_loss')
        self.train_sloss_writer = tf.summary.create_file_writer(self.log_dir + '/train/style_loss')

        self.history_log = args.history_dir
    
    @tf.function
    def train_step(self, content_img, style_img):
        '''
        :param content_img: content image batch, shape = (batch_size, H, W, 3), already pre-processed
        :param style_img: style imaage batch, shape = (batch_size, H, W, 3), already pre-processed
        :output: two loss values, (scalar)
        '''
        with tf.GradientTape() as grad_tape:

            # Get feature vectors of input images from VGG19 network (i.e. Encoding)
            content_feat = self.sf.enc.net(content_img)[-1]
            style_feat = self.sf.enc.net(style_img)[0:-1]
            
            # Adaptive Instance normalization with mean and var of style feature vector.
            aligned_content_feat = self.sf.AdaIN(content_feat, style_feat[-1])
            
            # Generate image from aligned feature (i.e. Decoding)
            synthesis_img = self.sf.dec.net(aligned_content_feat)

            # Get feature vectors of synthesized from VGG19 network.
            output_style_feat = self.sf.enc.net(synthesis_img)            
            
            # Calculate losses     
            content_loss = self.sf.content_loss(aligned_content_feat, output_style_feat[-1]) if self.Wc != 0 else 0
            style_loss = self.sf.style_loss(style_feat, output_style_feat[0:-1]) if self.Ws != 0 else 0

            # Is this right?
            tv_loss = self.sf.total_variation_loss(synthesis_img) if self.Wtv != 0 else 0
            
            total_loss = self.Wc * content_loss + self.Ws * style_loss + self.Wtv * tv_loss
                    
        # Calculate gradient for only decoder & apply to optimizer
        gradient = grad_tape.gradient(total_loss, self.sf.dec.net.trainable_variables)
        self.sf.optimizer.apply_gradients(zip(gradient, self.sf.dec.net.trainable_variables))
        
        return content_loss, style_loss
    
    def train(self, batchgen):
        # prepare arbitrary dataset for saving image.
        content_for_history = batchgen.next_batch('content')[0:4,:,:,:]
        style_for_history = batchgen.next_batch('style')[0:4,:,:,:]
        
        print('>> Training Start.')
        
        for epoch in range(self.epochs):
            print('\n>> Epoch {} [{}]'.format(epoch+1,datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            avg_content_loss = []
            avg_style_loss = []
            
            # For each epoch, several training iteration is done.
            for it in tqdm(range(self.max_iteration)):
                
                # Get the next batch set.
                content_batch = utils.preprocessing(batchgen.next_batch('content'))
                style_batch = utils.preprocessing(batchgen.next_batch('style'))
                    
                # One iteration.
                content_loss, style_loss = self.train_step(content_batch, style_batch)

                avg_content_loss.append(content_loss)
                avg_style_loss.append(style_loss)
                
                if (it+1) % int((self.max_iteration/4)) == 0:
                    # Save the output as images
                    self.generate_and_save_images(epoch+1, (it+1)//int((self.max_iteration/4)), content_for_history, style_for_history)
            

            # Calculate the average losses for the current epoch.
            avg_content_loss = tf.reduce_mean(avg_content_loss)
            avg_style_loss = tf.reduce_mean(avg_style_loss)

            print('>> Avg Content Loss: {}'.format(avg_content_loss))
            print('>> Avg Style Loss: {}'.format(avg_style_loss))
            
            # Write on the tensorboard
            self.make_summaries(avg_content_loss, avg_style_loss, epoch+1)
                     
        print('\n>>Training Done.')
        
    def make_summaries(self, content_loss, style_loss, epoch):
        with self.train_closs_writer.as_default():
            tf.summary.scalar('Content_loss', content_loss, step=epoch)
        with self.train_sloss_writer.as_default():
            tf.summary.scalar('Style_loss', style_loss, step=epoch)

    def generate_and_save_images(self, epoch, num, content_img, style_img):
            
            # this output is deprocessed img.
            output = self.sf(content_img, style_img)
            
            # scaling for display.
            output = utils.MinMax_Scale(output)
            content_img = utils.MinMax_Scale(content_img)
            style_img = utils.MinMax_Scale(style_img)

            plt.figure(figsize=(10, 10))
            for i in range(4):
                plt.subplot(4, 3, i*3 + 1)
                plt.imshow(content_img[i, :, :, :], interpolation='nearest')
                plt.axis('off')
                if i == 0: plt.title('Content')
                    
                plt.subplot(4, 3, i*3 + 2)
                plt.imshow(style_img[i, :, :, :], interpolation='nearest')
                plt.axis('off')
                if i == 0: plt.title('Style')
                    
                plt.subplot(4, 3, i*3 + 3)
                plt.imshow(output[i, :, :, :], interpolation='nearest')
                plt.axis('off')
                if i == 0: plt.title('Output')

            plt.tight_layout()
            plt.savefig(self.history_log + '/image_at_epoch_{:04d}_{}.png'.format(epoch, num))
            plt.close()                

    def make_train_history_gif(self):
        anim_file = self.history_log + '/Style_transfer.gif'

        with imageio.get_writer(anim_file, mode='I') as writer:
            filenames = glob.glob(self.history_log + '/image*.png')
            filenames = sorted(filenames)

            last = -1
            for i, filename in enumerate(filenames):
                image = imageio.imread(filename)
                writer.append_data(image)
            image = imageio.imread(filename)
            writer.append_data(image)
