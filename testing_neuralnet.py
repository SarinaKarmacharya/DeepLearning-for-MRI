#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 09:41:21 2017

@author: sk834
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 19:17:20 2017

@author: sk834
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import tensorflow as tf
import matplotlib as plt

def print_tensor_shape(tensor, string):

# input: tensor and string to describe it

    if __debug__:
        print('DEBUG ' + string, tensor.get_shape())
        
        
def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'img_raw': tf.FixedLenFeature([], tf.string),
            'label_raw': tf.FixedLenFeature([], tf.string),
        })


    image = tf.decode_raw(features['img_raw'], tf.int64)
    image.set_shape([10223616])
    image_re = tf.reshape(image, (256,256,156))

    image_re = tf.cast(image_re, tf.float32) * (1. / 1024)
    label = tf.decode_raw(features['label_raw'], tf.uint8)
    label.set_shape([10223616])
    label_re = tf.reshape(label, [256,256, 156])

    return image_re, label_re
    
def inputs(batch_size, num_epochs, filename):

    if not num_epochs: num_epochs = None

    with tf.name_scope('input'):

        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=num_epochs)


        image, label = read_and_decode(filename_queue)
     

        images, sparse_labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=2,
            capacity=15,
            min_after_dequeue = 10)


        return images, sparse_labels
    
def inference(input):
    print_tensor_shape(images, 'images shape inference')

    images_re=tf.reshape(images, [-1,256,256,156,1])
    print_tensor_shape(images_re, 'images_re shape')
#convolution 1 
   
    with tf.name_scope('Conv1'):
        W_conv1=tf.Variable(tf.truncated_normal([3,3,3,1,16], stddev=0.1,
                                                dtype=tf.float32), name='W_conv1')
        print_tensor_shape(W_conv1,  'W_conv1 shape')
        
        conv1_op=tf.nn.conv3d(images_re, W_conv1, strides=[1,1,1,1,1], 
                              padding="VALID", name='conv1_op')
        print_tensor_shape(conv1_op, 'conv1_op shape')
        
        relu1_op=tf.nn.relu(conv1_op, name='relu_op')
        print_tensor_shape(relu1_op, 'relu1_op shape')
        
    with tf.name_scope('Conv2'):
       W_conv2=tf.Variable(tf.truncated_normal([3,3,3,16,16], stddev=0.1,
                                        dtype=tf.float32), name='W_conv2')
       print_tensor_shape(W_conv2, 'W_conv2 shape') 
       
       conv2_op=tf.nn.conv3d(relu1_op, W_conv2, strides=[1,1,1,1,1],
                             padding="VALID", name='conv2_op')
       print_tensor_shape(conv2_op, 'conv2_op shape')
       
       relu2_op=tf.nn.relu(conv2_op, name='relu2_op')
       print_tensor_shape(relu2_op, 'relu2_op shape')
#pool #1
    with tf.name_scope('Pool1'):  
        
       pool1_op=tf.nn.max_pool3d(relu2_op, ksize=[1,2,2,2,1], 
                                 strides=[1,2,2,2,1], padding="VALID")
       print_tensor_shape(pool1_op, 'pool1_op shape')
       
#convolution 2
       
    with tf.name_scope('conv3'):   
        
       W_conv3=tf.Variable(tf.truncated_normal([3,3,3,16,32], stddev=0.1,
                                               dtype=tf.float32), name='W_conv3')
       print_tensor_shape(W_conv3, 'W_conv3 shape')
       conv3_op=tf.nn.conv3d(pool1_op, W_conv3, strides=[1,1,1,1,1],
                             padding="VALID", name='conv3_op')
       relu3_op=tf.nn.relu(conv3_op, name='relu3_op')
       print_tensor_shape(relu3_op, 'relu3_op shape')
    with tf.name_scope('conv4'):
        
        W_conv4=tf.Variable(tf.truncated_normal([3,3,3,32,32],stddev=0.1,
                                                dtype=tf.float32), name='W_conv4')
        print_tensor_shape(W_conv4, 'W_conv4 shape')
        conv4_op=tf.nn.conv3d(relu3_op, W_conv4, strides=[1,1,1,1,1],
                              padding="VALID", name='conv4_op')
        print_tensor_shape(conv4_op, 'conv4_op shape')
        relu4_op=tf.nn.relu(conv4_op, 'relu4_op')
        print_tensor_shape(relu4_op, 'relu4_op shape')
        
#pool 2
        
    with tf.name_scope('pool2'):    
        
        pool2_op=tf.nn.max_pool3d(relu4_op, ksize=[1,2,2,2,1],
                                  strides=[1,2,2,2,1], padding="VALID")
        print_tensor_shape(pool2_op, 'pool2_op shape')
        
#convolution 3
        
    with tf.name_scope('conv5'):
        
        W_conv5=tf.Variable(tf.truncated_normal([3,3,3,32,64],stddev=0.1,
                                                    dtype=tf.float32), name='W_conv5')
        print_tensor_shape(W_conv5, 'W_conv5 shape')
        conv5_op=tf.nn.conv3d(pool2_op, W_conv5, strides=[1,1,1,1,1],
                              padding="VALID", name='conv5_op')
        print_tensor_shape(conv5_op, 'conv5_op shape')
        relu5_op=tf.nn.relu(conv5_op, 'relu5_op')
        print_tensor_shape(relu5_op, 'relu5_op shape')
    with tf.name_scope('conv6'):
        
        W_conv6=tf.Variable(tf.truncated_normal([3,3,3,64,64], stddev=0.1,
                               dtype=tf.float32), name='W_conv6')
        print_tensor_shape(W_conv6, 'W_conv6 shape')
        conv6_op=tf.nn.conv3d(relu5_op, W_conv6, strides=[1,1,1,1,1],
                              padding="VALID", name='conv6_op')
        print_tensor_shape(conv6_op, 'conv6 shape')
        relu6_op=tf.nn.relu(conv6_op, 'relu6_op')
        print_tensor_shape(relu6_op, 'relu6_op shape')
        
 #pool 3
        
    with tf.name_scope('pool3'):
        pool3_op=tf.nn.max_pool3d(relu6_op, ksize=[1,2,2,2,1],
                                  strides=[1,2,2,2,1], padding="VALID")
        print_tensor_shape(pool3_op, 'pool3_op shape')
        
#convolution 4
        
    with tf.name_scope('conv7'):
        
        W_conv7=tf.Variable(tf.truncated_normal([3,3,3,64, 128], stddev=0.1,
                                                dtype=tf.float32), name='W_conv7')
        print_tensor_shape(W_conv7, 'W_conv7 shape')
        conv7_op=tf.nn.conv3d(pool3_op, W_conv7, strides=[1,1,1,1,1], 
                              padding="VALID", name='conv7_op')
        print_tensor_shape(conv7_op, 'conv7_op shape')
        relu7_op=tf.nn.relu(conv7_op, 'relu7_op')        
        print_tensor_shape(relu7_op, 'relu7_op shape')
        
    with tf.name_scope('conv8'):
    
        W_conv8=tf.Variable(tf.truncated_normal([3,3,3,128, 128], stddev=0.1,
                                                dtype=tf.float32), name='W_conv8')
        conv8_op=tf.nn.conv3d(relu7_op, W_conv8, strides=[1,1,1,1,1],
                              padding="VALID", name='conv8_op')
        print_tensor_shape(conv8_op, 'conv8_op shape')
        relu8_op=tf.nn.relu(conv8_op, 'relu8_op')
        print_tensor_shape(relu8_op, 'relu8_op shape')
#max pool 
    with tf.name_scope('pool4'):
        
        pool4_op=tf.nn.max_pool3d(relu8_op, ksize=[1,2,2,2,1], 
                                  strides=[1,2,2,2,1], padding="VALID")
        print_tensor_shape(pool4_op, 'pool4_op shape')
        
#convolution 5 this is the final downsampling 
       
    with tf.name_scope('downconv1'):
         W_downconv1=tf.Variable(tf.truncated_normal([3,3,3,128, 256], stddev=0.1,
                                                 dtype=tf.float32), name='W_downconv1')
         downconv1_op=tf.nn.conv3d(pool4_op, W_downconv1, strides=[1,1,1,1,1],
                               padding="VALID", name='downconv1_op')
         print_tensor_shape(downconv1_op, 'downconv1_op shape')
        
         downrelu1_op=tf.nn.relu(downconv1_op, 'downconv1_op')
         print_tensor_shape(downrelu1_op, 'downrelu1_op shape')
    with tf.name_scope('downconv2'):
        W_downconv2=tf.Variable(tf.truncated_normal([3,3,3,256,256], stddev=0.1, 
                                                 dtype=tf.float32), name='W_downconv2')
        downconv2_op=tf.nn.conv3d(downrelu1_op, W_downconv2, strides=[1,1,1,1,1], 
                               padding="VALID", name='downconv2_op')
        print_tensor_shape(downconv2_op, 'downconv2_op shape')
        downrelu2_op=tf.nn.relu(downconv2_op, 'downconv2_op')
        print_tensor_shape(downrelu2_op, 'downrelu2_op shape')
              
 # this is the end of down smapling; now it is upsampling with merging of convolutional layer with the pervious images that were downsampled

    with tf.name_scope('upconv1'):
#upsampling1 
        W_upconv1=tf.Variable(tf.truncated_normal([2,2,2,256,128],stddev=0.1,
                                                dtype=tf.float32), name='W_upcon1')
        print_tensor_shape(W_upconv1, 'W_upconv1 shape')
        upconv1_op=tf.nn.conv3d(downrelu2_op, W_upconv1, strides=[1,1,1,1,1],
                             padding="VALID", name='upconv1_op')
        uprelu1_op=tf.nn.relu(upconv1_op, 'uprelu1_op')
        print_tensor_shape(uprelu1_op, 'uprelu1_op shape')
        # deconvolution
        W_uptrans=tf.Variable(tf.truncated_normal([2,2,2,128,128],stddev=0.1,
                                                dtype=tf.float32), name='W_uptrans')
        x_shape0=int(uprelu1_op.get_shape()[0])
        x_shape1=int(uprelu1_op.get_shape()[1])
        x_shape2=int(uprelu1_op.get_shape()[2])
        x_shape3=int(uprelu1_op.get_shape()[3])
        x_shape4=int(uprelu1_op.get_shape()[4])
        up_conv1=tf.nn.conv3d_transpose(uprelu1_op, W_uptrans, [x_shape0, x_shape1*2, x_shape2*2, x_shape3*2, x_shape4], strides=[1, 1, 1, 1, 1], padding='VALID')
        print_tensor_shape(up_conv1, 'up_conv1 shape')
     
        
        W_uptrans=tf.Variable(tf.truncated_normal([2,2,2,256,256],stddev=0.1,
                                                dtype=tf.float32), name='W_uptrans')
        x_shape0=int(downrelu2_op.get_shape()[0])
        x_shape1=int(downrelu2_op.get_shape()[1])
        x_shape2=int(downrelu2_op.get_shape()[2])
        x_shape3=int(downrelu2_op.get_shape()[3])
        x_shape4=int(downrelu2_op.get_shape()[4])
        up_conv1=tf.nn.conv3d_transpose(downrelu2_op, W_uptrans, [x_shape0, x_shape1*2, x_shape2*2, x_shape3*2, x_shape4], strides=[1, 2, 2, 2, 1], padding='VALID')
        print_tensor_shape(up_conv1, 'up_conv1 shape')
    

    with tf.name_scope('up_h_convs1'):    
        x1_shape1=int(relu8_op.get_shape()[1])
        x1_shape2=int(relu8_op.get_shape()[2])
        x1_shape3=int(relu8_op.get_shape()[3])
        x2_shape1=int(up_conv1.get_shape()[1])
        x2_shape2=int(up_conv1.get_shape()[2])
        x2_shape3=int(up_conv1.get_shape()[3])
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape1 - x2_shape1) // 2, (x1_shape2 - x2_shape2) // 2,  (x1_shape3 - x2_shape3) // 2, 0]
        size = [-1, x2_shape1, x2_shape2, x2_shape3,  -1]
        x1_crop = tf.slice(relu8_op, offsets, size)
        print_tensor_shape(x1_crop, 'x1_crop shape')
        
        
        up_h_convs1 =tf.concat([x1_crop, up_conv1] , axis=4)  
        print_tensor_shape(up_h_convs1, 'up_h_convs1 shape')

    #print('size of up_h_convs[0] = ', up_h_convs[0].get_shape().as_list())
        
    with tf.name_scope('conv9'):
        W_conv9=tf.Variable(tf.truncated_normal([3,3,3,256, 128], stddev=0.1,
                                                 dtype=tf.float32), name='W_conv9')
        conv9_op=tf.nn.conv3d(up_h_convs1, W_conv9, strides=[1,1,1,1,1],
                               padding="VALID", name='conv9_op')
        print_tensor_shape(conv9_op, 'conv_op shape')
        
        relu9_op=tf.nn.relu(conv9_op, 'conv9_op')
    with tf.name_scope('conv10'):
        W_conv10=tf.Variable(tf.truncated_normal([3,3,3,128,128], stddev=0.1, 
                                                 dtype=tf.float32), name='W_conv10')
        conv10_op=tf.nn.conv3d(relu9_op, W_conv10, strides=[1,1,1,1,1], 
                               padding="VALID", name='conv10_op')
        print_tensor_shape(conv10_op, 'conv10_op shape')
        relu10_op=tf.nn.relu(conv10_op, 'relu10_op')
        print_tensor_shape(relu10_op, 'relu10_op shape')
        
 #upsampling 2      
    with tf.name_scope('upconv2'):
        W_upconv2=tf.Variable(tf.truncated_normal([2,2,2,128,64],stddev=0.1,
                                                dtype=tf.float32), name='W_upcon2')
        print_tensor_shape(W_upconv2, 'W_upconv2 shape')
        upconv2_op=tf.nn.conv3d(relu10_op, W_upconv2, strides=[1,1,1,1,1],
                             padding="VALID", name='upconv2_op')
        uprelu2_op=tf.nn.relu(upconv2_op, 'uprelu2_op')
        print_tensor_shape(uprelu2_op, 'uprelu2_op shape')
        W_uptrans2=tf.Variable(tf.truncated_normal([2,2,2,64,64],stddev=0.1,
                                                dtype=tf.float32), name='W_uptrans2')
        x_2shape0=int(uprelu2_op.get_shape()[0])
        x_2shape1=int(uprelu2_op.get_shape()[1])
        x_2shape2=int(uprelu2_op.get_shape()[2])
        x_2shape3=int(uprelu2_op.get_shape()[3])
        x_2shape4=int(uprelu2_op.get_shape()[4])
        up_conv2=tf.nn.conv3d_transpose(uprelu2_op, W_uptrans2, [x_2shape0, x_2shape1*2, x_2shape2*2, x_2shape3*2, x_2shape4], strides=[1, 1, 1, 1, 1], padding='VALID')
       

    with tf.name_scope('up_h_convs2'):    
        x11_shape1=int(relu6_op.get_shape()[1])
        x11_shape2=int(relu6_op.get_shape()[2])
        x11_shape3=int(relu6_op.get_shape()[3])
        x22_shape1=int(up_conv2.get_shape()[1])
        x22_shape2=int(up_conv2.get_shape()[2])
        x22_shape3=int(up_conv2.get_shape()[3])
              # offsets for the top left corner of the crop
        offsets = [0, (x11_shape1 - x22_shape1) // 2, (x11_shape2 - x22_shape2) // 2,  (x11_shape3 - x22_shape3) // 2, 0]
        size = [-1, x22_shape1, x22_shape2, x22_shape3,  -1]
        x2_crop = tf.slice(relu6_op, offsets, size)
        print_tensor_shape(x2_crop, 'x2_crop shape')
        print_tensor_shape(up_conv2, 'up_conv1 shape')
        
        up_h_convs2=tf.concat([x2_crop, up_conv2] , axis=4)  
        print_tensor_shape(up_h_convs2, 'up_h_convs2 shape')


    with tf.name_scope('conv11'):
        W_conv11=tf.Variable(tf.truncated_normal([3,3,3,128, 64], stddev=0.1,
                                                 dtype=tf.float32), name='W_conv11')
        conv11_op=tf.nn.conv3d(up_h_convs2, W_conv11, strides=[1,1,1,1,1],
                               padding="VALID", name='conv11_op')
        print_tensor_shape(conv11_op, 'conv11_op shape')
        
        relu11_op=tf.nn.relu(conv11_op, 'conv11_op')
    with tf.name_scope('conv12'):
        W_conv12=tf.Variable(tf.truncated_normal([3,3,3,64,64], stddev=0.1, 
                                                 dtype=tf.float32), name='W_conv12')
        conv12_op=tf.nn.conv3d(relu11_op, W_conv12, strides=[1,1,1,1,1], 
                               padding="VALID", name='conv12_op')
        print_tensor_shape(conv12_op, 'conv12_op shape')
        relu12_op=tf.nn.relu(conv12_op, 'conv12_op')
         
#upsampling 3     
    with tf.name_scope('upconv3'):
        W_upconv3=tf.Variable(tf.truncated_normal([2,2,2,64,32],stddev=0.1,
                                                dtype=tf.float32), name='W_upconv3')
        print_tensor_shape(W_upconv3, 'W_upconv3 shape')
        upconv3_op=tf.nn.conv3d(relu12_op, W_upconv3, strides=[1,1,1,1,1],
                             padding="VALID", name='upconv3_op')
        uprelu3_op=tf.nn.relu(upconv3_op, 'uprelu3_op')
        print_tensor_shape(uprelu3_op, 'uprelu3_op shape')
        W_uptrans3=tf.Variable(tf.truncated_normal([2,2,2,32,32],stddev=0.1,
                                                dtype=tf.float32), name='W_uptrans3')
        x_3shape0=int(uprelu3_op.get_shape()[0])
        x_3shape1=int(uprelu3_op.get_shape()[1])
        x_3shape2=int(uprelu3_op.get_shape()[2])
        x_3shape3=int(uprelu3_op.get_shape()[3])
        x_3shape4=int(uprelu3_op.get_shape()[4])
        up_conv3=tf.nn.conv3d_transpose(uprelu3_op, W_uptrans3, [x_3shape0, x_3shape1*2, x_3shape2*2, x_3shape3*2, x_3shape4], strides=[1, 1, 1, 1, 1], padding='VALID')
       

    with tf.name_scope('up_h_convs3'):    
        x111_shape1=int(relu4_op.get_shape()[1])
        x111_shape2=int(relu4_op.get_shape()[2])
        x111_shape3=int(relu4_op.get_shape()[3])
        x222_shape1=int(up_conv3.get_shape()[1])
        x222_shape2=int(up_conv3.get_shape()[2])
        x222_shape3=int(up_conv3.get_shape()[3])
              # offsets for the top left corner of the crop
        offsets = [0, (x111_shape1 - x222_shape1) // 2, (x111_shape2 - x222_shape2) // 2,  (x111_shape3 - x222_shape3) // 2, 0]
        size = [-1, x222_shape1, x222_shape2, x222_shape3,  -1]
        x3_crop = tf.slice(relu4_op, offsets, size)
        print_tensor_shape(x3_crop, 'x3_crop shape')
        print_tensor_shape(up_conv3, 'up_conv3 shape')
        
        up_h_convs3=tf.concat([x3_crop, up_conv3] , axis=4)  
        print_tensor_shape(up_h_convs3, 'up_h_convs3 shape')


          
    with tf.name_scope('conv13'):
        W_conv13=tf.Variable(tf.truncated_normal([3,3,3,64, 32], stddev=0.1,
                                                 dtype=tf.float32), name='W_conv13')
        conv13_op=tf.nn.conv3d(up_h_convs3, W_conv13, strides=[1,1,1,1,1],
                               padding="VALID", name='conv13_op')
        print_tensor_shape(conv13_op, 'conv13_op shape')
        
        relu13_op=tf.nn.relu(conv13_op, 'conv13_op')
        
    with tf.name_scope('conv14'):
        W_conv14=tf.Variable(tf.truncated_normal([3,3,3,32,32], stddev=0.1, 
                                                 dtype=tf.float32), name='W_conv14')
        conv14_op=tf.nn.conv3d(relu13_op, W_conv14, strides=[1,1,1,1,1], 
                               padding="VALID", name='conv14_op')
        print_tensor_shape(conv14_op, 'conv14_op shape')
        relu14_op=tf.nn.relu(conv14_op, 'relu14_op')
        print_tensor_shape(relu14_op, 'relu14_op')
        
#upsampling 4
    with tf.name_scope('upconv4'):
        W_upconv4=tf.Variable(tf.truncated_normal([2,2,2,32,16],stddev=0.1,
                                                dtype=tf.float32), name='W_upconv4')
        print_tensor_shape(W_upconv3, 'W_upconv4 shape')
        upconv4_op=tf.nn.conv3d(relu14_op, W_upconv4, strides=[1,1,1,1,1],
                             padding="VALID", name='upconv4_op')
        uprelu4_op=tf.nn.relu(upconv4_op, 'uprelu4_op')
        print_tensor_shape(uprelu4_op, 'uprelu4_op shape')
        W_uptrans4=tf.Variable(tf.truncated_normal([2,2,2,16,16],stddev=0.1,
                                                dtype=tf.float32), name='W_uptrans4')
        x_4shape0=int(uprelu4_op.get_shape()[0])
        x_4shape1=int(uprelu4_op.get_shape()[1])
        x_4shape2=int(uprelu4_op.get_shape()[2])
        x_4shape3=int(uprelu4_op.get_shape()[3])
        x_4shape4=int(uprelu4_op.get_shape()[4])
        up_conv4=tf.nn.conv3d_transpose(uprelu4_op, W_uptrans4, [x_4shape0, x_4shape1*2, x_4shape2*2, x_4shape3*2, x_4shape4], strides=[1, 1, 1, 1, 1], padding='VALID')
       

    with tf.name_scope('up_h_convs3'):    
        x1111_shape1=int(relu2_op.get_shape()[1])
        x1111_shape2=int(relu2_op.get_shape()[2])
        x1111_shape3=int(relu2_op.get_shape()[3])
        x2222_shape1=int(up_conv4.get_shape()[1])
        x2222_shape2=int(up_conv4.get_shape()[2])
        x2222_shape3=int(up_conv4.get_shape()[3])
              # offsets for the top left corner of the crop
        offsets = [0, (x1111_shape1 - x2222_shape1) // 2, (x1111_shape2 - x2222_shape2) // 2,  (x1111_shape3 - x2222_shape3) // 2, 0]
        size = [-1, x2222_shape1, x2222_shape2, x2222_shape3,  -1]
        x4_crop = tf.slice(relu2_op, offsets, size)
        print_tensor_shape(x4_crop, 'x4_crop shape')
        print_tensor_shape(up_conv4, 'up_conv4 shape')
        
        up_h_convs4=tf.concat([x4_crop, up_conv4] , axis=4)  
        print_tensor_shape(up_h_convs4, 'up_h_convs4 shape')


    with tf.name_scope('conv15'):
        W_conv15=tf.Variable(tf.truncated_normal([3,3,3,32,16],stddev=0.1,
                                                dtype=tf.float32), name='W_conv15')
        conv15_op=tf.nn.conv3d(up_h_convs4, W_conv15, strides=[1,1,1,1,1], 
                              padding="VALID", name='conv15_op')
        print_tensor_shape(conv15_op, 'conv15_op')
        relu15_op=tf.nn.relu(conv15_op, 'relu15_op')
        print_tensor_shape(relu15_op, 'relu15_op shape')

    with tf.name_scope('conv16'):
         W_conv16=tf.Variable(tf.truncated_normal([3,3,3,16, 16], stddev=0.1,
                                                 dtype=tf.float32), name='W_conv16')
         conv16_op=tf.nn.conv3d(relu15_op, W_conv16, strides=[1,1,1,1,1],
                               padding="VALID", name='conv16_op')
         print_tensor_shape(conv16_op, 'conv16_op shape')
        
         relu19_op=tf.nn.relu(conv16_op, 'conv16_op')
         print_tensor_shape(relu19_op, 'relu19_op shape')

        

        
                  
            
        
        
        
        
       
                               
       