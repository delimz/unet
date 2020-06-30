# This is the main script to train and predict masks from precomputed slice of large images

# the data should be separated in one directory named crop for the tiles of the
# images and one directory per segmentation class with png images of the masks
# eg: 

# !ls datadir
# 560 568 576 584 592 crop

# the naming of the images and corresponding crops is expected to be
# <image id>_<slice index>_<horizontal position>_<vertical position>_<horizontal size>_<vertical size>.<format>

# image id : a unique identifier for the original image
# slice index : to diferenciate multiple regions of interest in the same images
# horizontal and vertical position : postion of the upper left corner of the tile within the region of interest
# horizontal and vertical sizes : size of the tile
# format : jpg for image, png for mask


import functools
import os
import datetime
import re

from params import parse_params
from sys import argv

import tensorflow as tf
import numpy as np

from data.preprocessing import get_dataset,augment
from unet.model import unet
from unet.loss import superloss
from unet.metrics import Dice,DiceX,DiceG
from unet.utils import crop_size

from unet.preconf import conf

os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'

params=parse_params(argv[1:])
###initialization of the data split and classes

#the list of id for images used in training
imgs_train=params.imgs_train

#list of ids of the classes
terms=params.terms

#list of the id for images used in training to select the model to keep 
imgs_val=params.imgs_val

#the list of id for images used in testing for which fullsize mask will be written
imgs_test=params.imgs_test
datadir=params.datadir


#sort images by horzontal then vertical position to reconstruct the full images later
def sortkey(x):
    sp=x.split('_')
    return int(sp[2])*100000+int(sp[3])

filenames=sorted(os.listdir(os.path.join(datadir,'crop')),key=sortkey)


#filter data a compute list of actual files
def startswith(x,l):
    for e in l:
        if x.startswith(str(e)) and (x.endswith("_{}.png".format(params.crop_size)) or x.endswith("_{}.jpg".format(params.crop_size))):
            return True
    return False

train_filenames=[f[:-4] for f in filter(lambda x : startswith(x,imgs_train),filenames) ]

x_train_filenames = [os.path.join(datadir,'crop',x+'.jpg') for x in train_filenames]
y_train_filenames=[]
for i in range(len(terms)):
    y_train_filenames.append([os.path.join(datadir, str(terms[i]),x+'.png') for x in train_filenames])

val_filenames=[f[:-4] for f in filter(lambda x : startswith(x,imgs_val),filenames)]

x_val_filenames = [os.path.join(datadir,'crop',x+'.jpg') for x in val_filenames]
y_val_filenames=[]
for i in range(len(terms)):
    y_val_filenames.append([os.path.join(datadir, str(terms[i]),x+'.png') for x in val_filenames])

test_filenames=[f[:-4] for f in filter(lambda x : startswith(x,imgs_test),filenames)]

x_test_filenames = [os.path.join(datadir,'crop',x+'.jpg') for x in test_filenames]
y_test_filenames=[]
for i in range(len(terms)):
    y_test_filenames.append([os.path.join(datadir, str(terms[i]),x+'.png') for x in test_filenames])

# convert list of filenames into numpy arrays
x_train_filenames=np.array(x_train_filenames)
y_train_filenames=np.array(y_train_filenames).transpose()
x_val_filenames=np.array(x_val_filenames)
y_val_filenames=np.array(y_val_filenames).transpose()
x_test_filenames=np.array(x_test_filenames)
y_test_filenames=np.array(y_test_filenames).transpose()

print((x_train_filenames.shape,y_train_filenames.shape),(x_val_filenames.shape,y_val_filenames.shape),(x_test_filenames.shape,y_test_filenames.shape))

#initialize the parameter of the network
net = {
        "name" : params.checkpoint,
        "back_leg" : params.back_leg,
        "residuals" : params.residuals,
        "depth" : params.depth,
        "conv" : params.convolutions,
        "filters" : params.filter_sizes,
        "dilations" : params.dilations
}

if params.preset is not None:
    print('yes preset')
    try:
        model_conf=conf[params.preset]
    except:
        print('preset {} not found, exiting'.format(params.preset))
        exit(1)
    
else:
    print('no preset')
    print(net)
    model_conf=net

#compute the required size of input for desired size of output
min_output_size=512
def min_allowed_size(**conf):
    crop_size_conf=crop_size(conf["filters"],conf["dilations"],conf["conv"])
    return 2*crop_size_conf

for iss in range(min_allowed_size(**model_conf)+min_output_size,min_allowed_size(**model_conf)+min_output_size*2,2):
    input_shape=(iss,iss,3)
    try:
        input_layer=tf.keras.layers.Input(shape=input_shape)
        model=tf.keras.Model(input_layer,unet(input_layer,len(params.terms),**model_conf))
        output_shape=model.output_shape
        print(input_shape,output_shape)
        if output_shape[1]>min_output_size:
            break
    except Exception as e:
        print('not',input_shape,e)
    
#preprocessing and data augmentation configuration options
tr_cfg = {
    'nn_input_shape': (input_shape[1],input_shape[1]),
    'nn_output_shape': (output_shape[1],output_shape[1]),
    'hue_delta': params.hue_delta,
    'saturation_delta' : params.saturation_delta,
    'brightness_delta' : params.brightness_delta,
    'contrast_delta': params.contrast_delta,
    'rotate': params.rotate_range,
    'deformation_range' : params.deformation_range,
    'background_class' : False,
    'orig_shape': params.crop_size,
}

# Dataset initialization

tr_preprocessing_fn = functools.partial(augment, **tr_cfg)

train_ds=get_dataset(x_train_filenames,y_train_filenames,
                                x_train_filenames.shape[0],
                                    tr_preprocessing_fn,
                                    batch_size=params.batch_size,
                                    threads=params.n_jobs,
                                    deterministic=False,
                                    filter_empty=True)


val_cfg = {
    'nn_input_shape': (input_shape[1],input_shape[1]),
    'nn_output_shape': (output_shape[1],output_shape[1]),
    'background_class' : False,
    'orig_shape': params.crop_size
}
val_preprocessing_fn = functools.partial(augment, **val_cfg)

val_ds = get_dataset(x_val_filenames,
                                  y_val_filenames,
                                  x_val_filenames.shape[0],
                                  val_preprocessing_fn,
                                  batch_size=params.batch_size,
                                  deterministic=False,
                                  threads=params.n_jobs)

#model preparation
checkpoint_path = params.checkpoint
checkpoint_dir = os.path.dirname(checkpoint_path)

num_mask=len(params.terms)
model.compile('adam',superloss,metrics=[Dice(),*[DiceX(i) for i in range(num_mask)],DiceG(num_mask)])

try:
    model.load_weights(checkpoint_path)
except:
    print('checkpoint not loaded')
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 monitor='val_geo_dice',
                                                  mode='max',
                                                    save_best_only=True,
                                                 verbose=1)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch = '500,520')


#model training
history=model.fit(train_ds.batch(params.batch_size).prefetch(50),
          callbacks=[tensorboard_callback,cp_callback],
          validation_data=val_ds.batch(params.batch_size).prefetch(50),
          epochs=params.epochs)


test_batch_size=10

def get_masks(model,img_id,img_index,x_test_filenames_single,y_test_filename_single):
    ''' make the output masks for one image '''
    test_cfg = {
        'nn_input_shape': (input_shape[1],input_shape[1]),
        'nn_output_shape': (output_shape[1],output_shape[1]),
        'background_class' : False
    }
    test_preprocessing_fn = functools.partial(augment, **test_cfg)

    test_ds= get_dataset(x_test_filenames_single,
                                      y_test_filenames_single,
                                      x_test_filenames_single.shape[0],
                                      val_preprocessing_fn,
                                      batch_size=1,
                                      threads=params.n_jobs,
                                      deterministic=True,
                                      shuffle=False)

    
    
    
    ds=test_ds.batch(test_batch_size).prefetch(100)
    return ds

@tf.function
def predict(model,dataset,img_id,img_index):
    res=tf.TensorArray(tf.uint8,size=0, dynamic_size=True, clear_after_read=False)
    i=0

    for x,y in dataset:
        tf.print(x.get_shape())
        prediction=tf.cast(model(x)*255,tf.uint8)
        res=res.write(i,prediction)
        tf.print(i)
        i+=1


    res=res.concat()
    tf.print(res.get_shape())
    return res

@tf.function(experimental_relax_shapes=True)
def make_images(res,img_id,img_index,min_output_size,output_shape,line,col,model_name,num_channels,mask_names):
    tf.print(res.get_shape())

    #res=tf.image.central_crop(res,min_output_size/output_shape)
    offset=(output_shape-min_output_size)//2
    res=tf.image.crop_to_bounding_box(res,offset,offset,min_output_size,min_output_size)
    res=tf.reshape(res,(line,col,min_output_size,min_output_size,num_channels))
    res=tf.transpose(res,(0,2,1,3,4))
    res=tf.reshape(res,(line*min_output_size,col*min_output_size,num_channels))
    for i in range(num_channels):
        tf.io.write_file(mask_names[i],tf.image.encode_png(res[:,:,i:i+1]))
    return 1

#now make the output masks for each testing images
num_mask_t=tf.constant(num_mask)
output_shape_1=tf.constant(output_shape[1])
min_output_size=tf.constant(min_output_size)
model_name=tf.constant(model_conf['name'])
for img_id in imgs_test:
    for img_index in [0,1]:
        print("processing {}_{}".format(img_id,img_index))
        prog=re.compile(".*/{}_{}_.*".format(img_id,img_index))
        x_test_filenames_single=np.array([s for s in filter(lambda x:prog.match(x) is not None,x_test_filenames)])
        y_test_filenames_single=np.array([[s for s in filter(lambda x:prog.match(x) is not None, y_test_filenames[:,i])] for i in range(num_mask)]).transpose()
        print(x_test_filenames_single.shape,y_test_filenames_single.shape)
        prog1=re.compile('{}_{}_0_.*'.format(img_id,img_index))
        prog2=re.compile('{}_{}_.*_0_.*'.format(img_id,img_index))
        col=len([s for s in filter(lambda x: prog1.match(x) is not None,test_filenames)])
        line=len([s for s in filter(lambda x: prog2.match(x) is not None,test_filenames)])
        print(col,line)
        if col==0 or line==0:
            break
        mask_names=tf.constant(['fullmask_{}_{}_{}_{}.png'.format(model_conf['name'],img_id,img_index,i) for i in range(num_mask)])
        img_id=tf.constant(img_id)
        img_index=tf.constant(img_index) 
        line=tf.constant(line)
        col=tf.constant(col)
        ds=get_masks(model,img_id,img_index,x_test_filenames_single,y_test_filenames_single)
        res=predict(model,ds,img_id,img_index)
        make_images(res,img_id,img_index,min_output_size,output_shape_1,line,col,model_name,num_mask_t,mask_names)

