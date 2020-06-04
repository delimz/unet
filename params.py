from argparse import ArgumentParser
from sys import argv
import json
from tf.data.experimental import AUTOTUNE


def parsebool(x):
    return x in ['True','True','t','yes','y','yep','affirmative','yesplease','ya','oy','oui','ui','v','V','Vrai']

#parameter list of tuple 
#(parameter id, number of arguments, type, default value, long description)
cytomine_params=[
        ("cytomine_host",None,str,[],'address of the cytomine server'),
        ("cytomine_public_key",None,str,[],'api public key for cytomine server'),
        ("cytomine_private_key",None,str,[],'api private key for the cytomine server'),
        ("cytomine_id_project",None,int,[],'project id on the cytomine server'),
        ("cytomine_id_software",None,int,[],'software id on the cytomine server'),
    ]

crop_params=[
        ("imgs_train",'*',int,[],'id of the images used for training'),
        ("imgs_val",'*',int,[],'id of the images used for validation'),
        ("imgs_test",'*',int,[],'id of the images used for testing'),
        ("datadir",None,str,'/mnt/SSD_128/tmp','where the crops are stored'),
        ("terms",'*',int,[],'id of the terms to learn or predict'),
        ("slice_term",None,int,-1,'id of the annotation delimiting the zone to segment'),
        ("crop_size",None,int,1024,'size of the square tiles to crop'),
    ]

augmentations_params=[
        ("hue_delta",None,float,0.05,'data augmentation by hue alteration'),
        ("saturation_delta",None,float,0.05,'data augmentation by staturation alteration range'),
        ("brightness_delta",None,float,0.05,'data augmentation by brightness alteration range'),
        ("contrast_delta",None,float,0.05,'data augmentation by contrast alteration range'),
        ("horizontal_flip",None,parsebool,True,'data augmentation by flipping images horizontally'),
        ("rotate_range",None,float,15.0,'data augmentation by rotation angle range'),
        ("deformation_range",None,float,0.02,'data augmentation by streching of the image'),
    ]

model_params=[
        ("preset",None,str,None,'preset model parameters, either unet vnet mnet segnet'),
        ("depth",None,int,4,'depth of unet'),
        ("convolutions",'*',int,[2],'number of conv per encoder'),
        ("filter_factor",None,int,16,'base number of filters in first level of convolution'),
        ("dilations",'*',int,[1],'size of dilation'),
        ("filter_sizes",'*',int,[3],'size of the filters for convolutions'),
        ("residuals",None,parsebool,False,'use residual connection'),
        ("front_leg",None,parsebool,False,'deep supervision encoder side'),
        ("back_leg",None,parsebool,False,'deep supervision decoder side'),
        ("checkpoint",None,str,None,'checkpoint path to use'),
        ("min_output_size",None,int,512,'minimum size of output of the neural network'),
        ("n_jobs",None,int,AUTOTUNE,'number of cpu jobs for image preprocessing'),
        ("batch_size",None,int,5,'number of tiles per batch of training step'),
        ("epochs",None,int,5,'number of epochs to train for'),
    ]

postproc_params=[
        ("oc_num",None,int,4,'number of openning and closing'),
        ("threshold",None,float,0.5,'Threshold to retain'),
        ("upload",None,parsebool,False,'upload the result')
    ]

def parserer():
    parser = ArgumentParser(prog="unet")
    params=[]
    params.extend(cytomine_params)
    params.extend(crop_params)
    params.extend(augmentations_params)
    params.extend(model_params)
    params.extend(postproc_params)

    for param,nargs,typef,default,helps in params:
        parser.add_argument("--{}".format(param),nargs=nargs,default=default,type=typef,help=helps)

    return parser

def parse_params(argv):
    parser=parserer()
    parsed=parser.parse_args(argv)

    if len(parsed.filter_sizes)==1:
        parsed.filter_sizes=[parsed.filter_sizes[0] for i in range(parsed.depth+1)]

    if len(parsed.dilations)==1:
        parsed.dilations=[parsed.dilations[0] for i in range(parsed.depth+1)]

    if len(parsed.convolutions)==1:
        parsed.convolutions=[parsed.convolutions[0] for i in range(parsed.depth+1)]

    return parsed

def lay_parameters_defaults():
    params=[]
    params.extend(cytomine_params)
    params.extend(crop_params)
    params.extend(augmentations_params)
    params.extend(model_params)
    params.extend(postproc_params)

    for param,nargs,typef,default,helps in params:
        print("--{}".format(param),end=' ')
        if nargs == '*':
            for i in default:
                print(str(i),end=' ')
        else:
            print(default,end=' ')
        print('\\')


def types(n,x):
    if n is None:
        if x is int or x is float:
            return 'Number'
        if x is str:
            return 'String'
        if x is parsebool:
            return 'Boolean'
    if n == '*' or n == '+':
        return 'String'

def make_descriptor():
    descriptor = {
            'name' : 'ratseg2',
            'container-image' : {
                'image' : 'delimz/s_unet',
                'type' : 'singularity'
                },
            'schema-version' : 'cytomine-0.1',
            'description' : 'segmentation by unet-like neural network'
            }
    
    descriptor['inputs']=[]
    commandLine=[]

    for param,nargs,typef,default,helps in cytomine_params:
        commandLine.append(param.upper())
        descriptor['inputs'].append(
                                    {
                                        'id' : param,
                                        'value-key' : "@ID",
                                        'command-line-flag' : "--@id",
                                        'name' : param.replace('_',' '),
                                        'description' : helps,
                                        'set-by-server' : True,
                                        'optional' : False,
                                        'type' : types(nargs,typef),
                                        'default' : default,
                                        }
                                    )
    params=[]
    params.extend(model_params)
    params.extend(augmentations_params)
    params.extend(crop_params)
    params.extend(postproc_params)
    for param,nargs,typef,default,helps in params:
        commandLine.append(param.upper())
        descriptor['inputs'].append(
                                    {
                                        'id' : param,
                                        'value-key' : "@ID",
                                        'command-line-flag' : "--@id",
                                        'name' : param.replace('_',' '),
                                        'description' : helps,
                                        'set-by-server' : False,
                                        'optional' : True,
                                        'type' : types(nargs,typef),
                                        'default' : default,
                                        }
                                    )
    
    descriptor['command-line'] = ' '.join(commandLine)
    return json.dumps(descriptor)

if __name__ == '__main__':
    print(make_descriptor())

