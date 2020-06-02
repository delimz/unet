import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from time import time

from shapely.wkt import loads
from shapely.affinity import translate

import tensorflow.compat.v1 as tf

from cytomine import Cytomine
from cytomine.models import AnnotationCollection,Annotation
from cytomine.models.image import ImageInstanceCollection

from argparse import ArgumentParser
import sys

parser = ArgumentParser(prog="ratseg_postproc")
parser.add_argument('--cytomine_host', dest='host',
                    default='http://localhost-core', help="The Cytomine host")
parser.add_argument('--cytomine_public_key', dest='public_key',
                    default='39d81d3f-fcfc-494c-914e-8f0a8814de4e',
                    help="The Cytomine public key")
parser.add_argument('--cytomine_private_key', dest='private_key',
                    help="The Cytomine private key",
                    default='132cb1d0-ae3c-4d03-8271-c87dcfc612cd')
parser.add_argument('--cytomine_id_project', dest='id_project',
                    help="The project from which we want the images",
                    default=155)
parser.add_argument('--slice_term',type=int,
                    help="id of the ROI delimiting annotation",
                    default=30289)

parser.add_argument('--model','-m',help="name of the model to evaluate")

parser.add_argument('--imgs-val',type=int,nargs='+',default=[2319547,2319553,2319561,2319567])
parser.add_argument('--imgs-test',type=int,nargs='+',default=[2319587])#[2319573,2319579,2319587,2319595])

parser.add_argument('--terms',type=int,nargs='+',default=[1012286,1012259,1012265,1012280]) #gm l b d #1012294 = gc

parser.add_argument('--threshold',type=float,default=0.5)
parser.add_argument('--no',type=int,default=4,help="number of errosion and dilation passes for openning and closing")

parser.add_argument('--upload',type=bool,default=False,help='upload the resulting annotations to cytomine')

params=parser.parse_args(sys.argv[1:])

model=params.model
threshold=params.threshold
test_imgs=params.imgs_test
upload=params.upload

print("imgs",test_imgs)

host=params.host
public_key=params.public_key
private_key=params.private_key
id_project=params.id_project
slice_term=params.slice_term
terms=params.terms
no=params.no
threshold=params.threshold
crop_size=parmas.crop_size

def getpolygon(img,offset=(0,0)):
    res=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    poly="MULTIPOLYGON ({})"
    polys=[]
    if len(res[0])>0:
        for p in res[0]:
            if len(p)>4:
                subpoly="(({}))"
                try:
                    ps=["%d %d" % (x[0][0]+offset[0],x[0][1]+offset[1]) for x in p]
                except Exception as e:
                    print(res)
                    print(p)
                    print(e)
                    exit(1)
                ps.append(ps[0])
                points=','.join(ps)
                polys.append(subpoly.format(points))
    if len(polys)==0:
        polys=["EMPTY"]
    final=poly.format(",".join(polys))
    try:
        res=loads(final)
    except Exception as e:
        print(final)
        print('ERROR : ',e)
        exit(0)
        res=loads(poly.format("EMPTY"))

    return res

def one_of_us(s):
    for img in test_imgs:
        if s.startswith('fullmask_%s_%d' % (model,img)):
            return True
    return False

shapes={}
fm=sorted([f for f in filter(one_of_us ,os.listdir('.'))])
print(fm)

for f in fm:
    img=cv2.imread("./%s" % f,cv2.IMREAD_GRAYSCALE)
    imshape=img.shape
    ks=12
    start=time()
    kernel1=np.array(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks,ks))/255.0,dtype=np.float16)
    kernel1=kernel1.reshape((ks,ks,1))
    kernel2=np.array(np.matmul(cv2.getGaussianKernel(ks,-1),cv2.getGaussianKernel(ks,-1).transpose()),dtype=np.float16).reshape((ks,ks,1))

    img=tf.image.convert_image_dtype(img,dtype=tf.float16)
    img=tf.reshape(img,(1,imshape[0],imshape[1],1))
    for i in range(no):
        img=tf.nn.erosion2d(img,kernel2,[1,1,1,1],[1,1,1,1],'SAME')
    for i in range(no*2):
        img=tf.nn.dilation2d(img,kernel2,[1,1,1,1],[1,1,1,1],'SAME')
    for i in range(no):
        img=tf.nn.erosion2d(img,kernel2,[1,1,1,1],[1,1,1,1],'SAME')
    img=img[0,:,:,0]
    t=threshold
    cond=tf.less(img,tf.constant(t,shape=imshape,dtype=tf.float16))
    imgt=tf.where(cond,tf.zeros(tf.shape(img),dtype=tf.float16),img)
    imgt=tf.image.convert_image_dtype(imgt,dtype=tf.uint8)
    imgt=cv2.flip(imgt.numpy(),0)
    shapes[f]=getpolygon(imgt)
    print(time() - start)
    print(f,t,"valid?",shapes[f].is_valid)
        

ploys={}
for f in shapes.keys():
    if shapes[f].is_valid:
        print(0)
        ploys[f]=shapes[f]
        continue
    else:
        i=0
        while i<1000:
            if shapes[f].simplify(i).is_valid:
                break
            i=i+1
        print(i)
        ploys[f]=shapes[f].simplify(i)

#populate res with the annotations
with Cytomine(host=host, public_key=public_key, private_key=private_key) as cytomine:
    res={}
    annotations = AnnotationCollection()
    annotations.project = id_project
    annotations.showWKT = True
    annotations.showMeta = True
    annotations.showGIS = True
    annotations.showTerm = True
    annotations.showImage = True
    annotations.fetch()
    print(annotations)
    for annotation in annotations:
        '''
        print("ID: {} | Img: {} | Pjct: {} | Term: {} ".format(
            annotation.id,
            annotation.image,
            annotation.project,
            annotation.term
        ))
        '''
        if len(annotation.term)==1:
            if (annotation.term[0],annotation.image) not in res.keys():
                res[(annotation.term[0],annotation.image)]=[]
            res[(annotation.term[0],annotation.image)].append(loads(annotation.location))
            
            if not res[(annotation.term[0],annotation.image)][-1].is_valid:
                raise 'fuck'

results=[]
trues=[]
shapes=[]
for test_img in test_imgs:
    for index in range(len(res[slice_term,test_img])):
        ploy=[ploys['fullmask_%s_%d_%d_%d.png' % (model,test_img,index,t)] for t in range(len(terms))]

        maxres=[0.0]*len(terms)
        maxshape=None
        maxtrue=None
        for box in res[slice_term,test_img]:
            trueshape=[]
            for term in terms:
                tgm=res[term,test_img][0]
                for x in res[term,test_img]:
                    tgm=tgm.union(x)
                tgm=tgm.intersection(box)
                trueshape.append(tgm)

            bounds=box.bounds
            print(bounds)
            offx=bounds[0]
            offy=bounds[1]-((((bounds[3]-bounds[1])//512)+(crop_size//512))*512)+(bounds[3]-bounds[1])
            print('off=(',offx,',',offy,')')

            predshape=[]
            for i in range(len(terms)):
                predshape.append(translate(ploy[i],yoff=offy,xoff=offx))

            '''
            try:
                b=b.difference(l)
                d=d.difference(b.union(l))
                gm=gm.difference(l.union(b).union(d))

            except Exception as e:
                print('difference introduce error')
                print(e)
            '''
            tmpres=[]
            for i in range(len(terms)):
                if trueshape[i].area>0:
                    tmpres.append(predshape[i].intersection(trueshape[i]).area/predshape[i].union(trueshape[i]).area)
                else:
                    tmpres.append(0.0)
            if np.mean(tmpres) > np.mean(maxres):
                maxres=tmpres
                maxshape=predshape
                maxtrue=trueshape
            else:
                print('nope')
        results.append(maxres)
        shapes.append(maxshape)
        trues.append(maxtrue)

if upload:
    with Cytomine(host=host, public_key=public_key, private_key=private_key) as cytomine:
        num=0
        for test_img in test_imgs:
            for index in range(len(res[slice_term,test_img])):
                for i in range(len(terms)):
                    new_annotation=Annotation(location=shapes[num][i].wkt,id_image=test_img,id_terms=[terms[i]],id_project=id_project)
                    new_annotation.save()
                num+=1

print(results)
