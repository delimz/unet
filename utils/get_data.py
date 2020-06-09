import numpy as np
import cv2

from cytomine import Cytomine
from cytomine.models import AnnotationCollection,Annotation
from cytomine.models.image import ImageInstanceCollection

import cairosvg as cs
import openslide as osl
import os

from shapely.wkt import loads

from argparse import ArgumentParser
import sys

from get_images import get_image_map

parser = ArgumentParser(prog="ratseg_get_data")

parser.add_argument('--cytomine_host', dest='host',
                    default='http://localhost-core', help="The Cytomine host")
parser.add_argument('--cytomine_public_key', dest='public_key',
                    default='d5ebfff1-2517-47f9-9a71-a6073ef3250f',
                    help="The Cytomine public key")
parser.add_argument('--cytomine_private_key', dest='private_key',
                    help="The Cytomine private key",
                    default='0337a7a5-7a00-410d-9c62-d9080ea0de52')
parser.add_argument('--cytomine_id_project', dest='id_project',
                    help="The project from which we want the images",
                    default=198)
parser.add_argument('--slice_term',type=int,
                    help="id of the ROI delimiting annotation",
                    default=8760)

parser.add_argument('--download_path',
                    help="Where to store images",
                    default='/home/donovan/Downloads/')

parser.add_argument('--patch-size',type=int,default=1024)
parser.add_argument('--overlap',type=int,default=2)

params=parser.parse_args(sys.argv[1:])

w=params.patch_size
h=params.patch_size

host=params.host
public_key=params.public_key
private_key=params.private_key
id_project=params.id_project
img_path=params.download_path
slice_term_id=params.slice_term
imgs=get_image_map(params)
print(imgs)
overlap=params.overlap
datadir=params.download_path

imgs_l=[]
dest_base=params.download_path

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
        print("ID: {} | Img: {} | Pjct: {} | Term: {} ".format(
            annotation.id,
            annotation.image,
            annotation.project,
            annotation.term
        ))
        if len(annotation.term)==1:
            if (annotation.term[0],annotation.image) not in res.keys():
                res[(annotation.term[0],annotation.image)]=[]
            res[(annotation.term[0],annotation.image)].append(loads(annotation.location))
            last=res[(annotation.term[0],annotation.image)][-1]
            print(last.bounds,last.to_wkt().count(","))

ks=res.keys()
print(ks)
kl=[x for x in ks]
stuff=np.array(kl)[:,0]
stuff=stuff.tolist()
stuff=set(stuff)
print(stuff)
terms=[x for x in stuff]
terms.remove(slice_term_id)
print(terms)

with Cytomine(host=host,public_key=public_key,private_key=private_key) as cytomine:
    image_instances= ImageInstanceCollection().fetch_with_filter("project",id_project)
    
    for image in image_instances:
        if image.id in imgs.keys():
            print("Image ID: {} | Width: {} | Height: {} | Resolution: {} | Magnification: {} | Filename: {}".format(
                image.id, image.width, image.height, image.resolution, image.magnification, image.filename
            ))
            print(os.path.join(img_path,imgs[int(image.id)]))
            img=osl.OpenSlide(os.path.join(img_path,imgs[int(image.id)]))
            terms_annot={}
            for term in terms:
                anots=[]
                res_term_img=None
                try:
                    res_term_img=res[(term,image.id)]
                except:
                    print("no annotation with term {} in image {}".format(term,image.id))
                    continue

                for anotloc in res_term_img:
                    anot='{}'.format(
                        anotloc.svg(fill_color='#ffffff'))
                    anots.append(anot)
                terms_annot[term]=anots
            try:
                res_term_img=res[(slice_term_id,image.id)]
            except:
                print("no annotation with term {} in image {}".format(slice_term_id,image.id))
                continue
            bs=[np.array(box.bounds,dtype=np.int) for box in res_term_img]
            for num_slide in range(len(bs)):
                imgs_l.append("{}_{}".format(image.id,num_slide))
            num=0
            for b in bs:
                if os.path.exists(os.path.join(datadir,"{}_{}.jpg".format(image.id,num))):
                    print("got {}_{}.jpg".format(image.id,num))
                else:
                    crop=img.read_region((b[0],img.dimensions[1]-b[3]),0,(b[2]-b[0],b[3]-b[1]))
                    crop=crop.convert("RGB")
                    print("saving image {} {}".format(image.id,num))
                    crop.save(os.path.join(datadir,"{}_{}.jpg".format(image.id,num)))

                for term in terms:
                    if os.path.exists(os.path.join(datadir,"{}_{}_{}.png".format(image.id,num,term))):
                        print("got {}_{}_{}.png".format(image.id,num,term))
                        continue
                    s='<svg width="{}" height="{}" viewBox="{} {} {} {}"><rect x="{}" y="{}" width="{}" height="{}" style="fill:rgb(0,0,0);" />'
                    sf=s.format(
                        #width,height
                        b[2]-b[0],
                        b[3]-b[1],
                        #viewbox
                        b[2],
                        b[3],
                        b[0]-b[2],
                        b[1]-b[3],
                        #rectangle background
                        b[2],
                        b[3],
                        b[0]-b[2],
                        b[1]-b[3])
                    for anot in terms_annot[term]:
                        sf='{}<g transform="translate({},{}), scale(1,-1)">{}</g>'.format(sf,0,b[1]+b[3],anot)#.format(sf,b[0]+b[2],anot)
                    sf="{}{}".format(sf,"</svg>")
                    print("saving term {}".format(term))
                    cs.svg2png(bytestring=sf.replace('opacity="0.6"','opacity="1.0"'),
                               write_to=os.path.join(datadir,"{}_{}_{}.png".format(image.id,num,term)))
                num+=1

for term in terms:
    try:
        os.mkdir(os.path.join(datadir,str(term)))
    except:
        pass

try:
    os.mkdir(os.path.join(datadir,'crop'))
except:
    pass

#  293 725 184 pixels
from PIL import Image
Image.MAX_IMAGE_PIXELS=1000000000

for img in imgs_l:
    #crop=cv2.imread("{}.jpg".format(img))
    print("doing ",img)
    im= Image.open(os.path.join(datadir,"{}.jpg".format(img)))
    width,height= im.size
    crop=None
    masks={}
    #for term in terms:
    #    masks[term]=cv2.imread("{}_{}.png".format(img,term))
    #    print("-",term,masks[term].shape)
    for i in range(0,height,h//overlap):
        for j in range(0,width,w//overlap):
            if not os.path.exists(os.path.join(datadir,"crop","{}_{}_{}_{}_{}.jpg".format(img,i,j,h,w))):
                if crop is None:
                    crop=cv2.imread(os.path.join(datadir,"{}.jpg".format(img)))
                    print("loaded ",img," with size " ,crop.shape)
                cv2.imwrite(os.path.join(datadir,"crop","{}_{}_{}_{}_{}.jpg".format(img,i,j,h,w)),crop[i:i+h,j:j+w,:])

            for term in terms:
                if not os.path.exists(os.path.join(datadir,str(term),"{}_{}_{}_{}_{}.png".format(img,i,j,h,w))):
                    print("{}/{}/{}_{}_{}_{}_{}.png".format(datadir,term,img,i,j,h,w))
                    try:
                        cv2.imwrite(os.path.join(datadir,str(term),"{}_{}_{}_{}_{}.png".format(img,i,j,h,w)),masks[term][i:i+h,j:j+w,:])
                    except KeyError:
                        masks[term]=cv2.imread(os.path.join(datadir,"{}_{}.png".format(img,term)))
                        cv2.imwrite(os.path.join(datadir,str(term),"{}_{}_{}_{}_{}.png".format(img,i,j,h,w)),masks[term][i:i+h,j:j+w,:])

                else:
                    print("{}/{}/{}_{}_{}_{}_{}.png already".format(datadir,term,img,i,j,h,w))
    print("")

