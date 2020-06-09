import os
from cytomine import Cytomine
from cytomine.models.image import ImageInstanceCollection
from cytomine.models.annotation import AnnotationCollection

from shapely.wkt import loads
import numpy as np

def get_image_map(params):
    with Cytomine(host=params.host, public_key=params.public_key, private_key=params.private_key) as cytomine:
        image_instances = ImageInstanceCollection().fetch_with_filter("project", params.id_project)
        res=dict()
        for image in image_instances:
            filename=image.filename
            res[image.id]=filename
            if params.download_path:
                # To download the original files that have been uploaded to Cytomine
                path=os.path.join(params.download_path, filename)
                print(path)
                if not os.path.exists(path):
                    image.download(path)
        return res

def get_terms_list(params):
    with Cytomine(host=params.cytomine_host, public_key=params.cytomine_public_key, private_key=params.cytomine_private_key) as cytomine:
        res={}
        annotations = AnnotationCollection()
        annotations.project = params.cytomine_id_project
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
    terms.remove(params.slice_term)
    print(terms)
    return terms



if __name__ == "__main__":
    
    from argparse import ArgumentParser
    import sys

    parser = ArgumentParser(prog="ratseg_get_data")
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
    parser.add_argument('--download_path',
                        help="Where to store images",
                        default='/home/donovan/Downloads/')

    params=parser.parse_args(sys.argv[1:])

    for k in get_image_map(params).keys():
        print(k,end=" ")
    print("")

