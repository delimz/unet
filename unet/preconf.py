conf={}
conf['net'] = {
        "name" : "net",
    "back_leg" : False,
    "residuals" : True,
    "depth" : 4,
    "conv" : [2,2,2,2,2],
    "filters" : [3,3,3,3,3],
    "dilations" : [3,2,2,1,1]
}
conf['unet'] = {
        "name" : "unet",
    "back_leg" : False,
    "residuals" : False,
    "depth" : 4,
    "conv" : [2,2,2,2,2],
    "filters" : [3,3,3,3,3],
    "dilations" : [1,1,1,1,1]
}
conf['vnet'] = {
        "name" : "vnet",
    "back_leg" : False,
    "residuals" : True,
    "depth" : 4,
    "conv" : [2,2,2,2,2],
    "filters" : [7,5,5,3,3],
    "dilations" : [1,1,1,1,1]
}
conf['myunet'] = {
        "name" : "myunet",
    "back_leg" : False,
    "residuals" : False,
    "depth" : 5,
    "conv" : [3,3,3,3,3,3],
    "filters" : [3,3,3,3,3,3],
    "dilations" : [1,1,1,1,1,1]
}
conf['myvnet_res'] = {
        "name" : "myvnet_res",
    "back_leg" : False,
    "residuals" : True,
    "depth" : 5,
    "conv" : [3,3,3,3,3,3],
    "filters" : [7,7,5,5,3,3],
    "dilations" : [1,1,1,1,1,1],
}
conf['myvnet'] = {
        "name" : "myvnet",
    "back_leg" : False,
    "residuals" : False,
    "depth" : 5,
    "conv" : [3,3,3,3,3,3],
    "filters" : [7,7,5,5,3,3],
    "dilations" : [1,1,1,1,1,1],
}
conf['mysegnet'] = {
        "name" : "mysegnet",
    "back_leg" : False,
    "residuals" : False,
    "depth" : 5,
    "conv" : [3,3,3,3,3,3],
    "filters" : [3,3,3,3,3,3],
    "dilations" : [1,1,1,1,1,1]
}
conf['mymnet'] = {
        "name" : "mymnet",
    "back_leg" : True,
    "residuals" : False,
    "depth" : 5,
    "conv" : [3,3,3,3,3,3],
    "filters" : [3,3,3,3,3,3],
    "dilations" : [1,1,1,1,1,1]
}
