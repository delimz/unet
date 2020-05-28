import tensorflow as tf
from .utils import crop_size

def leg(input,concat,num_filters=1,name="leg"):
        upped = tf.keras.layers.UpSampling2D()(input)
        upped = tf.keras.layers.Cropping2D((upped.shape[1]-concat.shape[1])//2)(upped)
        merged = tf.keras.layers.Concatenate()([upped,concat])
        return merged

def unet(x,num_class,
         depth=4,
         conv=2,
         filters_factor=16,
         filters=None,
         dilations=None,
         residuals=True,
         front_leg=False,
         back_leg=False,
         **kwargs):
    encoder_blocks=[]
    encoder_pool=[]
    decoder_blocks=[]
    residual_convs_encoder=[]
    res_bn_encoder=[]
    res_acc_encoder=[]
    residual_convs_decoder=[]
    res_bn_decoder=[]
    res_acc_decoder=[]
    transconvs=[]
    residuals=residuals
    crops=[]
    rescrops=[]
    resadds=[]
    derescrops=[]
    deresadds=[]
    center=[]

    if filters is None:
        filters=[3 for i in range(depth+1)]

    elif len(filters)!=depth+1:
        raise Exception('depth and filter do not match  {} {} {}'.format(filters,dilations,conv))
    
    if dilations is None:
        dilations=[1 for i in range(depth+1)]
    elif len(dilations)!=depth+1:
        raise Exception('dilations and depth do not match  {} {} {}'.format(filters,dilations,conv))
        
    if isinstance(conv,int):
        convs=[conv for i in range(depth+1)]
    else:
        convs=conv
    if len(convs)!=depth+1:
        print(depth,convs,dilations,filters)
        raise Exception('conv does not match depth {} {} {}'.format(filters,dilations,conv))
    
    ###layers declaration
    for i in range(depth):
        conv=convs[i]
        encoder=[]
        rescrops.append(tf.keras.layers.Cropping2D((filters[i]-1)*dilations[i]//2*conv))
        for j in range(conv):
            encoder.append(tf.keras.layers.Conv2D(filters_factor*2**(i),
                                                  filters[i],dilation_rate=dilations[i],
                                                  name='enc_conv_%d_%d'%(i,j)))
            encoder.append(tf.keras.layers.BatchNormalization())
            encoder.append(tf.keras.layers.ELU())
        encoder_blocks.append(encoder)
        if residuals:
            residual_convs_encoder.append(tf.keras.layers.Conv2D(filters_factor*2**(i),1,
                                                                 name='enc_res_conv_%d'% i))
            res_bn_encoder.append(tf.keras.layers.BatchNormalization())
            res_acc_encoder.append(tf.keras.layers.ELU())
            resadds.append(tf.keras.layers.Add())
        else:
            residual_convs_encoder.append(None)
            res_bn_encoder.append(None)
            res_acc_encoder.append(None)
            resadds.append(None)
        
        encoder_pool.append(tf.keras.layers.MaxPool2D())
        
    i=depth
    if residuals:
        residual_conv_center=tf.keras.layers.Conv2D(filters_factor*2**(depth),1,name='cen_res_conv')
        residual_conv_bn=tf.keras.layers.BatchNormalization()
        residual_conv_activation=tf.keras.layers.ELU()
        rescrops_center=tf.keras.layers.Cropping2D((filters[i]-1)*dilations[i]//2*conv)
        resadd_center=tf.keras.layers.Add()
    for j in range(convs[i]):
        center.append(tf.keras.layers.Conv2D(filters_factor*2**(i),filters[i],
                                             dilation_rate=dilations[i],name='cen_conv_%d'%j))
        center.append(tf.keras.layers.BatchNormalization())
        center.append(tf.keras.layers.ELU())
    
    for i in reversed(list(range(depth))):
        derescrops.append(tf.keras.layers.Cropping2D((filters[i]-1)*dilations[i]//2*conv))
        deresadds.append(tf.keras.layers.Add())
        decoder = []
        transconvs.append(tf.keras.layers.Conv2DTranspose(filters_factor*2**(i),2,
                                                          strides=(2,2),name='tra_conv_%d'%i))
        
        
        #decoder.append(tf.keras.layers.BatchNormalization())
        #decoder.append(tf.keras.layers.ELU())
        
        for j in range(conv):
            decoder.append(tf.keras.layers.Conv2D(filters_factor*2**(i),filters[i],
                                                  dilation_rate=dilations[i],name='dec_conv_%d_%d'%(i,j)))
            decoder.append(tf.keras.layers.BatchNormalization())
            decoder.append(tf.keras.layers.ELU())
        
        
        decoder_blocks.append(decoder)
        if residuals:
            residual_convs_decoder.append(tf.keras.layers.Conv2D(filters_factor*2**(i),1,
                                                                 name='dec_res_conv_%d'%i))
            res_bn_decoder.append(tf.keras.layers.BatchNormalization())
            res_acc_decoder.append(tf.keras.layers.ELU())
        else:
            residual_convs_decoder.append(None)
            res_bn_decoder.append(None)
            res_acc_decoder.append(None)
            
    final=tf.keras.layers.Conv2D(num_class,1,activation='sigmoid',name='fin_conv')
    ###
    encoded=[]
    for block,respass,resbn,resacc,rescrop,add,pool in zip(encoder_blocks,
                                                  residual_convs_encoder,
                                                  res_bn_encoder,
                                                  res_acc_encoder,
                                                  rescrops,
                                                  resadds,
                                                  encoder_pool):
        print(x.shape)
        if residuals:
            res=rescrop(resacc(resbn(respass(x))))
        for layer in block:
            x=layer(x)
        if residuals:
            x=add([res,x])
        encoded.append(x)
        x=pool(x)
    if residuals:
        res=residual_conv_center(x)
        res=residual_conv_bn(res)
        res=residual_conv_activation(res)
        res=rescrops_center(res)
    
    for layer in center:
        x=layer(x)
    if residuals:
        x=resadd_center([res,x])
    
    backleg=x
    for block,respass,resbn,resacc,copy,rescrop,add,transconv in zip(decoder_blocks,
                                                     residual_convs_decoder,
                                                     res_bn_decoder,
                                                     res_acc_decoder,
                                                     reversed(encoded),
                                                     derescrops,
                                                     deresadds,
                                                     transconvs):
        x=transconv(x)
        x=tf.keras.layers.BatchNormalization()(x)
        x=tf.keras.layers.ELU()(x)
        crop=tf.keras.layers.Cropping2D((copy.shape[1]-x.shape[1])//2)
        print(x.shape)
        print(copy.shape)
        x=tf.keras.layers.concatenate([crop(copy),x])
        if residuals:
            res=rescrop(resacc(resbn(respass(x))))
        for layer in block:
            x=layer(x)
        if residuals:
            x=add([res,x])
        if back_leg:
            backleg=leg(backleg,x)
            
            
    if back_leg:
        x=backleg
    output=final(x)

    return output


