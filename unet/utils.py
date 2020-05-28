
def crop_size(filters,dilations,convs):
    """ 
    crop_size take the lists of filters sizes, dilations sizes and the number of convolution in each encoder
    to compute the number of pixels that will be lost by lack of context

    """
    total=0
    depth=0
    for f,d,c in zip(filters,dilations,convs):
        for i in range(c):
            total+=((f-1)*d//2)*(2**depth)
        depth+=1
    depth-=1
    for f,d,c in reversed(list(zip(filters,dilations,convs))[1:]):
        depth-=1
        for i in range(c):
            total+=((f-1)*d//2)*(2**depth)
    return total


