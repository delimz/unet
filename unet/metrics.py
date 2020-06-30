import tensorflow as tf

class Dice(tf.keras.metrics.Metric):
    """ compute the total average Dice of the batch 

    """
    def __init__(self,name='dice',**kwargs):
        super(Dice,self).__init__(name=name,**kwargs)
        self.intersection=self.add_weight(name='intersection',initializer='ones')
        self.union=self.add_weight(name='union',initializer='ones')

    def update_state(self,y_true,y_pred,sample_weight=None):
        self.intersection.assign_add(2*tf.reduce_sum(y_true*y_pred))
        self.union.assign_add(tf.reduce_sum(y_true+y_pred))

    def result(self):
        return self.intersection / self.union


class DiceX(tf.keras.metrics.Metric):
    """ compute the individual class average Dice of the batch 

    """
    def __init__(self,num=0,name='dice',**kwargs):
        name='dice{}'.format(num)
        super(DiceX,self).__init__(name=name,**kwargs)
        self.intersection=self.add_weight(name='intersection',initializer='ones')
        self.union=self.add_weight(name='union',initializer='ones')
        self.num=num

    def update_state(self,y_true,y_pred,sample_weight=None):
        self.intersection.assign_add(2*tf.reduce_sum(y_true[:,:,:,self.num]*y_pred[:,:,:,self.num]))
        self.union.assign_add(tf.reduce_sum(y_true[:,:,:,self.num]+y_pred[:,:,:,self.num]))

    def result(self):
        return self.intersection / self.union

class DiceG(tf.keras.metrics.Metric):
    """ compute the geometric mean of per-class dice

    """
    def __init__(self,num=1,name='geo_dice',**kwargs):
        super(DiceG,self).__init__(name=name,**kwargs)
        self.intersection=self.add_weight(name='intersection',shape=(num,),initializer='ones')
        self.union=self.add_weight(name='union',shape=(num,),initializer='ones')
        self.num=num

    def update_state(self,y_true,y_pred,sample_weight=None):
        self.intersection.assign_add(2*tf.reduce_sum(y_true*y_pred,axis=(0,1,2)))
        self.union.assign_add(tf.reduce_sum(y_true+y_pred,axis=(0,1,2)))

    def result(self):
        return tf.pow(tf.reduce_prod(self.intersection / self.union),1/self.num)

    def reset_states(self):
        self.intersection.assign([1 for i in range(self.num)])
        self.union.assign([1 for i in range(self.num)])

