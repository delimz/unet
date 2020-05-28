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


