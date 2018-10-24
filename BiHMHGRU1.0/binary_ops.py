import tensorflow as tf
from tensorflow.python.framework import ops

###################################
###################################
# Binary Neurons Operation
# Following code is adapted from:
# https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
###################################
###################################

def hard_sigmoid(x,ann_slope):
  
  slope = tf.constant(0.5,shape=x.get_shape())

  slope = ann_slope * slope

  shift = tf.constant(0.5,shape=x.get_shape())

  x = (x*slope) + shift

  return tf.clip_by_value(x,0.0,1.0)


###################
# BINARY ROUND: For deterministic binary units
###################
def binaryRound(x):
    """
    Rounds a tensor whose values are in [0,1] to a tensor with values in {0, 1}, 
    using the straight through estimator for the gradient.
    
    E.g.,:
    If x is >= 0.5, binaryRound(x) will be 1 and the gradient will be pass-through,
    otherwise, binaryRound(x) will be 0 and the gradient will be 0.
    """
    g = tf.get_default_graph()
    
    with ops.name_scope("BinaryRound") as name:
        # override "Floor" because tf.round uses tf.floor
        with g.gradient_override_map({"Round": "Identity"}):
            return tf.round(x, name=name)

###################
# BERNOULLI SAMPLING: for stochastic binary units
###################
def bernoulliSample(x):
    """
    Uses a tensor whose values are in [0,1] to sample a tensor with values in {0, 1},
    using the straight through estimator for the gradient.
    
    E.g.,:
    if x is 0.6, bernoulliSample(x) will be 1 with probability 0.6, and 0 otherwise,
    and the gradient will be pass-through (1) wih probability 0.6, and 0 otherwise. 
    """
    g = tf.get_default_graph()
    
    with ops.name_scope("BernoulliSample") as name:
        with g.gradient_override_map({"Ceil": "Identity","Sub": "BernoulliSample_ST"}):
            return tf.ceil(x - tf.random_uniform(tf.shape(x)), name=name)
        
@ops.RegisterGradient("BernoulliSample_ST")
def bernoulliSample_ST(op, grad):
    """Straight through estimator for the bernoulliSample op (identity if 1, else 0)."""
    sub = op.outputs[0] # x - tf.random_uniform... 
    res = sub.consumers()[0].outputs[0] # tf.ceil(sub)
    return [res * grad, tf.zeros(tf.shape(op.inputs[1]))]

###################
# PASS-THROUGH SIGMOID: a sigmoid with identity as derivative.
###################
def passThroughSigmoid(x):
    """Sigmoid that uses identity function as its gradient"""
    g = tf.get_default_graph()
    with ops.name_scope("PassThroughSigmoid") as name:
        with g.gradient_override_map({"Sigmoid": "Identity"}):
            return tf.sigmoid(x, name=name)

###################
# BINARY STOCHASTIC NEURONS DEFINITION
###################   
def binaryStochastic_ST(x, slope_tensor=None, pass_through=True, stochastic=True):
    """
    Sigmoid followed by either a random sample from a bernoulli distribution according 
    to the result (binary stochastic neuron) (default), or a sigmoid followed by a binary
    step function (if stochastic == False). Uses the straight through estimator. 
    See https://arxiv.org/abs/1308.3432.
    
    Arguments:
    * x: the pre-activation / logit tensor
    * slope_tensor: if passThrough==False, slope adjusts the slope of the sigmoid function 
        for purposes of the Slope Annealing Trick (see http://arxiv.org/abs/1609.01704)
    * pass_through: if True (default), gradient of the entire function is 1 or 0; 
        if False, gradient of 1 is scaled by the gradient of the sigmoid (required if
        Slope Annealing Trick is used)
    * stochastic: binary stochastic neuron if True (default), or step function if False
    """
    if slope_tensor is None:
        slope_tensor = tf.constant(1.0)

    if pass_through:
        p = passThroughSigmoid(x)
    else:
        p = hard_sigmoid(x,slope_tensor)
    
    if stochastic:
        return bernoulliSample(p)
    else:
        return binaryRound(p)

###################
# Binary operation wrapper
###################

def binary_wrapper(pre_activations_tensor,
                   stochastic_tensor=tf.constant(True), 
                   pass_through=True, 
                   slope_tensor=tf.constant(1.0)):
    """
    Turns a layer of pre-activations (logits) into a layer of binary stochastic neurons
    
    Keyword arguments:
    *stochastic_tensor: a boolean tensor indicating whether to sample from a bernoulli 
        distribution (True, default) or use a step_function (e.g., for inference)
    *pass_through: for ST only - boolean as to whether to substitute identity derivative on the 
        backprop (True, default), or whether to use the derivative of the sigmoid
    *slope_tensor: tensor specifying the slope for purposes of slope annealing
        trick
    """

    # When pass_through = True, the straight-through estimator is used.
    # Binary units can be stochastic or deterministc.
    if pass_through:
        return tf.cond(stochastic_tensor, 
                       lambda: binaryStochastic_ST(pre_activations_tensor), 
                       lambda: binaryStochastic_ST(pre_activations_tensor, stochastic=False))

    # When pass_through = False, during backprop the derivative of the binary activations is substituted
    # with the derivative of the sigmoid function, multiplied by the slope (slope-annealing trick).
    # Again, binary units can be stochastic or deterministic.
    else:
        return tf.cond(stochastic_tensor, 
                       lambda: binaryStochastic_ST(pre_activations_tensor, slope_tensor = slope_tensor, 
                                                   pass_through=False), 
                       lambda: binaryStochastic_ST(pre_activations_tensor, slope_tensor = slope_tensor, 
                                                   pass_through=False, stochastic=False))