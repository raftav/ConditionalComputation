import collections
import tensorflow as tf

from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import init_ops

from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.util import nest
from tensorflow.python.framework import dtypes

def variable_summaries(var,var_name):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(var_name+'_summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

###################################
###################################
# Binary Neurons Operation
# Following code is adapted from:
# https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
###################################
###################################

def hard_sigmoid(x,ann_slope):
  
  slope = 0.5*tf.ones_like(x)

  slope = ann_slope * slope

  shift = 0.5 * tf.ones_like(x)

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

######################################
# Here we define the HardGated RNN
######################################

##########################################
# Tuple that will be passed as cell state
##########################################
_HardGatedStateTuple = collections.namedtuple("HardGatedStateTuple", ("h", "z"))

##########################################
# Tuple that will be passed as cell output
##########################################
_HardGatedOutputTuple = collections.namedtuple("HardGatedOutputTuple", ("h", "z", "z_tilda"))


class HardGatedStateTuple(_HardGatedStateTuple):
  """Tuple used by HardGated Cells for `state_size`, `zero_state`, and output state.

  Stores two elements: `(h, z)`, in that order.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (h, z ) = self
    return h.dtype

class HardGatedOutputTuple(_HardGatedOutputTuple):
  """Tuple used by HardGated Cells for `state_size`, `zero_state`, and output state.

  Stores two elements: `(h, z)`, in that order.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (h, z , z_tilda) = self
    return h.dtype

#########################
# HGRU Cell definition
#########################  
class HGCell(tf.contrib.rnn.RNNCell):

  slope=None

  def __init__(self, num_units):
    self._num_units = num_units

  @property
  def state_size(self):
    return HardGatedStateTuple(self._num_units, 1 )

  @property
  def output_size(self):
    return HardGatedOutputTuple(self._num_units, 1 ,1 )

  def _norm(self,inp, scope , norm_gain=1.0, norm_shift=0.0):
    shape = inp.get_shape()[-1:]
    gamma_init = init_ops.constant_initializer(norm_gain)
    beta_init = init_ops.constant_initializer(norm_shift)

    with vs.variable_scope(scope):
      vs.get_variable("gamma", shape=shape, initializer=gamma_init)
      vs.get_variable("beta", shape=shape, initializer=beta_init)
    normalized = layers.layer_norm(inp, reuse=True, scope=scope)
    return normalized

  def __call__(self, inputs, state, scope=None):

    h_bottom, z_bottom, h_top_prev = inputs
    h_prev, z_prev  = state

    with vs.variable_scope(scope or type(self).__name__):

      ################
      # UPDATE MODULE
      ################

      # Matrix U_l^l
      U_curr = vs.get_variable("U_curr", [h_prev.get_shape()[1], self._num_units], dtype=tf.float32,
                                          initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))
      # Matrix U_{l-1}^l
      U_bottom = vs.get_variable("U_bottom", [h_bottom.get_shape()[1], self._num_units], dtype=tf.float32,
                              initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))

      # Matrix R_{l-1}^l
      R_bottom = vs.get_variable("R_bottom", [h_bottom.get_shape()[1], self._num_units], dtype=tf.float32,
                              initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))

      #Matrix R_l^l
      R_curr = vs.get_variable("R_curr", [h_prev.get_shape()[1], self._num_units], dtype=tf.float32,
                              initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))

      with tf.name_scope('reset_input') as input_scope:
        reset_input = tf.add(tf.matmul(h_bottom,R_bottom),tf.matmul(h_prev,R_curr),name=input_scope)


      reset_input = self._norm(reset_input,"reset",norm_shift=1.0)

      with tf.name_scope('reset_input_rescaled') as input_scope:
        reset_input = tf.identity(reset_input,name=input_scope)

      # reset gate as in GRU
      reset_gate = tf.sigmoid( reset_input )

      with tf.name_scope('update_input') as input_scope:
        u_input = tf.add(tf.matmul(h_bottom,U_bottom),tf.matmul(tf.multiply(reset_gate,h_prev),U_curr),name=input_scope)

      u_input = self._norm(u_input,"update")
      with tf.name_scope('update_input_rescaled') as input_scope:
        u_input=tf.identity(u_input,name=input_scope)

      # u_t^l : essentially a GRU
      u_t = tf.tanh( u_input )

      ################
      # FLUSH MODULE
      ################

      # Matrix W_{l+1}^l
      W_top = vs.get_variable("W_top", [h_top_prev.get_shape()[1], self._num_units], dtype=tf.float32,
                              initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))

      # Matrix W_{l-1}^l
      W_bottom = vs.get_variable("W_bottom", [h_bottom.get_shape()[1], self._num_units], dtype=tf.float32,
                                initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))

      with tf.name_scope('flush_input') as input_scope:
        flush_input = tf.add(tf.matmul(h_top_prev,W_top),tf.matmul(h_bottom,W_bottom),name=input_scope)

      flush_input = self._norm(flush_input,"flush")
      with tf.name_scope('flush_input_rescaled') as input_scope:
        flush_input = tf.identity(flush_input,name=input_scope)

      # f_t^l : the flush module
      f_t = tf.tanh( flush_input )

      ##################
      # BINARY UNIT
      ##################

      # Matrix V_l^l
      V_curr = vs.get_variable("V_curr", [h_prev.get_shape()[1], 1 ], dtype=tf.float32,
                              initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))

      V_bottom = vs.get_variable("V_bottom", [h_bottom.get_shape()[1], 1 ], dtype=tf.float32,
                              initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1)) 

      bias_z = vs.get_variable("bias_z", shape=[1],dtype=tf.float32,
      	                       initializer=tf.ones_initializer())

      with tf.name_scope('gates_input') as input_scope:
        z_tilda_input = tf.add(tf.matmul(h_prev,V_curr),tf.matmul(h_bottom,V_bottom),name=input_scope)
        z_tilda_input = tf.add(z_tilda_input,bias_z)

      z_tilda_input = z_bottom * z_tilda_input

      with tf.name_scope('gates_input_rescaled') as input_scope:
        z_tilda_logits = tf.identity(z_tilda_input,name=input_scope)

      z_new =  binary_wrapper(z_tilda_logits,
                             pass_through=False, 
                             stochastic_tensor=tf.constant(False),
                             slope_tensor=self.slope)

      #################
      # HIDDEN LAYER
      #################

      h_new = (tf.ones_like(z_new) - z_new) * ( (tf.ones_like(z_bottom) - z_bottom) * h_prev + z_bottom * u_t ) + z_new * f_t

    state_new = HardGatedStateTuple(h_new, z_new)
    output_new = HardGatedOutputTuple(h_new, z_new,z_tilda_logits)

    return output_new, state_new


#########################
# Vertical Stack of HGRU Cells
#########################

class MultiHGRNNCell(tf.contrib.rnn.RNNCell):
  def __init__(self, cells):
    """Create a RNN cell composed sequentially of a number of HGRNNCells.

    Args:
      cells: list of HGRNNCells that will be composed in this order.
    """
    if not cells:
      raise ValueError("Must specify at least one cell for MultiHmRNNCell.")
    self._cells = cells

  @property
  def state_size(self):
    return tuple(cell.state_size for cell in self._cells)

  @property
  def output_size(self):
    return tuple(cell.output_size for cell in self._cells)


  def __call__(self, inputs, state, scope=None):
    """Run this multi-layer cell on inputs, starting from state."""
    assert len(state) == len(self._cells)
    with vs.variable_scope(scope or type(self).__name__):  # "MultiHmRNNCell"

      # assign h_prev_top considering the special case of only one hidden layer
      if len(self._cells) > 1:
        h_prev_top = state[1].h
      else:
        h_prev_top = tf.zeros(tf.shape(state[0].h))

      # set inputs. 
      # here the gates at the first layer are set to 1.
      #input_boundaries = tf.get_variable('input_z',shape=[inputs.get_shape()[1], 1],initializer=tf.ones_initializer())
      #current_input = inputs, input_boundaries, h_prev_top
      current_input = inputs, tf.ones([tf.shape(inputs)[0], 1]), h_prev_top

      new_h_list = []
      new_states = []
       
      # Go through each cell in the different layers, going bottom to top
      # place cells in different devices

      for i, cell in enumerate(self._cells):
        with vs.variable_scope("Cell%d" % i):

          new_h, new_state = cell(current_input, state[i])

          # Set up the inputs for the next cell.
          if i < len(self._cells) - 2:
            # Next cell is not the top one.
            h_prev_top = state[i+2].h
          else:
            # The next cell is the top one, so give it zeros for its h_prev_top input.
            h_prev_top = tf.zeros(tf.shape(state[i].h))

          # update input  
          current_input = new_state.h, new_state.z, h_prev_top  # h_bottom, z_bottom, h_prev_top
          
          #save outputs and states
          new_h_list.append(new_h)
          new_states.append(new_state)

      # return a tuple with the activation of all the hidden layers
      output = tuple(new_h_list)

    return output, tuple(new_states)




