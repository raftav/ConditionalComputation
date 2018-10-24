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
###################################
###################################
# Binary Neurons Operation
# Following code is adapted from:
# https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
###################################
###################################

def hard_sigmoid(x,ann_slope):

  #slope = tf.constant(0.5,shape=tf.shape(x))
  slope = 0.5*tf.ones_like(x)

  slope = ann_slope * slope

  #shift = tf.constant(0.5,shape=x.get_shape())
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

    #TODO hard sigmoid:
    # z_tilda = tf.maximum(0, tf.minimum(1, (slope * z_t_logit) / 2))
  if pass_through:
    p = passThroughSigmoid(x)  # TODO hard sigmoid? pass through is typically used when we don't do slope annealing
  else:
    p = hard_sigmoid(x,slope_tensor) # TODO hard sigmoid
    
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
# Here we define the HM LSTM
######################################

_HmLstmStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h", "z"))

_HmLstmOutputTuple = collections.namedtuple("LSTMOutputTuple", ("h", "z", "z_tilda"))

class HmLstmStateTuple(_HmLstmStateTuple):
  """Tuple used by HmLstm Cells for `state_size`, `zero_state`, and output state.

  Stores three elements: `(c, h, z)`, in that order.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (c, h, z) = self
    if not c.dtype == h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(c.dtype), str(h.dtype)))
    return c.dtype


class HmLstmOutputTuple(_HmLstmOutputTuple):
  """Tuple used by HardGated Cells for `state_size`, `zero_state`, and output state.

  Stores two elements: `(h, z)`, in that order.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (h, z , z_tilda) = self
    return h.dtype


class HmLstmCell(tf.contrib.rnn.RNNCell):

  slope=None

  def __init__(self, num_units):
    # self._num_units determines the size of c and h.
    self._num_units = num_units

  # this property defines the size of the cell:
  # we have c,h,z here.
  @property
  def state_size(self):
    return HmLstmStateTuple(self._num_units, self._num_units, 1)

  @property
  def output_size(self):
    return HmLstmOutputTuple(self._num_units, 1 ,1 )

  # layer normalization
  def _norm(self,inp, scope , norm_gain=1.0, norm_shift=0.0):
    shape = inp.get_shape()[-1:]
    gamma_init = init_ops.constant_initializer(norm_gain)
    beta_init = init_ops.constant_initializer(norm_shift)

    with vs.variable_scope(scope):
      # Initialize beta and gamma for use by layer_norm.
      vs.get_variable("gamma", shape=shape, initializer=gamma_init)
      vs.get_variable("beta", shape=shape, initializer=beta_init)
    normalized = layers.layer_norm(inp, reuse=True, scope=scope)
    return normalized


  def __call__(self, inputs, state, scope=None):
    # vars from different layers.
    h_bottom, z_bottom, h_top_prev = inputs
    # vars from the previous time step on the same layer
    c_prev, h_prev, z_prev = state

    # Need enough rows in the shared matrix for f, i, o, g, z_stochastic_tilda
    num_rows = 4 * self._num_units + 1

    # scope: optional name for the variable scope, defaults to "HmLstmCell"
    with vs.variable_scope(scope or type(self).__name__):  # "HmLstmCell"
      # Matrix U_l^l
      U_curr = vs.get_variable("U_curr", [h_prev.get_shape()[1], num_rows], dtype=tf.float32,
                                          initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))
      # Matrix U_{l+1}^l
      # TODO This imples that the U matrix there has the same dimensionality as the
      # one used in equation 5. but that would only be true if you forced the h vectors
      # on the above layer to be equal in size to the ones below them. Is that a real restriction?
      # Or am I misunderstanding?
      U_top = vs.get_variable("U_top", [h_bottom.get_shape()[1], num_rows], dtype=tf.float32,
                              initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))
      # Matrix W_{l-1}^l
      W_bottom = vs.get_variable("W_bottom", [h_bottom.get_shape()[1], num_rows],dtype=tf.float32,
                                  initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))
      # b_l
      #bias = vs.get_variable("bias", [num_rows], dtype=tf.float32,initializer=tf.ones_initializer())

      s_curr = tf.matmul(h_prev, U_curr)
      #RT test
      #s_curr = (tf.ones_like(z_prev) - z_prev) * tf.matmul(h_prev, U_curr)
      s_top = z_prev * tf.matmul(h_top_prev, U_top)
      s_bottom = z_bottom * tf.matmul(h_bottom, W_bottom)
      gate_logits = s_curr + s_top + s_bottom # + bias

      f_logits = tf.slice(gate_logits, [0, 0], [-1, self._num_units])
      i_logits = tf.slice(gate_logits, [0, self._num_units], [-1, self._num_units])
      o_logits = tf.slice(gate_logits, [0, 2*self._num_units], [-1, self._num_units])
      g_logits = tf.slice(gate_logits, [0, 3*self._num_units], [-1, self._num_units])
      z_t_logit = tf.slice(gate_logits, [0, 4*self._num_units], [-1, 1])
      
      f_logits = self._norm(f_logits,"flush",norm_shift=1.0)
      f = tf.sigmoid(f_logits)

      i_logits = self._norm(i_logits,"input",norm_shift=0.0)
      i = tf.sigmoid(i_logits)

      o_logits = self._norm(o_logits,"output",norm_shift=0.0)
      o = tf.sigmoid(o_logits)

      g_logits = self._norm(g_logits,"cell_propr",norm_shift=0.0)
      g = tf.tanh(g_logits)

      # This is the stochastic neuron
      z_new = binary_wrapper(z_t_logit,
                             pass_through=False, 
                             stochastic_tensor=tf.constant(False), # TODO make this false if you do slope annealing
                             slope_tensor=self.slope)

      z_zero_mask = tf.equal(z_prev, tf.zeros_like(z_prev))
      copy_mask = tf.to_float(tf.logical_and(z_zero_mask, tf.equal(z_bottom, tf.zeros_like(z_bottom))))
      update_mask = tf.to_float(tf.logical_and(z_zero_mask, tf.cast(z_bottom, tf.bool)))
      flush_mask = z_prev

      
      c_flush = i * g
      c_update = f * c_prev + c_flush      
      c_new = copy_mask * c_prev + update_mask * c_update + flush_mask * c_flush

      h_flush = o * tf.tanh(c_flush)
      h_update = o * tf.tanh(c_update)
      h_new = copy_mask * h_prev + update_mask * h_update + flush_mask * h_flush
      
    state_new = HmLstmStateTuple(c_new, h_new, z_new)
    output_new = HmLstmOutputTuple(h_new, z_new,z_t_logit)

    return output_new, state_new



# The output for this is a list of h_vectors, one for each cell.
class MultiHmLstmCell(tf.contrib.rnn.RNNCell):
  def __init__(self, cells):
    """Create a RNN cell composed sequentially of a number of HmRNNCells.

    Args:
      cells: list of HmRNNCells that will be composed in this order.
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

  # 'inputs' should be a batch of word vectors
  # 'state' should be a list of HM cell state tuples of the same length as self._cells
  # 'slope' is a scalar tensor for slope annealing.
  def __call__(self, inputs, state, scope=None):
    """Run this multi-layer cell on inputs, starting from state."""
    assert len(state) == len(self._cells)

    with vs.variable_scope(scope or type(self).__name__):  # "MultiHmRNNCell"
      # if there is more than one layer, assign h_prev_top
      # otherwise this will be zero.
      if len(self._cells) > 1:
        h_prev_top = state[1].h
      else:
        h_prev_top = tf.zeros(state[0].h.get_shape())

      # h_bottom, z_bottom, h_prev_top
      current_input = inputs, tf.ones([inputs.get_shape()[0], 1]), h_prev_top

      new_h_list = []
      new_states = []
       
      # Go through each cell in the different layers, going bottom to top
      for i, cell in enumerate(self._cells):

        with vs.variable_scope("Cell%d" % i):
          ##new_h, new_state = cell(current_input, state[i] ,slope=slope) # state[i] = c_prev, h_prev, z_prev
          new_h, new_state = cell(current_input, state[i])

          # Set up the inputs for the next cell.
          if i < len(self._cells) - 2:
            # Next cell is not the top one.
            h_prev_top = state[i+2].h
          else:
            # The next cell is the top one, so give it zeros for its h_prev_top input.
            h_prev_top = tf.zeros(state[i].h.get_shape())

          current_input = new_state.h, new_state.z, h_prev_top  # h_bottom, z_bottom, h_prev_top

          new_h_list.append(new_h)
          new_states.append(new_state)

    return tuple(new_h_list), tuple(new_states)




