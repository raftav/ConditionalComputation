import collections
import tensorflow as tf

import rnn_ops
from binary_ops import binary_wrapper
import layer_norm as layers

from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops




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

    # self._num_units determines the size of c and h.
    self._num_units = num_units

  # this property defines the size of the cell state
  # carries h and z
  @property
  def state_size(self):
    return HardGatedStateTuple(self._num_units, 1 )

  # the output carries h
  @property
  def output_size(self):
    return HardGatedOutputTuple(self._num_units, 1 ,1 )

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

  # All RNNs return (output, new_state). For this type of LSTM, its 'output' is still its h vector,
  # and it's cell state is a h,z tuple.
  # 'inputs' are a tuple of h_bottom, z_bottom, and h_top_prev.
  # 'state' is a h,z HardGateStateTuple
  def __call__(self, inputs, state, scope=None):
    # vars from different layers.
    h_bottom, z_bottom, h_top_prev = inputs
    # vars from the previous time step on the same layer
    h_prev, z_prev  = state

    # scope: optional name for the variable scope, defaults to "HGCell"
    with vs.variable_scope(scope or type(self).__name__):  # "HGCell"

      ################
      # UPDATE MODULE
      ################
      initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1)

      with tf.variable_scope('update_module'):
        with tf.variable_scope('reset_gate'):
          r = rnn_ops.linear([h_bottom, h_prev], self._num_units, bias=False,weights_init=initializer)

          r = self._norm(r,"reset",norm_shift=1.0)
          r = tf.sigmoid(r)

        u = rnn_ops.linear([h_bottom,r*h_prev],self._num_units,bias=False,weights_init=initializer)
        u = self._norm(u,"update")
        u=tf.tanh(u)

      with tf.variable_scope('flush_module'):
        f = rnn_ops.linear([h_bottom,h_top_prev],self._num_units,bias=False,weights_init=initializer)
        f = self._norm(f,"flush")
        f = tf.tanh(f)

      with tf.variable_scope('gate_module'):
        z_logits = rnn_ops.linear([h_prev,h_bottom],1,bias=True,bias_start=0.5,weights_init=initializer)
        z_logits = z_bottom * z_logits

        z_new = binary_wrapper(z_logits,
                             pass_through=False, 
                             stochastic_tensor=tf.constant(False),
                             slope_tensor=self.slope)

      h_new = (tf.ones_like(z_new) - z_new) * ( (tf.ones_like(z_bottom) - z_bottom) * h_prev + z_bottom * u ) + z_new * f

    state_new = HardGatedStateTuple(h_new, z_new)
    output_new = HardGatedOutputTuple(h_new, z_new,z_logits)


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
      current_input = inputs, tf.ones([inputs.get_shape().as_list()[0], 1]), h_prev_top

      new_h_list = []
      new_states = []

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




