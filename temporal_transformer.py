import pdb
import tensorflow as tf

def transformer(U, theta, out_size, name='tt'):
  """Temporal transformer

  Attempts to perform temporal localisation on temporal/sequential data
  instead of spatial data like in the Deepmind STN.

  Reference
  ---------
  [1] https://github.com/tensorflow/models/blob/master/transformer/spatial_transformer.py
  [2] Spatial Transformer Network
      Max Jaberberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
      Google Deepmind
      https://arxiv.org/pdf/1506.02025.pdf

  #TODOS
  - padding at the end
  """


  def _repeat(x, n_repeats):
    with tf.variable_scope('repeat'):
      rep = tf.transpose(
              tf.expand_dims(tf.ones(shape=tf.stack([n_repeats,])),1), [1,0])
      rep = tf.cast(rep, 'int32')
      x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
      return tf.reshape(x, [-1])


  def _extract(seq, x, out_size):
    with tf.variable_scope('_extract'):
      num_batch = tf.shape(seq)[0]
      timestep = tf.shape(seq)[1]
      channel = tf.shape(seq)[2]

      x = tf.cast(x, 'float32')
      timestep_f = tf.cast(timestep, 'float32')
      zero = tf.zeros([], dtype='int32')
      max_timestep = tf.cast(timestep-1, 'int32')

      #scale indices from [-1,1] to [0, timesteps]
      x = (x+1) * timestep_f / 2.0

      #sample
      x0 = tf.cast(tf.floor(x), 'int32')
      x1 = x0 + 1

      x0 = tf.clip_by_value(x0, zero, max_timestep)
      x1 = tf.clip_by_value(x1, zero, max_timestep)

      base = _repeat(tf.range(num_batch)*timestep, out_size) # assume equal timestep
      idx_a = base + x0
      idx_b = base + x1

      seq_flat = tf.reshape(seq, tf.stack([-1, channel]))
      seq_flat = tf.cast(seq_flat, 'float32')
      Ia = tf.gather(seq_flat, idx_a)
      Ib = tf.gather(seq_flat, idx_b)

      x0_f = tf.cast(x0, 'float32')
      x1_f = tf.cast(x1, 'float32')
      wa = tf.expand_dims((x1_f-x), 1)
      wb = tf.expand_dims((x0_f-x), 1)
      output = tf.add_n([wa*Ia, wb*Ib])

      return output


  def _transform(seq, theta, out_size):
    with tf.variable_scope('_transform'):
      num_batch = tf.shape(seq)[0]
      timestep = tf.shape(seq)[1]
      channel = tf.shape(seq)[2]
      theta = tf.reshape(theta, (-1, 1, 2)) # CHECK
      theta = tf.cast(theta, 'float32')

      #get grid
      timestep_f = tf.cast(timestep, 'float32')
      grid = _meshgrid(out_size)
      grid = tf.expand_dims(grid, 0)
      grid = tf.reshape(grid, [-1])
      grid = tf.tile(grid, tf.stack([num_batch]))
      grid = tf.reshape(grid, tf.stack([num_batch, 2, -1]))

      T_g = tf.matmul(theta, grid)
      x_s = tf.slice(T_g, [0,0,0], [-1,1,-1])
      x_s_flat = tf.reshape(x_s, [-1])

      input_transformed = _extract(seq, x_s_flat, out_size)

      output = tf.reshape(input_transformed, tf.stack([num_batch, out_size, channel]))
      return output

  def _meshgrid(outsize):
    """
    gets the (x, 1) grid to be multiplied w theta transform
    """
    x_t = tf.expand_dims(tf.linspace(-1.0, 1.0, out_size), 1)
    ones = tf.ones_like(x_t)
    grid = tf.concat(axis=0, values=[x_t, ones])
    return grid

  with tf.variable_scope(name):
    output = _transform(U, theta, out_size)
    return output
