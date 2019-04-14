import tensorflow as tf
from hellonlp.attention.base import BaseAttention


class SimpleAttention(BaseAttention):

  def __init__(self, energy_units, **kwargs):
    super().__init__(**kwargs)
    self._layers = []
    for n in energy_units:
      self._layers += [tf.layers.Dense(n, activation=tf.nn.relu),
                       tf.layers.Dropout()]
    self._layers += [tf.layers.Dense(1)]

  def energy(self, x):
    with tf.name_scope('energy'):
      for layer in self._layers:
        x = layer(x)
      return x
