import abc
import tensorflow as tf


class BaseAttention(abc.ABC):
  """Abstract base class of attention.

  C.f. https://github.com/GSimas/Deep-LearningAI/blob/master/Course%205/Week%203/Neural%20Machine%20Translation/images/attn_mechanism.png?raw=true  # noqa:E501
  """

  # Abbreviations for shapes:
  #   batch_shape -> B (list of `int`s)
  #   seqlen -> L (`int`)
  #   query_dim -> Q (`int`)
  #   key_dim -> K (`int`)
  #   value_dim -> V (`int`)

  def __call__(self, query, keys, values, name='Attention'):
    """Returns the context, with attention-score.

    Args:
      query: Tensor with shape `batch_shape + [query_dim]`.
      keys: Tensor with shape `batch_shape + [seqlen, key_dim]`.
      values: Tensor with shape `batch_shape + [seqlen, value_dim]`.
      name: String.

    Returns:
      Tuple of two tensors. The first with shape `batch_shape + [value_dim]`
      as the context; and the second with shape `batch_shape + [seqlen]` as
      the attention-score.
    """
    with tf.name_scope(name):
      score = self._score(query, keys)  # B + [L, 1]
      # score * values: B + [L, V]
      context = tf.reduce_sum(score * values, axis=-2)  # B + [V]
      score = tf.squeeze(score, axis=-1)  # B + [L]
      return score, context

  def _score(self, query, keys, name='attention_score'):
    """Returns the attention-score.

    Args:
      query: Tensor with shape `batch_shape + [query_dim]`.
      keys: Tensor with shape `batch_shape + [seqlen, key_dim]`.
      name: String.

    Returns:
      Tensor with shape `batch_shape + [seqlen, 1]`. The additional `1`
      is made for convienence for being contracted by a "values" tensor
      with shape `batch_shape + [seqlen, value_dim]` along the `seqlen`
      axis.
    """
    with tf.name_scope(name):
      with tf.name_scope('repeat'):  # along `seqlen`-axis.
        query = tf.expand_dims(query, axis=-2)  # B + [1, Q]
        # Compute the `repeats` argument in `tf.tile()`
        shape = keys.get_shape().as_list()
        rank = len(shape)
        seqlen = shape[-2]
        multiples = [1] * (rank - 2) + [seqlen, 1]
        query = tf.tile(query, multiples)  # B + [L, Q]

      concated = tf.concat([query, keys], axis=-1)  # B + [L, Q+K]
      energy = self.energy(concated)  # B + [L, 1]
      # Softmax along the `L`-axis
      attention_score = tf.nn.softmax(energy, axis=-2)  # B + [L, 1]
      return attention_score

  @abc.abstractmethod
  def energy(self, x):
    """
    Args:
      x: Tensor with shape `batch_shape + [query_dim + key_dim]`.

    Returns:
      Tensor with shape `batch_shape + [1]`.
    """
    pass
