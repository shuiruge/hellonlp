import tensorflow as tf


class BaseGloVe(object):
  r"""
  Description
  -----------
    Even though both employing shallow neural network, they have different
    tasks to do:

    * The word2vec model is aiming at the task that predicts if two words are
      in neighbour. The prediction is either `False` or `True`.
    * And GloVe model pushes further, aiming at the task that predicts the co-
      occurance counts of the two words being in neighbour. The prediction is a
      quantative number.

    So, GloVe is expected to perform better than word2vec.

    ```math
    Firstly, it is illustrated that, not the $P(x_i \mid x_j)$, but the

    \begin{equation}
      \frac{ P(x_i \mid x_k) }{ P(x_j \mid x_k) }
    \end{equation}

    for arbitrary context word $x_k$ that recovers the relationship between
    $x_i$ and $x_j$. C.f. the Table 1 of the GloVe paper.

    Let $X_{ij}$ the co-occurance counts of the two word $x_i$ and $x_j$ being
    in neighbour in the corpus. The total counts of word $x_i$ thus is
    $\sum_j X_{ij}$. The probablity $P(x_i \mid x_j)$ is thus $X_{ij} / X_j$.
    For every word $x_i$, represent it by vectors $w_i$, and by $\tilde{w}_i$
    when treated as context (i.e. the $\text{word}_k$). We are to find the
    vectors for every word so as to fit

    \begin{equation}
      F\left[ \left( w_i - w_j \right) \cdot \tilde{w}_k \right] =
        \frac{ P(x_i \mid x_k) }{ P(x_j \mid x_k) }
    \end{equation}

    The most general form of $F$ is $F(x) = \exp \{a x + b\}$.

    By some simplification including symbolic re-definitions, we get the
    fitting target

    \begin{equation}
      \left( w_i - w_j \right) \cdot \tilde{w}_k + (b_i - b_j) \to
        \ln \{ P(x_i \mid x_k) \} - \ln \{ P(x_j \mid x_k) \},
    \end{equation}

    which can be further simplified to a symmetric form:

    \begin{equation}
      w_i \cdot \tilde{w}_k + b_i + \tilde{b}_k \to \ln X_{ik}.
    \end{equation}

    Noticing the symmetry between the $w$ and the $\tilde{w}$ in the final
    form, after the training, the final embedding matrix is

    \begin{equation}
      w_{\text{final}} = \frac{w + \tilde{w}}{2}.
    \end{equation}
    ```

  References
  ----------
    1. [GloVe paper](https://nlp.stanford.edu/pubs/glove.pdf)

  Code Notations
  --------------
    `w0` is the :math:`w` in the paper, and `w1` the :math:`\tilde{w}`.
    Both have the shape `[vocabulary_size, embed_dimension]`. `X` is the
    word-word co-occurance counts :math:`X_{ij}` in the paper.
  """

  @staticmethod
  def get_loss(weight_fn, w0, w1, b0, b1,
               distance=lambda x, y: tf.square(x - y)):
    r"""Returns the loss-function for the GloVe model.

    Args:
      weight_fn: Callable from tensor to tensor, with the same shape and
        dtype, as the :math:`f()` in the paper.
      w0: Tensor with shape `[vocab_size, embed_dim]`, as the :math:`w` in the
        paper.
      w1: Tensor with shape `[vocab_size, embed_dim]`, as the :math:`\tilde{w}`
        in the paper.
      b0: Tensor with shape `[vocab_size]`, as the :math:`b` in the paper.
      b1: Tensor with shape `[vocab_size]`, as the :math:`\tilde{b}` in the
        paper.
      distance: Callable from two tensors a tensor, all with the same shape
        and dtype.

    Returns:
      Callable with signature:
        Args:
          j: Tensor with shape `[batch_size]` and `int` dtype.
          k: Tensor with shape `[batch_size]` and `int` dtype.
          X_jk: Tensor with shape `[batch_size]` and `int` dtype.
        Returns:
          Scalar.
    """

    def loss(i, j, X_ij, name='glove_loss'):
      """Implements the GloVe loss, i.e. the eq.(8) in the GloVe paper.

      Args:
        i: Tensor with shape `[batch_size]` and `int` dtype.
        j: Tensor with shape `[batch_size]` and `int` dtype.
        X_ij: Tensor with shape `[batch_size]` and `int` dtype.
      Returns:
        Scalar.
      """
      with tf.name_scope(name):
        # The loss is for fitting the prediction and the target,
        # with the distance between them measured by `distance`,
        # and weighted by `f()`.

        # All the following computed tensors are by batch.
        # Shape-notations:
        #   B: batch-size;
        #   E: embedding dimension.

        with tf.name_scope('predict'):
          w0_i = tf.gather(w0, i)  # [B, E]
          w1_j = tf.gather(w1, j)  # [B, E]
          b0_i = tf.gather(b0, i)  # [B]
          b1_j = tf.gather(b1, j)  # [B]
          predict = (
              tf.reduce_sum(w0_i * w1_j, axis=1) +  # inner-product.
              b0_i + b1_j)  # [B]

        with tf.name_scope('target'):
          with tf.name_scope('X_ij'):
            # Unify the dtype
            X_ij = tf.cast(X_ij, w0.dtype)
            # For numerical stability in log()
            X_ij = X_ij + 1
          target = tf.log(X_ij)  # [B]

        with tf.name_scope('weight'):
          weight = weight_fn(X_ij)  # [B]
          weight = tf.cast(weight, w0.dtype)

        with tf.name_scope('weighted_loss'):
          loss_ij = weight * distance(predict, target)  # [B]
          return tf.reduce_mean(loss_ij)  # []

    return loss
