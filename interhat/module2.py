import tensorflow as tf


def recur_attention(queries,
                    keys,
                    values,
                    attention_size,
                    scope,
                    reuse=None,
                    regularize_scale=None,
                    ):
    """Single attention

    :param queries: 2-D Tensor, shape=[N, C], query vector
    :param keys: 3-D Tensor, shape=[N, T, emb_size], key tensor
    :param values: 3-D Tensor, shape=[N, T, emb_size], value tensor
    :param regularize: Boolean. Do regularization or not.
    :param regularize_scale: float
    :return:
    """
    # print("query shape", queries.get_shape())
    # _, T, C = keys.get_shape().as_list()
    _, T, _ = keys.get_shape().as_list()
    C = attention_size

    if regularize_scale:
        regularizer = tf.contrib.layers.l2_regularizer(scale=regularize_scale)
    else:
        regularizer = None

    initializer = tf.contrib.layers.xavier_initializer()

    keys = tf.layers.dense(keys, attention_size,
                           activation=tf.nn.relu)  # [N, T, a_s]
    # values = tf.layers.dense(values, attention_size,
    #                          activation=tf.nn.relu)  # [N, T, a_s]
    queries = tf.layers.dense(queries, attention_size,
                              activation=tf.nn.relu)  # [N, T, a_s]

    # ** Here C = a_s **
    with tf.variable_scope(scope, reuse=reuse):
        # W: T * C
        W = tf.get_variable(name="prev_order_cross",
                            dtype=tf.float32,
                            shape=(T * C),
                            initializer=initializer,
                            regularizer=regularizer)

        # b: C
        b = tf.get_variable(name="single_attn_bias",
                            dtype=tf.float32,
                            shape=(C),
                            initializer=initializer,
                            regularizer=regularizer)

        # vector to make it scalar
        # h: C
        h = tf.get_variable(name="single_attn_h",
                            dtype=tf.float32,
                            shape=(C),
                            initializer=initializer,
                            regularizer=regularizer)

        """
        Math equation of a_i
            a_i = h^T . RELU(W * outer(Q, K[i]) + b)
        """

        # outer(Q, K[i])
        kq_outer = tf.reshape(
            # [N, T, C] mult [N, 1, C]
            tf.multiply(keys, queries), # [N, T, C]
            shape=[-1, T * C]
        )  # (N, T * C)

        # relu(W * outer(Q, k[i]) + b)

        linear_activation = tf.nn.relu(
            tf.reshape(
                tf.multiply(kq_outer, 
                            tf.expand_dims(W, axis=0)),  # [N, T*C]
                shape=[-1, T, C])  # [N, T, C]
            + tf.reshape(b, shape=[1, 1, -1])  # (1, 1, C)
        )   # (N, T, C)

        # h^T relu (W * outer(Q, k[i]) + b)

        h_ = tf.reshape(h, shape=[1, 1, C])
        attention_factor = tf.reduce_sum(
            tf.multiply(linear_activation, h_),
            axis=-1
        )  # (N, T)

        attention_factor = tf.expand_dims(
            attention_factor,
            axis=1
        )  # (N, T) to (N, 1, T)

        weighted_value = tf.matmul(attention_factor, values)

    return weighted_value


def agg_attention(query,
                  keys,
                  values,
                  attention_size,
                  regularize_scale=None):
    """

    :param query: [a_s]
    :param keys: [N, T, dim]
    :param values: [N, T, dim]
    :param attention_size: [attn_size]
    :param regularize_scale:
    :return:
    """
    if regularize_scale:
        regularizer = tf.contrib.layers.l2_regularizer(scale=regularize_scale)
    else:
        regularizer=None



    #(?, 1638, 12==>3)
    # project keys to attention space
    projected_keys = tf.layers.dense(keys, attention_size,
                                     activation=tf.nn.relu,
                                     kernel_regularizer=regularizer,
                                     bias_regularizer=regularizer)  # [N, T, a_s]
    # reshape query
    query_ = tf.reshape(query, [1, 1, -1])  # [1, 1, a_s]

    # multiply query_, keys (broadcast)
    #(?, 1638) = rf.reduce_sum((?,1638,3) multiply  (1,1,3))
    attention_energy = tf.reduce_sum(
        tf.multiply(projected_keys, query_),  # [N, T, a_s]
        axis=2
    )  # [N, T]

    # generate attention weights
    attentions = tf.nn.softmax(logits=attention_energy,
                               name="attention")  # [N, T]
    #这部分就是将可能的权重加到features ==> [?,1638,12]*[?,1638,1]  上
    results = tf.reduce_sum(
        tf.multiply(values,
                    tf.expand_dims(attentions, axis=2)),  # [N, T, dim]
        axis=1
    )  # [N, dim]
    return results, attentions


