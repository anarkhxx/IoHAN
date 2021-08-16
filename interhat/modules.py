# -*- coding: utf-8 -*-
#/usr/bin/python2

'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

from __future__ import print_function
import tensorflow as tf
import numpy as np




# dictionary = np.load('dictionary_3digit.pkl', allow_pickle=True)
# glove1 = np.load('glove_100dim.7.npz', allow_pickle=True)
# glove2 = np.load('gram_test.79.npz', allow_pickle=True)
# tree = np.load('tree.types', allow_pickle=True)
# # glove = (glove1['w'] + glove1['w_tilde']) / 2
# glove = glove2['W_emb']
# dic = list(dictionary.values())


# 将tree.types里的医疗代码整理成dictionary
# def get_top_three(list1):
#     j = 0
#     if j < 4894:
#         for i in list1:
#             if list1[j] != 'D_':
#                 if i[2] == 'E':
#                     list1[j] = i[:6]
#                 else:
#                     list1[j] = i[:5]
#             j = j + 1
#     return list1
#
#
# a = np.array(get_top_three(list(tree.keys())))
#
# rng = np.random.RandomState(1234)
#
#
# def get_emb(list2, list3):
#     j = 0
#     if j < 1072:
#         for i in list2:
#             b = np.where(list3 == i)[0]  # [233,3,4]
#             if len(b) != 0:
#                 emb_ave = np.zeros([100])
#                 for n in b:
#                     emb_ave += glove[n]
#                 emb_ave = emb_ave / len(b)
#                 list2[j] = list(emb_ave)
#                 j += 1
#             else:
#                 list2[j] = list(np.asarray(rng.uniform(low=-0.1, high=0.1, size=100)))
#                 j += 1
#     return list2
#
#
# l = get_emb(dic, a)


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.
    
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs


def embedding(inputs, 
              vocab_size, 
              num_units, 
              zero_pad=False,
              last_pad=False,
              scale=True,
              scope="embedding", 
              reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.
        
    For example,
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]    
    ```    
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # lookup_table = tf.Variable(l)
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        # 没办法，我是采用最大的整形作为填充的
        if last_pad:
            lookup_table = tf.concat((lookup_table[:-1, :],
                                      tf.zeros(shape=[1, num_units])), 0)

        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        if scale:
            outputs = outputs * (num_units ** 0.5)

    return outputs
    

def positional_encoding(inputs,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):
    '''Sinusoidal Positional_Encoding.

    Args:
      inputs: A 2d Tensor with shape of (N, T).
      num_units: Output dimensionality
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
        A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    '''

    N, T = inputs.get_shape().as_list()
    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units**0.5

        return outputs


def multihead_attention(queries,
                        keys, 
                        num_units=None, 
                        num_heads=8, 
                        dropout_rate=0,
                        is_training=True,
                        scope="multihead_attention", 
                        reuse=None):
    '''Applies multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size, the C
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    '''
    queries=features,
    keys=features,
    num_units=self.attention_size*self.num_head（30*12）,
    num_heads=self.num_head,
    dropout_rate=self.dropout_rate,
    is_training=self.is_training,
    scope="multihead_attention"
    '''

    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        #12
        emb_size = keys.get_shape().as_list()[-1]
        
        # Linear projections
        #num_units=attention_size*num_head（30*12)
        #==》结果360个神经元后 变成(?, 1638, 360)
        #inputs：输入该网络层的数据
        #units：输出的维度大小，改变inputs的最后一维
        #activation：激活函数，即神经网络的非线性变化
        #use_bias：使用bias为True（默认使用），不用bias改成False即可，是否使用偏置项
        #trainable=True:表明该层的参数是否参与训练。如果为真则变量加入到图集合中

        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        #(?,1638,360)  ==> (12*?,1638,30)
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        #tf.transpose(K_, [0, 2, 1])将 (12*?,1638,30) 变成 (12*?,30,1638)
        #(12*?,1638,30) matmul (12*?,30,1638) ==> (12*?,1638,1638)
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        #由于attention_size=30，所以是outputs/5.477225575051661。
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        
        # Key Masking
        #那啥，原本的keys是(?,1638,embedding_size) ==> (?,1638)
        #如果x < 0,则有 y = sign(x) = -1；如果x == 0,则有 0 或者tf.is_nan(x)；如果x > 0,则有1.
        #所以0或者1了。
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)


        #tile的例子
        #a=[1 2 1 2 1 2]
        #b = tf.tile(a,[2,3])
        #b=[[1 2 1 2 1 2]
        #   [3 4 3 4 3 4]
        #   [1 2 1 2 1 2]
        #   [3 4 3 4 3 4]]
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        #反正变成和outpus一样的shape
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)

        #(12*?,1638,1638)结构，但是都是极小值
        paddings = tf.ones_like(outputs)*(-2**32+1)
        #如果outputs里头有0替换padding，否则替换outputs原本数据。
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)

        cross_significance = outputs
         
        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        # (?, 1638, 1638)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        #(?, 1638, 30)
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)


        # Restore shape

        #(?, 1638, 360)
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        #(?, 1638, 12)
        outputs = tf.layers.dense(outputs, emb_size, activation=tf.nn.relu)

        # Residual connection
        #(?, 1638, 12)
        outputs += queries

        # Normalize
        outputs = normalize(outputs)  # (N, T_q, C)


    return outputs, cross_significance


def feedforward(inputs, 
                num_units=[2048, 512],
                scope="multihead_attention", 
                reuse=None):
    '''Point-wise feed forward net.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        #一维卷积
        '''
        inputs = tf.placeholder('float', shape=[None, 6, 8])
        out = tf.layers.conv1d(inputs, 5, 3)
        说明： 对于一个样本而言，句子长度为6个字，字向量的维度为8
        filters=5, kernel_size=3， 所以卷积核的维度为3*8
        那么输入6*8经过3*8的卷积核卷积后得到的是4*1的一个向量(4=6-3+1)
        又因为有5个过滤器，所以是得到5个4*1的向量
        '''
        #(?, 1638, 48)  ==> 卷核是1638*1, ==>所以可以得到1638*1的向量，由于filter是48个==> 1638 * 48
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        ##(?, 1638, 12) ==> 卷核是1638*1, ==>所以可以得到1638*1的向量，由于filter是12个==> 1638 * 12
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs
        # Normalize
        outputs = normalize(outputs)

    return outputs


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.
    
    For example,
    
    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1], 
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)
       
    outputs = label_smoothing(inputs)
    
    with tf.Session() as sess:
        print(sess.run([outputs]))
    
    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
    ```    
    '''
    K = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / K)

