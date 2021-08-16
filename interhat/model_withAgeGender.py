'''
给用户添加了年龄，和性别特征，来做预测。。
'''
import tensorflow as tf

from const import Constant
from modules import multihead_attention, feedforward, embedding
from module2 import agg_attention


# ===== InterpRecSys Base Model =====
class InterprecsysBase:

    def __init__(self
                 , embedding_dim
                 , field_size
                 , feature_size
                 , learning_rate
                 , batch_size
                 , num_block
                 , num_head
                 , attention_size
                 , pool_filter_size
                 , dropout_rate
                 , regularization_weight
                 , random_seed=Constant.RANDOM_SEED
                 , scale_embedding=False
                 ):
        # config parameters
        self.embedding_dim = embedding_dim  # the C
        self.scale_embedding = scale_embedding  # bool
        self.field_size = field_size
        self.feat_size = feature_size  # the T

        self.dropout_rate = dropout_rate
        self.random_seed = random_seed
        self.num_block = num_block  # num of blocks of multi-head attn
        self.num_head = num_head  # num of heads
        self.attention_size = attention_size
        self.regularization_weight = regularization_weight
        self.pool_filter_size = pool_filter_size

        # training parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # ===== Create None variables for object =====
        # variables [None]
        self.embedding_lookup = None
        self.emb = None  # raw features

        # placeholders
        self.X_ind, self.X_val, self.label = None, None, None
        self.is_training = None

        # ports to the outside
        self.sigmoid_logits = None
        self.regularization_loss = None
        self.logloss, self.mean_logloss = None, None
        self.overall_loss = None

        # train/summary operations
        self.train_op, self.merged = None, None

        # intermediate results
        self.feature_weights = None
        self.sigmoid_logits = None

        # global training steps
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        # operations
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                beta1=0.9,
                                                beta2=0.98,
                                                epsilon=1e-8)

        self.build_graph()

    def build_graph(self):
        # Define input
        with tf.name_scope("input_ph"):
            # 普通占位符：batchsize*特征数
            self.X_ind = tf.placeholder(dtype=tf.int32,
                                        shape=[None, self.field_size],
                                        name="X_index")
            self.label = tf.placeholder(dtype=tf.float32,
                                        shape=[None],
                                        name="label")
            self.is_training = tf.placeholder(dtype=tf.bool,
                                              shape=(),
                                              name="is_training")
        #添加了age和gender两个特征，最后两个就是
        self.fuck_ =tf.split(value=self.X_ind, num_or_size_splits=[33,2], axis=1)
        self.X_ind_then=self.fuck_[0]
        self.ageGender=self.fuck_[1]

        # lookup and process embedding。就是随机向量一下
        with tf.name_scope("embedding"):
            self.emb = embedding(inputs=self.X_ind_then,
                                 last_pad=True,
                                 vocab_size=self.feat_size,
                                 num_units=self.embedding_dim,
                                 scale=self.scale_embedding,
                                 scope="embedding_process")

        # self.emb: raw embedding, features: used for later
        # 这tm就是个map,feat_size个code对应的向量，维度为embedding_dim
        # batchsize*feat_size*embedding_size
        # (?,1638,12)
        features = self.emb
        features_split_1, features_split_2, features_split_3 = \
            tf.split(value=features, num_or_size_splits=3, axis=1)

        with tf.name_scope("Multilayer_attn"):
            with tf.variable_scope("attention_head") as scope:
                # features, _ = multihead_attention(
                #     queries=features,
                #     keys=features,
                #     num_units=self.attention_size*self.num_head,
                #     num_heads=self.num_head,
                #     dropout_rate=self.dropout_rate,
                #     is_training=self.is_training,
                #     scope="multihead_attention"
                # )
                #
                # features = feedforward(
                #     inputs=features,
                #     num_units=[4 * self.embedding_dim,
                #                self.embedding_dim],
                #     scope="feed_forward"
                # )  # [N, T, dim]
                features_split_1, _1 = multihead_attention(
                    queries=features_split_1,
                    keys=features_split_1,
                    num_units=self.attention_size * self.num_head,
                    num_heads=self.num_head,
                    dropout_rate=self.dropout_rate,
                    is_training=self.is_training,
                    scope="multihead_attention1"
                )

                features_split_1 = feedforward(
                    inputs=features_split_1,
                    num_units=[4 * self.embedding_dim,
                               self.embedding_dim],
                    scope="feed_forward1"
                )
                features_split_2, _2 = multihead_attention(
                    queries=features_split_2,
                    keys=features_split_2,
                    num_units=self.attention_size * self.num_head,
                    num_heads=self.num_head,
                    dropout_rate=self.dropout_rate,
                    is_training=self.is_training,
                    scope="multihead_attention2"
                )

                features_split_2 = feedforward(
                    inputs=features_split_2,
                    num_units=[4 * self.embedding_dim,
                               self.embedding_dim],
                    scope="feed_forward2"
                )
                features_split_3, _3 = multihead_attention(
                    queries=features_split_3,
                    keys=features_split_3,
                    num_units=self.attention_size * self.num_head,
                    num_heads=self.num_head,
                    dropout_rate=self.dropout_rate,
                    is_training=self.is_training,
                    scope="multihead_attention3"
                )

                features_split_3 = feedforward(
                    inputs=features_split_3,
                    num_units=[4 * self.embedding_dim,
                               self.embedding_dim],
                    scope="feed_forward3"
                )
                # features=tf.concat([features_split_1,features_split_2],1)

        # multi-head feature to agg 1st order feature
        with tf.name_scope("Agg_first_order") as scope:
            ctx_order_1 = tf.get_variable(
                name="context_order_1",
                shape=(self.attention_size),
                dtype=tf.float32)

            agg_feat_1, self.attn_1 = agg_attention(
                query=ctx_order_1,
                keys=features_split_1,
                values=features_split_1,
                attention_size=self.attention_size,
                regularize_scale=self.regularization_weight
            )  # [N, dim]
        with tf.name_scope("Agg_second_order") as scope:
            ctx_order_2 = tf.get_variable(
                name="context_order_2",
                shape=(self.attention_size),
                dtype=tf.float32)
            agg_feat_2, self.attn_2 = agg_attention(
                query=ctx_order_2,
                keys=features_split_2,
                values=features_split_2,
                attention_size=self.attention_size,
                regularize_scale=self.regularization_weight
            )  # [N, dim]
        with tf.name_scope("Agg_third_order") as scope:
            ctx_order_3 = tf.get_variable(
                name="context_order_3",
                shape=(self.attention_size),
                dtype=tf.float32)
            agg_feat_3, self.attn_3 = agg_attention(
                query=ctx_order_3,
                keys=features_split_3,
                values=features_split_3,
                attention_size=self.attention_size,
                regularize_scale=self.regularization_weight
            )  # [N, dim]

        print("look something")
        #此处可以添加上LSTM试试1
        cell1=tf.contrib.rnn.BasicLSTMCell(num_units=50,state_is_tuple=False)
        test_output1,test_laststate1=tf.nn.dynamic_rnn(cell=cell1,inputs=tf.stack([agg_feat_1],axis=1),dtype=tf.float32)
        agg_feat_1=test_laststate1
        #2
        #cell2 = tf.contrib.rnn.BasicLSTMCell(num_units=50, state_is_tuple=False)
        test_output2, test_laststate2 = tf.nn.dynamic_rnn(cell=cell1, inputs=tf.stack([agg_feat_2], axis=1),
                                                          dtype=tf.float32)
        agg_feat_2 = test_laststate2
        #3
        #cell3 = tf.contrib.rnn.BasicLSTMCell(num_units=50, state_is_tuple=False)
        test_output3, test_laststate3 = tf.nn.dynamic_rnn(cell=cell1, inputs=tf.stack([agg_feat_3], axis=1),
                                                          dtype=tf.float32)
        agg_feat_3 = test_laststate3

        feat_visit_all = tf.stack([agg_feat_1, agg_feat_2, agg_feat_3], axis=1, name="visit_all")
        with tf.name_scope("Agg_visit_order") as scope:
            # 现在整合两个visit
            all_order = tf.get_variable(
                name="all_order",
                shape=(self.attention_size),
                dtype=tf.float32)
            agg_all, self.attn_k = agg_attention(
                query=all_order,
                keys=feat_visit_all,
                values=feat_visit_all,
                attention_size=self.attention_size,
                regularize_scale=self.regularization_weight
            )  # [N, dim]

        # build second order cross
        # with tf.name_scope("Second_order") as scope:
        #     feat_2 = tf.multiply(
        #         features,
        #         tf.expand_dims(agg_feat_1, axis=1)
        #         )  # [N, T, dim]
        #
        #     feat_2 += features  # Add the residual, [N, T, dim]
        #
        #     ctx_order_2 = tf.get_variable(
        #         name="context_order_2",
        #         shape=(self.attention_size),
        #         dtype=tf.float32
        #         )
        #
        #     agg_feat_2, self.attn_2 = agg_attention(
        #         query=ctx_order_2,
        #         keys=feat_2,
        #         values=feat_2,
        #         attention_size=self.attention_size,
        #         regularize_scale=self.regularization_weight
        #         )
        #
        # # build third order cross
        # with tf.name_scope("Third_order") as scope:
        #     feat_3 = tf.multiply(
        #         features,
        #         tf.expand_dims(agg_feat_2, axis=1)
        #         )  # [N, T, dim]
        #
        #     feat_3 += feat_2  # Add the residual, [N, T, dim]
        #
        #     ctx_order_3 = tf.get_variable(
        #         name="context_order_3",
        #         shape=(self.attention_size),
        #         dtype=tf.float32
        #         )
        #
        #     agg_feat_3, self.attn_3 = agg_attention(
        #         query=ctx_order_3,
        #         keys=feat_3,
        #         values=feat_3,
        #         attention_size=self.attention_size,
        #         regularize_scale=self.regularization_weight
        #         )

        with tf.name_scope("Merged_features"):
            # concatenate [enc, second_cross, third_cross]
            # TODO: can + multihead_features
            # (?,3,12)
            all_features = tf.stack([
                #agg_all,
                tf.concat([agg_all,tf.to_float(self.ageGender)],1)
                # agg_feat_1,
                # agg_feat_2,
                # agg_feat_3,
            ],
                axis=1, name="concat_feature")  # (N, k, C)
        # map C to pool_filter_size dimension
        # kernel_size=1,所以有batchsize*3，然后pool_filter_size个过滤器==> (batchsize,3,pf_size)
        mapped_all_feature = tf.layers.conv1d(
            inputs=all_features,
            filters=self.pool_filter_size,
            kernel_size=1,
            use_bias=True,
            name="Mapped_all_feature"
        )  # (N, k, pf_size)

        # apply context vector
        # tf.squeeze:删掉指定维度为1的：
        # 全连接将mapped_all_feature结构==》 （batchsize,3,1）
        # squeeze将==> (batchsize,3)
        # softmax==>(batchSize,3)

        feature_weights = tf.nn.softmax(
            tf.squeeze(
                tf.layers.dense(
                    mapped_all_feature,
                    units=1,
                    activation=None,
                    use_bias=False
                ),  # (N, k, 1),
                [2]
            ),  # (N, k)
        )  # (N, k)
        # self.attn_k = feature_weights

        # weighted sum
        # all_features = [?,3,12]
        # feature_weights= [?,3]
        # multify ==> [?,3,12]
        # reduce_sum ==> [?,12] ==>以1为维度的reduce_sum
        weighted_sum_feat = tf.reduce_sum(
            tf.multiply(
                all_features,
                tf.expand_dims(feature_weights, axis=2),
            ),  # (N, k, C)
            axis=[1],
            name="Attn_weighted_sum_feature"
        )  # (N, C)

        # last non-linear
        hidden_logits = tf.layers.dense(
            weighted_sum_feat,
            units=self.embedding_dim // 2,
            activation=tf.nn.relu,
            use_bias=False,
            name="HiddenLogits"
        )  # (N, C/2)

        # the last dense for logits
        logits = tf.squeeze(
            tf.layers.dense(
                hidden_logits,
                units=1,
                activation=None,
                use_bias=False,
                name="Logits"
            ),  # (N, 1)
            axis=[1]
        )  # (N,)

        # 后面都是计算loss了，没啥说的。。
        # sigmoid logits
        self.sigmoid_logits = tf.nn.sigmoid(logits)

        # regularization term
        self.regularization_loss = tf.losses.get_regularization_loss()

        self.logloss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.expand_dims(self.label, -1),
                logits=tf.expand_dims(logits, -1),
                name="SumLogLoss"))

        self.mean_logloss = tf.divide(
            self.logloss,
            tf.to_float(self.batch_size),
            name="MeanLogLoss"
        )

        # overall loss
        self.overall_loss = tf.add(
            self.mean_logloss,
            self.regularization_loss,
            name="OverallLoss"
        )

        tf.summary.scalar("Mean_LogLoss", self.mean_logloss)
        tf.summary.scalar("Reg_Loss", self.regularization_loss)
        tf.summary.scalar("Overall_Loss", self.overall_loss)

        self.train_op = self.optimizer.minimize(self.overall_loss,
                                                global_step=self.global_step)
        self.merged = tf.summary.merge_all()
