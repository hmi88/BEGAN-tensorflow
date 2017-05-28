from src.layer.layers import *
from src.operator.op_BEGAN import Operator


class BEGAN(Operator):
    def __init__(self, args, sess):
        Operator.__init__(self, args, sess)

    def generator(self, x, reuse=None):
        with tf.variable_scope('gen_') as scope:
            if reuse:
                scope.reuse_variables()

            w = self.data_size
            f = self.filter_number
            p = "SAME"

            x = fc(x, 8 * 8 * f, name='fc')
            x = tf.reshape(x, [-1, 8, 8, f])

            x = conv2d(x, [3, 3, f, f], stride=1, padding=p, name='conv1_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, f, f], stride=1,  padding=p, name='conv1_b')
            x = tf.nn.elu(x)

            if self.data_size == 128:
                x = resize_nn(x, w / 8)
                x = conv2d(x, [3, 3, f, f], stride=1,  padding=p, name='conv2_a')
                x = tf.nn.elu(x)
                x = conv2d(x, [3, 3, f, f], stride=1,  padding=p, name='conv2_b')
                x = tf.nn.elu(x)

            x = resize_nn(x, w / 4)
            x = conv2d(x, [3, 3, f, f], stride=1,  padding=p, name='conv3_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, f, f], stride=1,  padding=p, name='conv3_b')
            x = tf.nn.elu(x)

            x = resize_nn(x, w / 2)
            x = conv2d(x, [3, 3, f, f], stride=1,  padding=p, name='conv4_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, f, f], stride=1,  padding=p, name='conv4_b')
            x = tf.nn.elu(x)

            x = resize_nn(x, w)
            x = conv2d(x, [3, 3, f, f], stride=1,  padding=p,name='conv5_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, f, f], stride=1,  padding=p,name='conv5_b')
            x = tf.nn.elu(x)

            x = conv2d(x, [3, 3, f, 3], stride=1,  padding=p,name='conv6_a')
        return x

    def encoder(self, x, reuse=None):
        with tf.variable_scope('disc_') as scope:
            if reuse:
                scope.reuse_variables()

            f = self.filter_number
            h = self.embedding
            p = "SAME"

            x = conv2d(x, [3, 3, 3, f], stride=1,  padding=p,name='conv1_enc_a')
            x = tf.nn.elu(x)

            x = conv2d(x, [3, 3, f, f], stride=1,  padding=p,name='conv2_enc_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, f, f], stride=1,  padding=p,name='conv2_enc_b')
            x = tf.nn.elu(x)

            x = conv2d(x, [1, 1, f, 2 * f], stride=1,  padding=p,name='conv3_enc_0')
            x = pool(x, r=2, s=2)
            x = conv2d(x, [3, 3, 2 * f, 2 * f], stride=1,  padding=p,name='conv3_enc_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, 2 * f, 2 * f], stride=1,  padding=p,name='conv3_enc_b')
            x = tf.nn.elu(x)

            x = conv2d(x, [1, 1, 2 * f, 3 * f], stride=1,  padding=p,name='conv4_enc_0')
            x = pool(x, r=2, s=2)
            x = conv2d(x, [3, 3, 3 * f, 3 * f], stride=1,  padding=p,name='conv4_enc_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, 3 * f, 3 * f], stride=1,  padding=p,name='conv4_enc_b')
            x = tf.nn.elu(x)

            x = conv2d(x, [1, 1, 3 * f, 4 * f], stride=1,  padding=p,name='conv5_enc_0')
            x = pool(x, r=2, s=2)
            x = conv2d(x, [3, 3, 4 * f, 4 * f], stride=1,  padding=p,name='conv5_enc_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, 4 * f, 4 * f], stride=1,  padding=p,name='conv5_enc_b')
            x = tf.nn.elu(x)

            if self.data_size == 128:
                x = conv2d(x, [1, 1, 4 * f, 5 * f], stride=1,  padding=p,name='conv6_enc_0')
                x = pool(x, r=2, s=2)
                x = conv2d(x, [3, 3, 5 * f, 5 * f], stride=1,  padding=p,name='conv6_enc_a')
                x = tf.nn.elu(x)
                x = conv2d(x, [3, 3, 5 * f, 5 * f], stride=1,  padding=p,name='conv6_enc_b')
                x = tf.nn.elu(x)

            x = fc(x, h, name='enc_fc')
        return x

    def decoder(self, x, reuse=None):
        with tf.variable_scope('disc_') as scope:
            if reuse:
                scope.reuse_variables()

            w = self.data_size
            f = self.filter_number
            p = "SAME"

            x = fc(x, 8 * 8 * f, name='fc')
            x = tf.reshape(x, [-1, 8, 8, f])

            x = conv2d(x, [3, 3, f, f], stride=1, padding=p, name='conv1_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, f, f], stride=1, padding=p, name='conv1_b')
            x = tf.nn.elu(x)

            if self.data_size == 128:
                x = resize_nn(x, w / 8)
                x = conv2d(x, [3, 3, f, f], stride=1, padding=p, name='conv2_a')
                x = tf.nn.elu(x)
                x = conv2d(x, [3, 3, f, f], stride=1, padding=p, name='conv2_b')
                x = tf.nn.elu(x)

                x = resize_nn(x, w / 4)
            x = conv2d(x, [3, 3, f, f], stride=1, padding=p, name='conv3_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, f, f], stride=1, padding=p, name='conv3_b')
            x = tf.nn.elu(x)

            x = resize_nn(x, w / 2)
            x = conv2d(x, [3, 3, f, f], stride=1, padding=p, name='conv4_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, f, f], stride=1, padding=p, name='conv4_b')
            x = tf.nn.elu(x)

            x = resize_nn(x, w)
            x = conv2d(x, [3, 3, f, f], stride=1, padding=p, name='conv5_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, f, f], stride=1, padding=p, name='conv5_b')
            x = tf.nn.elu(x)

            x = conv2d(x, [3, 3, f, 3], stride=1, padding=p, name='conv6_a')
        return x
