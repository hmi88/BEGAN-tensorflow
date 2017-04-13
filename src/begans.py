from src.layers import *
from src.op_base import op


class BEGANS(op):
    def __init__(self, args, sess):
        op.__init__(self, args, sess)

    def generator(self, x, reuse=None):
        with tf.variable_scope('gen_') as scope:
            if reuse:
                scope.reuse_variables()

            w = self.data_size
            f = self.filter_number

            x = fc(x, 8 * 8 * f, name='fc')
            x = tf.reshape(x, [-1, 8, 8, f])

            x = conv2d(x, [3, 3, f, f], stride=1, name='conv1_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, f, f], stride=1, name='conv1_b')
            x = tf.nn.elu(x)

            if self.data_size == 128:
                x = tf.image.resize_nearest_neighbor(x, size=(w / 8, w / 8))
                x = conv2d(x, [3, 3, f, f], stride=1, name='conv2_a')
                x = tf.nn.elu(x)
                x = conv2d(x, [3, 3, f, f], stride=1, name='conv2_b')
                x = tf.nn.elu(x)

            x = tf.image.resize_nearest_neighbor(x, size=(w / 4, w / 4))
            x = conv2d(x, [3, 3, f, f], stride=1, name='conv3_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, f, f], stride=1, name='conv3_b')
            x = tf.nn.elu(x)

            x = tf.image.resize_nearest_neighbor(x, size=(w / 2, w / 2))
            x = conv2d(x, [3, 3, f, f], stride=1, name='conv4_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, f, f], stride=1, name='conv4_b')
            x = tf.nn.elu(x)

            x = tf.image.resize_nearest_neighbor(x, size=(w, w))
            x = conv2d(x, [3, 3, f, f], stride=1, name='conv5_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, f, f], stride=1, name='conv5_b')
            x = tf.nn.elu(x)

            x = conv2d(x, [3, 3, f, 3], stride=1, name='conv6_a')
        return x

    def encoder(self, x, reuse=None):
        with tf.variable_scope('disc_') as scope:
            if reuse:
                scope.reuse_variables()

            f = self.filter_number
            h = self.embedding

            x = conv2d(x, [3, 3, 3, f], stride=1, name='conv1_enc_a')
            x = tf.nn.elu(x)

            x = conv2d(x, [3, 3, f, f], stride=1, name='conv2_enc_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, f, f], stride=1, name='conv2_enc_b')
            x = tf.nn.elu(x)

            x = conv2d(x, [1, 1, f, 2 * f], stride=1, name='conv3_enc_0')
            x = Pool(x, r=2, s=2)
            x = conv2d(x, [3, 3, 2 * f, 2 * f], stride=1, name='conv3_enc_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, 2 * f, 2 * f], stride=1, name='conv3_enc_b')
            x = tf.nn.elu(x)

            x = conv2d(x, [1, 1, 2 * f, 3 * f], stride=1, name='conv4_enc_0')
            x = Pool(x, r=2, s=2)
            x = conv2d(x, [3, 3, 3 * f, 3 * f], stride=1, name='conv4_enc_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, 3 * f, 3 * f], stride=1, name='conv4_enc_b')
            x = tf.nn.elu(x)

            x = conv2d(x, [1, 1, 3 * f, 4 * f], stride=1, name='conv5_enc_0')
            x = Pool(x, r=2, s=2)
            x = conv2d(x, [3, 3, 4 * f, 4 * f], stride=1, name='conv5_enc_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, 4 * f, 4 * f], stride=1, name='conv5_enc_b')
            x = tf.nn.elu(x)

            if self.data_size == 128:
                x = conv2d(x, [1, 1, 4 * f, 5 * f], stride=1, name='conv6_enc_0')
                x = Pool(x, r=2, s=2)
                x = conv2d(x, [3, 3, 5 * f, 5 * f], stride=1, name='conv6_enc_a')
                x = tf.nn.elu(x)
                x = conv2d(x, [3, 3, 5 * f, 5 * f], stride=1, name='conv6_enc_b')
                x = tf.nn.elu(x)

            x = fc(x, h, name='enc_fc')
        return x

    def decoder(self, x, reuse=None):
        with tf.variable_scope('disc_') as scope:
            if reuse:
                scope.reuse_variables()

            w = self.data_size
            f = self.filter_number

            x = fc(x, 8 * 8 * f, name='fc')
            x = tf.reshape(x, [-1, 8, 8, f])

            x = conv2d(x, [3, 3, f, f], stride=1, name='conv1_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, f, f], stride=1, name='conv1_b')
            x = tf.nn.elu(x)

            if self.data_size == 128:
                x = tf.image.resize_nearest_neighbor(x, size=(w / 8, w / 8))
                x = conv2d(x, [3, 3, f, f], stride=1, name='conv2_a')
                x = tf.nn.elu(x)
                x = conv2d(x, [3, 3, f, f], stride=1, name='conv2_b')
                x = tf.nn.elu(x)

            x = tf.image.resize_nearest_neighbor(x, size=(w / 4, w / 4))
            x = conv2d(x, [3, 3, f, f], stride=1, name='conv3_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, f, f], stride=1, name='conv3_b')
            x = tf.nn.elu(x)

            x = tf.image.resize_nearest_neighbor(x, size=(w / 2, w / 2))
            x = conv2d(x, [3, 3, f, f], stride=1, name='conv4_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, f, f], stride=1, name='conv4_b')
            x = tf.nn.elu(x)

            x = tf.image.resize_nearest_neighbor(x, size=(w, w))
            x = conv2d(x, [3, 3, f, f], stride=1, name='conv5_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, f, f], stride=1, name='conv5_b')
            x = tf.nn.elu(x)

            x = conv2d(x, [3, 3, f, 3], stride=1, name='conv6_a')
        return x

    def build_model(self):
        DEVICES = ['/gpu:{}'.format(i) for i in self.gpu_number.split(",")]

        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_size], name='x')
        self.y = tf.placeholder(tf.float32, shape=[self.batch_size, self.data_size, self.data_size, 3], name='y')
        self.kt = tf.placeholder(tf.float32, name='kt')
        self.lr = tf.placeholder(tf.float32, name='lr')

        self.x_split = tf.split(self.x, len(DEVICES))
        self.y_split = tf.split(self.y, len(DEVICES))

        opt_g = tf.train.AdamOptimizer(self.lr, self.mm)
        opt_d = tf.train.AdamOptimizer(self.lr, self.mm)

        all_grads_g, all_grads_d = [], []
        self.recon_gen, self.recon_dec = [], []

        with tf.variable_scope(tf.get_variable_scope()):
            for device_idx, (device, x, y) in enumerate(zip(DEVICES, self.x_split, self.y_split)):
                with tf.device(device):
                    print device
                    # Generator
                    recon_gen = self.generator(x)
                    self.recon_gen.append(recon_gen)

                    # Discriminator (Critic)
                    d_real = self.decoder(self.encoder(y))
                    d_fake = self.decoder(self.encoder(recon_gen, reuse=True), reuse=True)
                    recon_dec = self.decoder(x, reuse=True)
                    self.recon_dec.append(recon_dec)

                    # Loss
                    self.d_real_loss = l1_loss(y, d_real)
                    self.d_fake_loss = l1_loss(self.recon_gen, d_fake)
                    self.d_loss = self.d_real_loss - self.kt * self.d_fake_loss
                    self.g_loss = self.d_fake_loss
                    self.m_global = self.d_real_loss + tf.abs(self.gamma * self.d_real_loss - self.d_fake_loss)

                    # Reuse variables for the next tower
                    tf.get_variable_scope().reuse_variables()

                    # Variables
                    t_vars = tf.global_variables()
                    g_vars = [var for var in t_vars if 'gen_'.format(device_idx) in var.name]
                    d_vars = [var for var in t_vars if 'disc_'.format(device_idx) in var.name]

                    # compute_gradients
                    grads_g = opt_g.compute_gradients(self.g_loss, var_list=g_vars)
                    grads_d = opt_d.compute_gradients(self.d_loss, var_list=d_vars)

                    all_grads_g.append(grads_g)
                    all_grads_d.append(grads_d)

        self.recon_gen = tf.concat(self.recon_gen, 0)
        self.recon_dec = tf.concat(self.recon_dec, 0)

        # summary
        tf.summary.scalar('loss', self.d_loss + self.g_loss)
        tf.summary.scalar('g_loss', self.g_loss)
        tf.summary.scalar('d_loss', self.d_loss)
        tf.summary.scalar('d_real_loss', self.d_real_loss)
        tf.summary.scalar('d_fake_loss', self.d_fake_loss)
        tf.summary.scalar('kt', self.kt)
        tf.summary.scalar('m_global', self.m_global)
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.project_dir, self.sess.graph)

        avg_grad_g = average_gradients(all_grads_g)
        avg_grad_d = average_gradients(all_grads_d)

        self.opt_g = opt_g.apply_gradients(avg_grad_g)
        self.opt_d = opt_d.apply_gradients(avg_grad_d)

        # tf saver
        self.saver = tf.train.Saver(max_to_keep=(self.max_to_keep))

        # initializer
        self.sess.run(tf.global_variables_initializer())

    def train(self, Train_flag):
        try:
            self.load()
            print 'Load !!'
        except:
            pass
        op.train_gan(self, Train_flag)

    def test(self, Train_flag):
        op.test_gan(self, Train_flag)

    def save(self):
        op.save(self)

    def load(self):
        op.load(self)
