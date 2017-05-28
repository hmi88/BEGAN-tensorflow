import glob
import time
import datetime
from src.layer.layers import *
from src.function.functions import *
from src.operator.op_base import op_base


class Operator(op_base):
    def __init__(self, args, sess):
        op_base.__init__(self, args, sess)
        self.build_model()

    def build_model(self):
        # Input placeholder
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_size], name='x')
        self.y = tf.placeholder(tf.float32, shape=[self.batch_size, self.data_size, self.data_size, 3], name='y')
        self.kt = tf.placeholder(tf.float32, name='kt')
        self.lr = tf.placeholder(tf.float32, name='lr')

        # Generator
        self.recon_gen = self.generator(self.x)

        # Discriminator (Critic)
        d_real = self.decoder(self.encoder(self.y))
        d_fake = self.decoder(self.encoder(self.recon_gen, reuse=True), reuse=True)
        self.recon_dec = self.decoder(self.x, reuse=True)

        # Loss
        self.d_real_loss = l1_loss(self.y, d_real)
        self.d_fake_loss = l1_loss(self.recon_gen, d_fake)
        self.d_loss = self.d_real_loss - self.kt * self.d_fake_loss
        self.g_loss = self.d_fake_loss
        self.m_global = self.d_real_loss + tf.abs(self.gamma * self.d_real_loss - self.d_fake_loss)

        # Variables
        g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "gen_")
        d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "disc_")

        # Optimizer
        self.opt_g = tf.train.AdamOptimizer(self.lr, self.mm).minimize(self.g_loss, var_list=g_vars)
        self.opt_d = tf.train.AdamOptimizer(self.lr, self.mm).minimize(self.d_loss, var_list=d_vars)


        # initializer
        self.sess.run(tf.global_variables_initializer())

        # tf saver
        self.saver = tf.train.Saver(max_to_keep=(self.max_to_keep))

        try:
            self.load(self.sess, self.saver, self.ckpt_dir)
        except:
            # save full graph
            self.saver.save(self.sess, self.ckpt_model_name, write_meta_graph=True)

        # Summary
        if self.flag:
            tf.summary.scalar('loss/loss', self.d_loss + self.g_loss)
            tf.summary.scalar('loss/g_loss', self.g_loss)
            tf.summary.scalar('loss/d_loss', self.d_loss)
            tf.summary.scalar('loss/d_real_loss', self.d_real_loss)
            tf.summary.scalar('loss/d_fake_loss', self.d_fake_loss)
            tf.summary.scalar('misc/kt', self.kt)
            tf.summary.scalar('misc/m_global', self.m_global)
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(self.project_dir, self.sess.graph)

    def train(self, train_flag):
        # load data
        data_path = '{0}/{1}/{2}_{3}'.format(self.data_dir, self.dataset, self.data_size, self.data_opt)

        if os.path.exists(data_path + '.npy'):
            data = np.load(data_path + '.npy')
        else:
            data = sorted(glob(os.path.join(data_path, "*.*")))
            np.save(data_path + '.npy', data)

        print('Shuffle ....')
        random_order = np.random.permutation(len(data))
        data = [data[i] for i in random_order[:]]
        print('Shuffle Done')

        # initial parameter
        start_time = time.time()
        kt = np.float32(0.)
        lr = np.float32(self.learning_rate)
        self.count = 0

        for epoch in range(self.niter):
            batch_idxs = len(data) // self.batch_size

            for idx in range(0, batch_idxs):
                self.count += 1

                batch_x = np.random.uniform(-1., 1., size=[self.batch_size, self.input_size])
                batch_files = data[idx * self.batch_size: (idx + 1) * self.batch_size]
                batch_data = [get_image(batch_file) for batch_file in batch_files]

                # opt & feed list (different with paper)
                g_opt = [self.opt_g, self.g_loss, self.d_real_loss, self.d_fake_loss]
                d_opt = [self.opt_d, self.d_loss, self.merged]
                feed_dict = {self.x: batch_x, self.y: batch_data, self.kt: kt, self.lr: lr}

                # run tensorflow
                _, loss_g, d_real_loss, d_fake_loss = self.sess.run(g_opt, feed_dict=feed_dict)
                _, loss_d, summary = self.sess.run(d_opt, feed_dict=feed_dict)

                # update kt, m_global
                kt = np.maximum(np.minimum(1., kt + self.lamda * (self.gamma * d_real_loss - d_fake_loss)), 0.)
                m_global = d_real_loss + np.abs(self.gamma * d_real_loss - d_fake_loss)
                loss = loss_g + loss_d

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, "
                      "loss: %.4f, loss_g: %.4f, loss_d: %.4f, d_real: %.4f, d_fake: %.4f, kt: %.8f, M: %.8f"
                      % (epoch, idx, batch_idxs, time.time() - start_time,
                         loss, loss_g, loss_d, d_real_loss, d_fake_loss, kt, m_global))

                # write train summary
                self.writer.add_summary(summary, self.count)

                # Test during Training
                if self.count % self.niter_snapshot == (self.niter_snapshot - 1):
                    # update learning rate
                    lr *= 0.95
                    # save & test
                    self.saver.save(self.sess, self.ckpt_model_name, global_step=self.count, write_meta_graph=False)
                    self.test(train_flag)

    def test(self, train_flag=True):
        # generate output
        img_num = self.batch_size
        img_size = self.data_size

        output_f = int(np.sqrt(img_num))
        im_output_gen = np.zeros([img_size * output_f, img_size * output_f, 3])
        im_output_dec = np.zeros([img_size * output_f, img_size * output_f, 3])

        test_data = np.random.uniform(-1., 1., size=[img_num, self.input_size])
        output_gen = (self.sess.run(self.recon_gen, feed_dict={self.x: test_data}))  # generator output
        output_dec = (self.sess.run(self.recon_dec, feed_dict={self.x: test_data}))  # decoder output

        output_gen = [inverse_image(output_gen[i]) for i in range(img_num)]
        output_dec = [inverse_image(output_dec[i]) for i in range(img_num)]

        for i in range(output_f):
            for j in range(output_f):
                im_output_gen[i * img_size:(i + 1) * img_size, j * img_size:(j + 1) * img_size, :] \
                    = output_gen[j + (i * output_f)]
                im_output_dec[i * img_size:(i + 1) * img_size, j * img_size:(j + 1) * img_size, :] \
                    = output_dec[j + (i * output_f)]

        # output save
        if train_flag:
            scm.imsave(self.project_dir + '/result/' + str(self.count) + '_output.bmp', im_output_gen)
        else:
            now = datetime.datetime.now()
            nowDatetime = now.strftime('%Y-%m-%d_%H:%M:%S')
            scm.imsave(self.project_dir + '/result_test/gen_{}_output.bmp'.format(nowDatetime), im_output_gen)
            scm.imsave(self.project_dir + '/result_test/dec_{}_output.bmp'.format(nowDatetime), im_output_dec)
