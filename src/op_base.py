import time
import glob
import os
import numpy as np
import tensorflow as tf

from src.functions import *

class op(object):
    def __init__(self, args, sess):
        self.sess = sess

        # Train ###########################################################################
        self.gpu_number = args.gpu_number
        self.project = args.project

        # Train Data
        self.data_dir = args.data_dir
        self.dataset = args.dataset  # set5, set19,
        self.data_opt = args.data_opt
        self.data_size = args.data_size

        # Train Iteration
        self.niter = args.niter
        self.niter_snapshot = args.nsnapshot
        self.max_to_keep = args.max_to_keep

        # Train Parameter
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.mm = args.momentum
        self.mm2 = args.momentum2
        self.lamda = args.lamda
        self.gamma = args.gamma
        self.filter_number = args.filter_number
        self.input_size = args.input_size
        self.embedding = args.embedding

        # Result Dir & File
        self.project_dir = '{0}_{1}_{2}/'.format(self.project, self.dataset, self.data_size)
        make_project_dir(self.project_dir)
        self.ckpt_dir = os.path.join(self.project_dir, 'models')
        self.model_name = "{0}.model".format(self.project)

        # Test #############################################################################
        self.build_model()

    def train_gan(self, Train_flag):
        # load data
        data_path = '{0}/{1}/{2}_{3}'.format(self.data_dir, self.dataset, self.data_size, self.data_opt)

        if os.path.exists(data_path + '.npy'):
            data = np.load(data_path + '.npy')
        else:
            data = sorted(glob(os.path.join(data_path, "*.*")))
            np.save(data_path + '.npy', data)

        print 'Shuffle ....'
        random_order = np.random.permutation(len(data))
        data = [data[i] for i in random_order[:]]
        print 'Shuffle Done'

        # initial parameter
        start_time = time.time()
        kt = np.float32(0.)
        lr = np.float32(self.learning_rate)
        self.count = 0

        for epoch in xrange(self.niter):
            batch_idxs = len(data) // self.batch_size

            for idx in xrange(0, batch_idxs):
                self.count += 1

                batch_x = np.random.uniform(-1., 1., size=[self.batch_size, self.input_size])
                batch_files = data[idx * self.batch_size: (idx + 1) * self.batch_size]
                batch_data = [get_image(batch_file) for batch_file in batch_files]

                # optimize model (different with paper)
                g_opt = [self.opt_g, self.g_loss, self.d_real_loss, self.d_fake_loss]
                d_opt = [self.opt_d, self.d_loss, self.merged]
                feed_dict = {self.x: batch_x, self.y: batch_data, self.kt: kt, self.lr:lr}
                _, loss_g, d_real_loss, d_fake_loss = self.sess.run(g_opt, feed_dict=feed_dict)
                _, loss_d, summary = self.sess.run(d_opt, feed_dict=feed_dict)
                self.writer.add_summary(summary, self.count)

                # update kt, m_global
                kt = np.maximum(np.minimum(1., kt + self.lamda * (self.gamma * d_real_loss - d_fake_loss)), 0.)
                m_global = d_real_loss + np.abs(self.gamma * d_real_loss - d_fake_loss)
                loss = loss_g + loss_d

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, "
                      "loss: %.4f, loss_g: %.4f, loss_d: %.4f, d_real: %.4f, d_fake: %.4f, kt: %.8f, M: %.8f"
                      % (epoch, idx, batch_idxs, time.time() - start_time,
                         loss, loss_g, loss_d, d_real_loss, d_fake_loss, kt, m_global))

                # Test during Training
                if self.count % self.niter_snapshot == (self.niter_snapshot - 1):
                    # update learning rate
                    lr *= 0.95
                    # save & test
                    self.save()
                    self.test(Train_flag)

    def test_gan(self, Train_flag=True):
        # load weight
        self.load()

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
        if Train_flag:
            cv2.imwrite(self.project_dir + '/result/'  + str(self.count) + '_output.bmp', im_output_gen)
        else:
            cv2.imwrite(self.project_dir + 'test_gen_output.bmp', im_output_gen)
            cv2.imwrite(self.project_dir + 'test_dec_output.bmp', im_output_dec)

    def save(self):
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.saver.save(self.sess, os.path.join(self.ckpt_dir, self.model_name), global_step=self.count)

    def load(self):
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(self.ckpt_dir, ckpt_name))
