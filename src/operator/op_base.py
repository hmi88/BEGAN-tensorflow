import glob
import time
import numpy as np
import tensorflow as tf
from src.function.functions import *

class op_base:
    def __init__(self, args, sess):
        self.sess = sess

        # Train
        self.flag = args.flag
        self.gpu_number = args.gpu_number
        self.project = args.project

        # Train Data
        self.data_dir = args.data_dir #./Data
        self.dataset = args.dataset  # celeba
        self.data_size = args.data_size  # 64 or 128
        self.data_opt = args.data_opt  # raw or crop

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
        self.project_dir = 'assets/{0}_{1}_{2}_{3}/'.format(self.project, self.dataset, self.data_opt, self.data_size)
        self.ckpt_dir = os.path.join(self.project_dir, 'models')
        self.model_name = "{0}.model".format(self.project)
        self.ckpt_model_name = os.path.join(self.ckpt_dir, self.model_name)

        # etc.
        if not os.path.exists('assets'):
            os.makedirs('assets')
        make_project_dir(self.project_dir)

    def load(self, sess, saver, ckpt_dir):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(ckpt_dir, ckpt_name))
