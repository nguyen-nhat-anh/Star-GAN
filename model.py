import numpy as np
import os
import pandas as pd
import tensorflow as tf
import time

from utils import TrainingDataset, display_samples, decode_img, display_test_result
from generator import make_generator_model
from discriminator import make_discriminator_model
from losses import adversarial_loss, classification_loss, reconstruction_loss


class StarGAN:
    def __init__(self, config):
        # for training
        self.img_dir = config.img_dir  # os.path.join('data', 'img_align_celeba', 'img_align_celeba')
        self.label_df = pd.read_csv(config.label_path)  # os.path.join('data', 'list_attr_celeba.csv')
        self.selected_attributes = config.selected_attributes  # ['Male', 'Young']
        self.label_dim = len(self.selected_attributes)  # 2
        self.buffer_size = config.buffer_size  # 5000
        self.batch_size = config.batch_size  # 16

        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim

        self.lambda_cls = config.lambda_cls  # 1.0
        self.lambda_rec = config.lambda_rec  # 10.0
        self.lambda_gp = config.lambda_gp  # 10.0

        self.n_critics = config.n_critics  # 5
        self.g_init_lr = config.g_init_lr  # 1e-4
        self.d_init_lr = config.d_init_lr  # 1e-4

        self.decay_iter = config.decay_iter  # 100000
        self.decay_freq = config.decay_freq  # 1000

        self.ckpt_dir = config.ckpt_dir  # 'model'
        self.sample_dir = config.sample_dir  # 'samples'

        self.start_iter = config.start_iter  # 0
        self.end_iter = config.end_iter  # 200000
        self.display_freq = config.display_freq  # 1000
        self.save_freq = config.save_freq  # 10000

        # for testing
        self.test_img_path = config.test_img_path   # e.g 'test/test_img.png'
        self.attr_values = config.attr_values  # e.g [1, 0]

    def prepare_dataset(self):
        training_set = TrainingDataset(self.label_df, self.selected_attributes)
        dataset = tf.data.Dataset.list_files(os.path.join(self.img_dir, '*.*'))
        dataset = dataset.map(training_set.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(self.buffer_size).batch(self.batch_size, drop_remainder=True).\
            repeat().prefetch(tf.data.experimental.AUTOTUNE)
        return dataset  # (128, 128, 3), (label_dim,)

    def build_model(self):
        self.generator = make_generator_model(label_dim=self.label_dim, g_conv_dim=self.g_conv_dim)
        self.discriminator = make_discriminator_model(label_dim=self.label_dim, d_conv_dim=self.d_conv_dim)

        self.g_lr_ph = tf.placeholder(tf.float32, name='g_learning_rate')
        self.d_lr_ph = tf.placeholder(tf.float32, name='d_learning_rate')
        self.g_optimizer = tf.train.AdamOptimizer(self.g_lr_ph, beta1=0.5, beta2=0.999)
        self.d_optimizer = tf.train.AdamOptimizer(self.d_lr_ph, beta1=0.5, beta2=0.999)

    def gradient_penalty(self, img_hat):
        with tf.GradientTape() as disc_tape:
            disc_tape.watch(img_hat)  # GradientTape only automatically tracks trainable variables.
                                      # Tensors can be watched by invoking the `watch()` method on the context manager.
            out_src_hat, _ = self.discriminator(img_hat)
        grads = disc_tape.gradient(out_src_hat, img_hat)  # (None, 128, 128, 3)
        grads = tf.reshape(grads, shape=[grads.shape.as_list()[0], -1])  # (None, 128*128*3)
        grad_norms = tf.norm(grads, axis=1)  # (None,)
        return tf.reduce_mean(tf.square(grad_norms - 1))

    def compute_losses(self, img_real, label_org, label_trg):
        # discriminator loss
        out_src_real, out_cls_real = self.discriminator(img_real)
        img_fake = self.generator([img_real, label_trg])
        out_src_fake, out_cls_fake = self.discriminator(img_fake)

        d_adv_loss = -adversarial_loss(out_src_fake, out_src_real)
        d_cls_loss = classification_loss(label_org, out_cls_real)

        alpha = tf.random.uniform(shape=[img_real.shape.as_list()[0], 1, 1, 1])
        img_hat = alpha * img_real + (1 - alpha) * img_fake
        d_gp_loss = self.gradient_penalty(img_hat)

        self.d_loss = d_adv_loss + self.lambda_cls * d_cls_loss + self.lambda_gp * d_gp_loss

        # generator loss
        g_adv_loss = adversarial_loss(out_src_fake)
        g_cls_loss = classification_loss(label_trg, out_cls_fake)

        img_rec = self.generator([img_fake, label_org])
        g_rec_loss = reconstruction_loss(img_real, img_rec)

        self.g_loss = g_adv_loss + self.lambda_cls * g_cls_loss + self.lambda_rec * g_rec_loss

    def generate_samples(self, img_real, label_org):
        """
        params:
         img_real - (batch_size, 128, 128, 3)
         label_org - (batch_size, n_attributes)
        returns:
         samples - (n_attributes + 1, batch_size, 128, 128, 3)
         where samples[0] are original images,
               samples[i] are generated images with the i-th attribute flipped, i > 0
        """
        batch_size, n_attributes = label_org.shape.as_list()
        id_matrix = tf.eye(n_attributes)  # (n_attributes, n_attributes)
        label_flipped = 1.0 - label_org  # (batch_size, n_attributes)
        sample_list = [img_real]
        for i in range(n_attributes):
            id_vector = tf.cast(tf.expand_dims(id_matrix[i], axis=0), tf.bool)  # (1, n_attributes)
            cond = tf.tile(id_vector, [batch_size, 1])  # (batch_size, n_attributes)
            label = tf.where(cond, x=label_flipped, y=label_org)  # (batch_size, n_attributes)
            img_generated = self.generator([img_real, label])  # (batch_size, 128, 128, 3)
            sample_list.append(img_generated)
        return tf.stack(sample_list)  # (n_attributes + 1, batch_size, 128, 128, 3)

    def train(self):
        # inputs
        print('1. Preparing dataset...')
        dataset = self.prepare_dataset()
        iterator = dataset.make_one_shot_iterator()
        img_real, label_org = iterator.get_next()

        label_trg = tf.random.shuffle(label_org)  # generate random target labels by shuffling the original labels

        # create generator and discriminator
        print('2. Creating model...')
        self.build_model()

        # create loss ops for generator and discriminator
        self.compute_losses(img_real, label_org, label_trg)

        # create training ops
        d_train_op = self.d_optimizer.minimize(self.d_loss, var_list=self.discriminator.trainable_variables)
        g_train_op = self.g_optimizer.minimize(self.g_loss, var_list=self.generator.trainable_variables)

        # sample results
        samples = self.generate_samples(img_real, label_org)

        # training loop
        print('3. Training...')
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=1)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init)

            # checkpoint
            latest_ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
            if latest_ckpt is None:
                print('No checkpoint found!')
            else:
                print('Found checkpoint : "{}"'.format(latest_ckpt))
                saver.restore(sess, latest_ckpt)
                self.start_iter = int(latest_ckpt.split('-')[-1])

            # learning rate
            if self.start_iter < (self.end_iter - self.decay_iter):
                g_lr = self.g_init_lr
                d_lr = self.d_init_lr
            else:
                offset = self.start_iter - (self.end_iter - self.decay_iter)  # offset from starting decay
                k = offset // self.decay_freq + 1
                g_lr = self.g_init_lr - k * self.g_init_lr / float(self.decay_iter)
                d_lr = self.d_init_lr - k * self.d_init_lr / float(self.decay_iter)

            # training loop
            for iteration in range(self.start_iter, self.end_iter):
                start = time.time()

                sess.run(d_train_op, feed_dict={self.d_lr_ph: d_lr})

                if (iteration + 1) % self.n_critics == 0:
                    sess.run(g_train_op, feed_dict={self.g_lr_ph: g_lr})

                # print info
                print('Time for iteration {}/{} is {} sec'.format(iteration + 1, self.end_iter, time.time() - start))

                # display result samples
                if (iteration + 1) % self.display_freq == 0:
                    samples_arr = sess.run(samples)
                    display_samples(samples_arr, sample_dir=self.sample_dir, index=iteration + 1, nrows=5)

                # update learning rate
                if (iteration + 1) % self.decay_freq == 0 and (iteration + 1) > (self.end_iter - self.decay_iter):
                    g_lr -= (self.g_init_lr / float(self.decay_iter))
                    d_lr -= (self.d_init_lr / float(self.decay_iter))
                    print('Discriminator learning rate updated: {}'.format(d_lr))
                    print('Generator learning rate updated: {}'.format(g_lr))

                # save model
                if (iteration + 1) % self.save_freq == 0:
                    saver.save(sess, os.path.join(self.ckpt_dir, 'model'), global_step=iteration + 1)

    def test(self):
        test_img = tf.expand_dims(decode_img(self.test_img_path), axis=0)
        label_trg = tf.expand_dims(tf.constant(self.attr_values, dtype=tf.float32), axis=0)

        self.generator = make_generator_model(label_dim=self.label_dim, g_conv_dim=self.g_conv_dim)
        generated_img = self.generator([test_img, label_trg])

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=1)

        with tf.Session() as sess:
            sess.run(init)

            # checkpoint
            latest_ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
            if latest_ckpt is None:
                print('No checkpoint found!')
            else:
                print('Found checkpoint : "{}"'.format(latest_ckpt))
                saver.restore(sess, latest_ckpt)

            # generate result
            test_img_arr, generated_img_arr = sess.run([test_img, generated_img])  # (1, 128, 128, 3)
            test_img_arr = np.squeeze(test_img_arr, axis=0)  # (128, 128, 3)
            generated_img_arr = np.squeeze(generated_img_arr, axis=0)  # (128, 128, 3)

            display_test_result(test_img_arr, generated_img_arr,
                                self.selected_attributes, self.attr_values, self.test_img_path)
