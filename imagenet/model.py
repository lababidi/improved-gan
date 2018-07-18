import os
import time

import numpy as np
import tensorflow as tf

from imagenet.build_model import build_model
from imagenet.generator import Generator
from imagenet.ops import BatchNorm, batch_norm_second_half, batch_norm_first_half, batch_norm_cross, linear, lrelu, \
    conv2d
from imagenet.utils import save_images
from .ops import variables_on_gpu0, avg_grads

filename = "/media/NAS_SHARED/imagenet/imagenet_train_128.tfrecords"


class DCGAN(object):
    def __init__(self, sess, image_size=108, is_crop=True,
                 batch_size=64, image_shape=[64, 64, 3],
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 d_label_smooth=.25,
                 generator_target_prob=1.,
                 checkpoint_dir=None, sample_dir='samples',
                 config=None,
                 devices=None,
                 disable_vbn=False,
                 sample_size=64,
                 out_init_b=0.,
                 out_stddev=.15):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph_def)
        self.saver = tf.train.Saver()
        self.disable_vbn = disable_vbn
        self.devices = devices
        self.d_label_smooth = d_label_smooth
        self.out_init_b = out_init_b
        self.out_stddev = out_stddev
        self.config = config
        self.generator_target_prob = generator_target_prob
        self.generator = Generator(self)
        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.image_shape = image_shape
        self.sample_dir = sample_dir

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = 3

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = BatchNorm(batch_size, name='d_bn1')
        self.d_bn2 = BatchNorm(batch_size, name='d_bn2')

        if not self.y_dim:
            self.d_bn3 = BatchNorm(batch_size, name='d_bn3')

        self.g_bn0 = BatchNorm(batch_size, name='g_bn0')
        self.g_bn1 = BatchNorm(batch_size, name='g_bn1')
        self.g_bn2 = BatchNorm(batch_size, name='g_bn2')

        if not self.y_dim:
            self.g_bn3 = BatchNorm(batch_size, name='g_bn3')
        # Not used by all generators
        self.g_bn4 = BatchNorm(batch_size, name='g_bn4')
        self.g_bn5 = BatchNorm(batch_size, name='g_bn5')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        all_d_grads = []
        all_g_grads = []
        config = self.config
        d_opt = tf.train.AdamOptimizer(config.discriminator_learning_rate, beta1=config.beta1)
        g_opt = tf.train.AdamOptimizer(config.generator_learning_rate, beta1=config.beta1)

        for idx, device in enumerate(self.devices):
            with tf.device("/%s" % device):
                with tf.name_scope("device_%s" % idx):
                    with variables_on_gpu0():
                        self.build_model_single_gpu(self, idx)
                        d_grads = d_opt.compute_gradients(self.d_losses[-1], var_list=self.d_vars)
                        g_grads = g_opt.compute_gradients(self.g_losses[-1], var_list=self.g_vars)
                        all_d_grads.append(d_grads)
                        all_g_grads.append(g_grads)
                        tf.get_variable_scope().reuse_variables()
        avg_d_grads = avg_grads(all_d_grads)
        avg_g_grads = avg_grads(all_g_grads)
        self.d_optim = d_opt.apply_gradients(avg_d_grads)
        self.g_optim = g_opt.apply_gradients(avg_g_grads)

    def build_model_single_gpu(self, gpu_idx):
        assert not self.y_dim

        if gpu_idx == 0:
            filename_queue = tf.train.string_input_producer([filename])  # num_epochs=self.config.epoch)
            self.get_image, self.get_label = read_and_decode_with_labels(filename_queue)

            with tf.variable_scope("misc"):
                chance = 1.  # TODO: declare this down below and make it 1. - 1. / num_classes
                avg_error_rate = tf.get_variable('avg_error_rate', [],
                                                 initializer=tf.constant_initializer(0.),
                                                 trainable=False)
                num_error_rate = tf.get_variable('num_error_rate', [],
                                                 initializer=tf.constant_initializer(0.),
                                                 trainable=False)

        images, sparse_labels = tf.train.shuffle_batch([self.get_image, self.get_label],
                                                       batch_size=self.batch_size,
                                                       num_threads=2,
                                                       capacity=1000 + 3 * self.batch_size,
                                                       min_after_dequeue=1000,
                                                       name='real_images_and_labels')
        if gpu_idx == 0:
            self.sample_images = tf.placeholder(tf.float32, [self.sample_size] + self.image_shape,
                                                name='sample_images')
            self.sample_labels = tf.placeholder(tf.int32, [self.sample_size], name="sample_labels")

            self.reference_G, self.reference_zs = self.generator(is_ref=True)
            # Since I don't know how to turn variable reuse off, I can only activate it once.
            # So here I build a dummy copy of the discriminator before turning variable reuse on for the generator.
            dummy_joint = tf.concat(0, [images, self.reference_G])
            dummy = self.discriminator(dummy_joint, reuse=False, prefix="dummy")

        G, zs = self.generator(is_ref=False)

        if gpu_idx == 0:
            G_means = tf.reduce_mean(G, 0, keep_dims=True)
            G_vars = tf.reduce_mean(tf.square(G - G_means), 0, keep_dims=True)
            G = tf.Print(G, [tf.reduce_mean(G_means), tf.reduce_mean(G_vars)], "generator mean and average var",
                         first_n=1)
            image_means = tf.reduce_mean(images, 0, keep_dims=True)
            image_vars = tf.reduce_mean(tf.square(images - image_means), 0, keep_dims=True)
            images = tf.Print(images, [tf.reduce_mean(image_means), tf.reduce_mean(image_vars)],
                              "image mean and average var", first_n=1)
            self.Gs = []
            self.zses = []
        self.Gs.append(G)
        self.zses.append(zs)

        joint = tf.concat(0, [images, G])
        class_logits, D_on_data, D_on_data_logits, D_on_G, D_on_G_logits = self.discriminator(joint, reuse=True,
                                                                                              prefix="joint ")
        # D_on_G_logits = tf.Print(D_on_G_logits, [D_on_G_logits], "D_on_G_logits")

        self.d_sum = tf.histogram_summary("d", D_on_data)
        self.d__sum = tf.histogram_summary("d_", D_on_G)
        self.G_sum = tf.image_summary("G", G)

        d_label_smooth = self.d_label_smooth
        d_loss_real = sigmoid_kl_with_logits(D_on_data_logits, 1. - d_label_smooth)
        class_loss_weight = 1.
        d_loss_class = class_loss_weight * tf.nn.sparse_softmax_cross_entropy_with_logits(class_logits,
                                                                                          tf.to_int64(sparse_labels))
        error_rate = 1. - tf.reduce_mean(tf.to_float(tf.nn.in_top_k(class_logits, sparse_labels, 1)))
        # self.d_loss_class = tf.Print(self.d_loss_class, [error_rate], "gpu " + str(gpu_idx) + " current minibatch error rate")
        if gpu_idx == 0:
            update = tf.assign(num_error_rate, num_error_rate + 1.)
            with tf.control_dependencies([update]):
                # Start off as a true average for 1st 100 samples
                # Then switch to a running average to compensate for ongoing learning
                tc = tf.maximum(.01, 1. / num_error_rate)
            update = tf.assign(avg_error_rate, (1. - tc) * avg_error_rate + tc * error_rate)
            with tf.control_dependencies([update]):
                d_loss_class = tf.Print(d_loss_class,
                                        [avg_error_rate], "running top-1 error rate")
        # Do not smooth the negative targets.
        # If we use positive targets of alpha and negative targets of beta,
        # then the optimal discriminator function is
        # D(x) = (alpha p_data(x) + beta p_generator(x)) / (p_data(x) + p_generator(x)).
        # This means if we want to get less extreme values, we shrink alpha.
        # Increasing beta makes the generator self-reinforcing.
        # Note that using this one-sided label smoothing also shifts the equilibrium
        # value to alpha/2.
        d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(D_on_G_logits,
                                                              tf.zeros_like(D_on_G_logits))
        g_loss = sigmoid_kl_with_logits(D_on_G_logits, self.generator_target_prob)
        d_loss_class = tf.reduce_mean(d_loss_class)
        d_loss_real = tf.reduce_mean(d_loss_real)
        d_loss_fake = tf.reduce_mean(d_loss_fake)
        g_loss = tf.reduce_mean(g_loss)
        if gpu_idx == 0:
            self.g_losses = []
        self.g_losses.append(g_loss)

        d_loss = d_loss_real + d_loss_fake + d_loss_class
        if gpu_idx == 0:
            self.d_loss_reals = []
            self.d_loss_fakes = []
            self.d_loss_classes = []
            self.d_losses = []
        self.d_loss_reals.append(d_loss_real)
        self.d_loss_fakes.append(d_loss_fake)
        self.d_loss_classes.append(d_loss_class)
        self.d_losses.append(d_loss)

        # self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
        # self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

        if gpu_idx == 0:
            get_vars(self)

    def discriminator(self, image, reuse=False, y=None, prefix=""):
        num_classes = 1001

        if reuse:
            tf.get_variable_scope().reuse_variables()

        batch_size = int(image.get_shape()[0])
        assert batch_size == 2 * self.batch_size

        """
        # L1 distance to average value of corresponding pixel in positive and negative batch
        # Included as a feature to prevent early mode collapse
        b, r, c, ch = [int(e) for e in image.get_shape()]
        pos = tf.slice(image, [0, 0, 0, 0], [self.batch_size, r, c, ch])
        neg = tf.slice(image, [self.batch_size, 0, 0, 0], [self.batch_size, r, c, ch])
        pos = tf.reshape(pos, [self.batch_size, -1])
        neg = tf.reshape(neg, [self.batch_size, -1])
        mean_pos = tf.reduce_mean(pos, 0, keep_dims=True)
        mean_neg = tf.reduce_mean(neg, 0, keep_dims=True)

        # difference from mean, with each example excluding itself from the mean
        pos_diff_pos = (1. + 1. / (self.batch_size - 1.)) * pos - mean_pos
        pos_diff_neg = pos - mean_neg
        neg_diff_pos = neg - mean_pos
        neg_diff_neg = (1. + 1. / (self.batch_size - 1.)) * neg - mean_neg

        diff_feat = tf.concat(0, [tf.concat(1, [pos_diff_pos, pos_diff_neg]),
                                  tf.concat(1, [neg_diff_pos, neg_diff_neg])])

        with tf.variable_scope("d_diff_feat"):
            scale = tf.get_variable("d_untied_scale", [128 * 128 * 3 * 2], tf.float32,
                                     tf.random_normal_initializer(mean=1., stddev=0.1))

        diff_feat = diff_feat = tf.exp(- tf.abs(scale) * tf.abs(diff_feat))
        diff_feat = self.bnx(diff_feat, name="d_bnx_diff_feat")
        """

        noisy_image = image + tf.random_normal([batch_size, 128, 128, 3],
                                               mean=0.0,
                                               stddev=.1)

        print("Discriminator shapes")
        print("image: ", image.get_shape())

        def tower(bn, suffix):
            assert not self.y_dim
            print("\ttower " + suffix)
            h0 = lrelu(bn(conv2d(noisy_image, self.df_dim, name='d_h0_conv' + suffix, d_h=2, d_w=2,
                                 k_w=3, k_h=3), "d_bn_0" + suffix))
            print("\th0 ", h0.get_shape())
            h1 = lrelu(bn(conv2d(h0, self.df_dim * 2, name='d_h1_conv' + suffix, d_h=2, d_w=2,
                                 k_w=3, k_h=3), "d_bn_1" + suffix))
            print("\th1 ", h1.get_shape())
            h2 = lrelu(bn(conv2d(h1, self.df_dim * 4, name='d_h2_conv' + suffix, d_h=2, d_w=2,
                                 k_w=3, k_h=3), "d_bn_2" + suffix))
            print("\th2 ", h2.get_shape())

            h3 = lrelu(bn(conv2d(h2, self.df_dim * 4, name='d_h3_conv' + suffix, d_h=1, d_w=1,
                                 k_w=3, k_h=3), "d_bn_3" + suffix))
            print("\th3 ", h3.get_shape())
            h4 = lrelu(bn(conv2d(h3, self.df_dim * 4, name='d_h4_conv' + suffix, d_h=1, d_w=1,
                                 k_w=3, k_h=3), "d_bn_4" + suffix))
            print("\th4 ", h4.get_shape())
            h5 = lrelu(bn(conv2d(h4, self.df_dim * 8, name='d_h5_conv' + suffix, d_h=2, d_w=2,
                                 k_w=3, k_h=3), "d_bn_5" + suffix))
            print("\th5 ", h5.get_shape())

            h6 = lrelu(bn(conv2d(h5, self.df_dim * 8, name='d_h6_conv' + suffix,
                                 k_w=3, k_h=3), "d_bn_6" + suffix))
            print("\th6 ", h6.get_shape())
            # return tf.reduce_mean(h6, [1, 2])
            h6_reshaped = tf.reshape(h6, [batch_size, -1])
            print('\th6_reshaped: ', h6_reshaped.get_shape())

            h7 = lrelu(bn(linear(h6_reshaped, self.df_dim * 40, scope="d_h7" + suffix), "d_bn_7" + suffix))

            return h7

        h = tower(self.bnx, "")
        print("h: ", h.get_shape())

        n_kernels = 300
        dim_per_kernel = 50
        x = linear(h, n_kernels * dim_per_kernel, scope="d_h")
        activation = tf.reshape(x, (batch_size, n_kernels, dim_per_kernel))

        big = np.zeros((batch_size, batch_size), dtype='float32')
        big += np.eye(batch_size)
        big = tf.expand_dims(big, 1)

        abs_dif = tf.reduce_sum(
            tf.abs(tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)), 2)
        mask = 1. - big
        masked = tf.exp(-abs_dif) * mask

        def half(tens, second):
            m, n, _ = tens.get_shape()
            m = int(m)
            n = int(n)
            return tf.slice(tens, [0, 0, second * self.batch_size], [m, n, self.batch_size])

        # TODO: speedup by allocating the denominator directly instead of constructing it by sum
        #       (current version makes it easier to play with the mask and not need to rederive
        #        the denominator)
        f1 = tf.reduce_sum(half(masked, 0), 2) / tf.reduce_sum(half(mask, 0))
        f2 = tf.reduce_sum(half(masked, 1), 2) / tf.reduce_sum(half(mask, 1))

        minibatch_features = [f1, f2]

        x = tf.concat(1, [h] + minibatch_features)
        print("x: ", x.get_shape())
        # x = tf.nn.dropout(x, .5)

        class_logits = linear(x, num_classes, 'd_indiv_logits')

        image_means = tf.reduce_mean(image, 0, keep_dims=True)
        mean_sub_image = image - image_means
        image_vars = tf.reduce_mean(tf.square(mean_sub_image), 0)

        generated_class_logits = tf.squeeze(tf.slice(class_logits, [0, num_classes - 1], [batch_size, 1]))
        positive_class_logits = tf.slice(class_logits, [0, 0], [batch_size, num_classes - 1])

        """
        # make these a separate matmul with weights initialized to 0, attached only to generated_class_logits, or things explode
        generated_class_logits = tf.squeeze(generated_class_logits) + tf.squeeze(linear(diff_feat, 1, stddev=0., scope="d_indivi_logits_from_diff_feat"))
        assert len(generated_class_logits.get_shape()) == 1
        # re-assemble the logits after incrementing the generated class logits
        class_logits = tf.concat(1, [positive_class_logits, tf.expand_dims(generated_class_logits, 1)])
        """

        mx = tf.reduce_max(positive_class_logits, 1, keep_dims=True)
        safe_pos_class_logits = positive_class_logits - mx

        gan_logits = tf.log(tf.reduce_sum(tf.exp(safe_pos_class_logits), 1)) + tf.squeeze(mx) - generated_class_logits
        assert len(gan_logits.get_shape()) == 1

        probs = tf.nn.sigmoid(gan_logits)

        return [tf.slice(class_logits, [0, 0], [self.batch_size, num_classes]),
                tf.slice(probs, [0], [self.batch_size]),
                tf.slice(gan_logits, [0], [self.batch_size]),
                tf.slice(probs, [self.batch_size], [self.batch_size]),
                tf.slice(gan_logits, [self.batch_size], [self.batch_size])]

    def train(self, config):
        """Train DCGAN"""

        d_optim = self.d_optim
        g_optim = self.g_optim

        tf.initialize_all_variables().run()

        # self.g_sum = tf.merge_summary([#self.z_sum,
        #    self.d__sum,
        #    self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        # self.d_sum = tf.merge_summary([#self.z_sum,
        #     self.d_sum, self.d_loss_real_sum, self.d_loss_sum])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Hang onto a copy of z so we can feed the same one every time we store
        # samples to disk for visualization
        assert self.sample_size > self.batch_size
        assert self.sample_size % self.batch_size == 0
        sample_z = []
        steps = self.sample_size // self.batch_size
        assert steps > 0
        sample_zs = []
        for i in range(steps):
            cur_zs = self.sess.run(self.zses[0])
            assert all(z.shape[0] == self.batch_size for z in cur_zs)
            sample_zs.append(cur_zs)
        sample_zs = [np.concatenate([batch[i] for batch in sample_zs], axis=0) for i in range(len(sample_zs[0]))]
        assert all(sample_z.shape[0] == self.sample_size for sample_z in sample_zs)

        counter = 1

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        start_time = time.time()
        print_time = time.time()
        sample_time = time.time()
        save_time = time.time()
        idx = 0
        try:
            while not coord.should_stop():
                idx += 1
                batch_start_time = time.time()

                (
                    _d_optim, _d_sum, _g_optim,
                    errD_fake, errD_real, errD_class, errG
                ) = self.sess.run([d_optim, self.d_sum,
                                   g_optim,  # self.g_sum,
                                   self.d_loss_fakes[0],
                                   self.d_loss_reals[0],
                                   self.d_loss_classes[0],
                                   self.g_losses[0]])

                counter += 1
                if time.time() - print_time > 15.:
                    print_time = time.time()
                    total_time = print_time - start_time
                    d_loss = errD_fake + errD_real + errD_class
                    sec_per_batch = (print_time - start_time) / (idx + 1.)
                    sec_this_batch = print_time - batch_start_time
                    print(
                        "[Batch %(idx)d] time: %(total_time)4.4f, d_loss: %(d_loss).8f, g_loss: %(errG).8f, "
                        "d_loss_real: %(errD_real).8f, d_loss_fake: %(errD_fake).8f, "
                        "d_loss_class: %(errD_class).8f, sec/batch: %(sec_per_batch)4.4f, "
                        "sec/this batch: %(sec_this_batch)4.4f"
                        % locals())

                if (idx < 300 and idx % 10 == 0) or time.time() - sample_time > 300:
                    sample_time = time.time()
                    samples = []
                    # generator hard codes the batch size
                    for i in range(self.sample_size // self.batch_size):
                        feed_dict = {}
                        for z, zv in zip(self.zses[0], sample_zs):
                            if zv.ndim == 2:
                                feed_dict[z] = zv[i * self.batch_size:(i + 1) * self.batch_size, :]
                            elif zv.ndim == 4:
                                feed_dict[z] = zv[i * self.batch_size:(i + 1) * self.batch_size, :, :, :]
                            else:
                                assert False
                        cur_samples, = self.sess.run(
                            [self.Gs[0]],
                            feed_dict=feed_dict
                        )
                        samples.append(cur_samples)
                    samples = np.concatenate(samples, axis=0)
                    assert samples.shape[0] == self.sample_size
                    save_images(samples, [8, 8],
                                self.sample_dir + '/train_%s.png' % idx)

                if time.time() - save_time > 3600:
                    save_time = time.time()
                    self.save(config.checkpoint_dir, counter)
        except tf.errors.OutOfRangeError:
            print("Done training; epoch limit reached.")
        finally:
            coord.request_stop()

        coord.join(threads)
        # sess.close()

    def bn(self, tensor, name, batch_size=None):
        # the batch size argument is actually unused
        assert name.startswith('g_') or name.startswith('d_'), name
        if not hasattr(self, name):
            setattr(self, name, BatchNorm(batch_size, name=name))
        bn = getattr(self, name)
        return bn(tensor)

    def bn2(self, tensor, name):
        assert name.startswith('g_') or name.startswith('d_'), name
        if not hasattr(self, name):
            setattr(self, name, batch_norm_second_half(name=name))
        bn = getattr(self, name)
        return bn(tensor)

    def bn1(self, tensor, name):
        assert name.startswith('g_') or name.startswith('d_'), name
        if not hasattr(self, name):
            setattr(self, name, batch_norm_first_half(name=name))
        bn = getattr(self, name)
        return bn(tensor)

    def bnx(self, tensor, name):
        assert name.startswith('g_') or name.startswith('d_'), name
        if not hasattr(self, name):
            setattr(self, name, batch_norm_cross(name=name))
        bn = getattr(self, name)
        return bn(tensor)

    def vbn(self, tensor, name, half=None):
        if self.disable_vbn:
            class Dummy(object):
                def __init__(self, tensor, ignored, half):
                    self.reference_output = tensor

                def __call__(self, x):
                    return x

            VBN_cls = Dummy
        else:
            VBN_cls = VBN
        if not hasattr(self, name):
            vbn = VBN_cls(tensor, name, half=half)
            setattr(self, name, vbn)
            return vbn.reference_output
        vbn = getattr(self, name)
        return vbn(tensor)

    def vbnl(self, tensor, name, half=None):
        if self.disable_vbn:
            class Dummy(object):
                def __init__(self, tensor, ignored, half):
                    self.reference_output = tensor

                def __call__(self, x):
                    return x

            VBN_cls = Dummy
        else:
            VBN_cls = VBNL
        if not hasattr(self, name):
            vbn = VBN_cls(tensor, name, half=half)
            setattr(self, name, vbn)
            return vbn.reference_output
        vbn = getattr(self, name)
        return vbn(tensor)

    def vbnlp(self, tensor, name, half=None):
        if self.disable_vbn:
            class Dummy(object):
                def __init__(self, tensor, ignored, half):
                    self.reference_output = tensor

                def __call__(self, x):
                    return x

            VBN_cls = Dummy
        else:
            VBN_cls = VBNLP
        if not hasattr(self, name):
            vbn = VBN_cls(tensor, name, half=half)
            setattr(self, name, vbn)
            return vbn.reference_output
        vbn = getattr(self, name)
        return vbn(tensor)

    def vbn1(self, tensor, name):
        return self.vbn(tensor, name, half=1)

    def vbn2(self, tensor, name):
        return self.vbn(tensor, name, half=2)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            print("Bad checkpoint: ", ckpt)
            return False


class BuildModel(object):
    """
    A class that builds the generator forward prop when called.

    Parameters
    ----------
    dcgan: The DCGAN object to build within.
    func: The function to do it with.
    """

    def __init__(self, dcgan):
        self.dcgan = dcgan
        self.func = build_model

    def __call__(self):
        return self.func(self.dcgan)


class GeneratorF(object):
    """
    A class that builds the generator forward prop when called.

    Parameters
    ----------
    dcgan: The DCGAN object to build the generator within.
    func: The function to do it with.
    """

    def __init__(self, dcgan, func):
        self.dcgan = dcgan
        self.func = func

    def __call__(self, z, y=None):
        return self.func(self.dcgan, z, y)


def get_vars(self):
    t_vars = tf.trainable_variables()
    self.d_vars = [var for var in t_vars if var.name.startswith('d_')]
    self.g_vars = [var for var in t_vars if var.name.startswith('g_')]
    for x in self.d_vars:
        assert x not in self.g_vars
    for x in self.g_vars:
        assert x not in self.d_vars
    for x in t_vars:
        assert x in self.g_vars or x in self.d_vars, x.name
    self.all_vars = t_vars


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape(128 * 128 * 3)
    image = tf.reshape(image, [128, 128, 3])

    image = tf.cast(image, tf.float32) * (2. / 255) - 1.

    return image


def read_and_decode_with_labels(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape(128 * 128 * 3)
    image = tf.reshape(image, [128, 128, 3])

    image = tf.cast(image, tf.float32) * (2. / 255) - 1.

    label = tf.cast(features['label'], tf.int32)

    return image, label


def sigmoid_kl_with_logits(logits, targets):
    # broadcasts the same target value across the whole batch
    # this is implemented so awkwardly because tensorflow lacks an x log x op
    assert isinstance(targets, float)
    if targets in [0., 1.]:
        entropy = 0.
    else:
        entropy = - targets * np.log(targets) - (1. - targets) * np.log(1. - targets)
    return tf.nn.sigmoid_cross_entropy_with_logits(logits, tf.ones_like(logits) * targets) - entropy


class VBNL(object):
    """
    Virtual Batch Normalization, Log scale for the scale parameter
    """

    def __init__(self, x, name, epsilon=1e-5, half=None):
        """
        x is the reference batch
        """
        assert isinstance(epsilon, float)

        self.half = half
        shape = x.get_shape().as_list()
        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()
        with tf.variable_scope(name) as scope:
            assert name.startswith("d_") or name.startswith("g_")
            self.epsilon = epsilon
            self.name = name
            if self.half is None:
                half = x
            elif self.half == 1:
                half = tf.slice(x, [0, 0, 0, 0],
                                [shape[0] // 2, shape[1], shape[2], shape[3]])
            elif self.half == 2:
                half = tf.slice(x, [shape[0] // 2, 0, 0, 0],
                                [shape[0] // 2, shape[1], shape[2], shape[3]])
            else:
                assert False
            self.mean = tf.reduce_mean(half, [0, 1, 2], keep_dims=True)
            self.mean_sq = tf.reduce_mean(tf.square(half), [0, 1, 2], keep_dims=True)
            self.batch_size = int(half.get_shape()[0])
            assert x is not None
            assert self.mean is not None
            assert self.mean_sq is not None
            out = self._normalize(x, self.mean, self.mean_sq, "reference")
            if needs_reshape:
                out = tf.reshape(out, orig_shape)
            self.reference_output = out

    def __call__(self, x):

        shape = x.get_shape().as_list()
        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()
        with tf.variable_scope(self.name) as scope:
            new_coeff = 1. / (self.batch_size + 1.)
            old_coeff = 1. - new_coeff
            new_mean = tf.reduce_mean(x, [1, 2], keep_dims=True)
            new_mean_sq = tf.reduce_mean(tf.square(x), [1, 2], keep_dims=True)
            mean = new_coeff * new_mean + old_coeff * self.mean
            mean_sq = new_coeff * new_mean_sq + old_coeff * self.mean_sq
            out = self._normalize(x, mean, mean_sq, "live")
            if needs_reshape:
                out = tf.reshape(out, orig_shape)
            return out

    def _normalize(self, x, mean, mean_sq, message):
        # make sure this is called with a variable scope
        shape = x.get_shape().as_list()
        assert len(shape) == 4
        self.gamma_driver = tf.get_variable("gamma_driver", [shape[-1]],
                                            initializer=tf.random_normal_initializer(0., 0.02))
        gamma = tf.exp(self.gamma_driver)
        gamma = tf.reshape(gamma, [1, 1, 1, -1])
        self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.))
        beta = tf.reshape(self.beta, [1, 1, 1, -1])
        assert self.epsilon is not None
        assert mean_sq is not None
        assert mean is not None
        std = tf.sqrt(self.epsilon + mean_sq - tf.square(mean))
        out = x - mean
        out = out / std
        # out = tf.Print(out, [tf.reduce_mean(out, [0, 1, 2]),
        #    tf.reduce_mean(tf.square(out - tf.reduce_mean(out, [0, 1, 2], keep_dims=True)), [0, 1, 2])],
        #    message, first_n=-1)
        out = out * gamma
        out = out + beta
        return out


class VBNLP(object):
    """
    Virtual Batch Normalization, Log scale for the scale parameter, per-Pixel normalization
    """

    def __init__(self, x, name, epsilon=1e-5, half=None):
        """
        x is the reference batch
        """
        assert isinstance(epsilon, float)

        self.half = half
        shape = x.get_shape().as_list()
        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()
        with tf.variable_scope(name) as scope:
            assert name.startswith("d_") or name.startswith("g_")
            self.epsilon = epsilon
            self.name = name
            if self.half is None:
                half = x
            elif self.half == 1:
                half = tf.slice(x, [0, 0, 0, 0],
                                [shape[0] // 2, shape[1], shape[2], shape[3]])
            elif self.half == 2:
                half = tf.slice(x, [shape[0] // 2, 0, 0, 0],
                                [shape[0] // 2, shape[1], shape[2], shape[3]])
            else:
                assert False
            self.mean = tf.reduce_mean(half, [0], keep_dims=True)
            self.mean_sq = tf.reduce_mean(tf.square(half), [0], keep_dims=True)
            self.batch_size = int(half.get_shape()[0])
            assert x is not None
            assert self.mean is not None
            assert self.mean_sq is not None
            out = self._normalize(x, self.mean, self.mean_sq, "reference")
            if needs_reshape:
                out = tf.reshape(out, orig_shape)
            self.reference_output = out

    def __call__(self, x):

        shape = x.get_shape().as_list()
        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()
        with tf.variable_scope(self.name) as scope:
            new_coeff = 1. / (self.batch_size + 1.)
            old_coeff = 1. - new_coeff
            new_mean = x
            new_mean_sq = tf.square(x)
            mean = new_coeff * new_mean + old_coeff * self.mean
            mean_sq = new_coeff * new_mean_sq + old_coeff * self.mean_sq
            out = self._normalize(x, mean, mean_sq, "live")
            if needs_reshape:
                out = tf.reshape(out, orig_shape)
            return out

    def _normalize(self, x, mean, mean_sq, message):
        # make sure this is called with a variable scope
        shape = x.get_shape().as_list()
        assert len(shape) == 4
        self.gamma_driver = tf.get_variable("gamma_driver", shape[1:],
                                            initializer=tf.random_normal_initializer(0., 0.02))
        gamma = tf.exp(self.gamma_driver)
        gamma = tf.expand_dims(gamma, 0)
        self.beta = tf.get_variable("beta", shape[1:],
                                    initializer=tf.constant_initializer(0.))
        beta = tf.expand_dims(self.beta, 0)
        assert self.epsilon is not None
        assert mean_sq is not None
        assert mean is not None
        std = tf.sqrt(self.epsilon + mean_sq - tf.square(mean))
        out = x - mean
        out = out / std
        # out = tf.Print(out, [tf.reduce_mean(out, [0, 1, 2]),
        #    tf.reduce_mean(tf.square(out - tf.reduce_mean(out, [0, 1, 2], keep_dims=True)), [0, 1, 2])],
        #    message, first_n=-1)
        out = out * gamma
        out = out + beta
        return out


class VBN(object):
    """
    Virtual Batch Normalization
    """

    def __init__(self, x, name, epsilon=1e-5, half=None):
        """
        x is the reference batch
        """
        assert isinstance(epsilon, float)

        self.half = half
        shape = x.get_shape().as_list()
        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()
        with tf.variable_scope(name) as scope:
            assert name.startswith("d_") or name.startswith("g_")
            self.epsilon = epsilon
            self.name = name
            if self.half is None:
                half = x
            elif self.half == 1:
                half = tf.slice(x, [0, 0, 0, 0],
                                [shape[0] // 2, shape[1], shape[2], shape[3]])
            elif self.half == 2:
                half = tf.slice(x, [shape[0] // 2, 0, 0, 0],
                                [shape[0] // 2, shape[1], shape[2], shape[3]])
            else:
                assert False
            self.mean = tf.reduce_mean(half, [0, 1, 2], keep_dims=True)
            self.mean_sq = tf.reduce_mean(tf.square(half), [0, 1, 2], keep_dims=True)
            self.batch_size = int(half.get_shape()[0])
            assert x is not None
            assert self.mean is not None
            assert self.mean_sq is not None
            out = self._normalize(x, self.mean, self.mean_sq, "reference")
            if needs_reshape:
                out = tf.reshape(out, orig_shape)
            self.reference_output = out

    def __call__(self, x):

        shape = x.get_shape().as_list()
        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()
        with tf.variable_scope(self.name) as scope:
            new_coeff = 1. / (self.batch_size + 1.)
            old_coeff = 1. - new_coeff
            new_mean = tf.reduce_mean(x, [1, 2], keep_dims=True)
            new_mean_sq = tf.reduce_mean(tf.square(x), [1, 2], keep_dims=True)
            mean = new_coeff * new_mean + old_coeff * self.mean
            mean_sq = new_coeff * new_mean_sq + old_coeff * self.mean_sq
            out = self._normalize(x, mean, mean_sq, "live")
            if needs_reshape:
                out = tf.reshape(out, orig_shape)
            return out

    def _normalize(self, x, mean, mean_sq, message):
        # make sure this is called with a variable scope
        shape = x.get_shape().as_list()
        assert len(shape) == 4
        self.gamma = tf.get_variable("gamma", [shape[-1]],
                                     initializer=tf.random_normal_initializer(1., 0.02))
        gamma = tf.reshape(self.gamma, [1, 1, 1, -1])
        self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.))
        beta = tf.reshape(self.beta, [1, 1, 1, -1])
        assert self.epsilon is not None
        assert mean_sq is not None
        assert mean is not None
        std = tf.sqrt(self.epsilon + mean_sq - tf.square(mean))
        out = x - mean
        out = out / std
        # out = tf.Print(out, [tf.reduce_mean(out, [0, 1, 2]),
        #    tf.reduce_mean(tf.square(out - tf.reduce_mean(out, [0, 1, 2], keep_dims=True)), [0, 1, 2])],
        #    message, first_n=-1)
        out = out * gamma
        out = out + beta
        return out
