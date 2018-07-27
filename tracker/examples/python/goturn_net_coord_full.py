import tensorflow as tf
import numpy as np
from helper.config import POLICY
from logger.logger import setup_logger

logger = setup_logger(logfile=None)


def expit_tensor(x):
    return 1. / (1. + tf.exp(-x))


# https://github.com/tensorflow/tensorflow/issues/4079
def LeakyReLU(x, alpha=0.1, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1.0 + alpha)
        f2 = 0.5 * (1.0 - alpha)
        return f1 * x + f2 * abs(x)


class TRACKNET:
    def __init__(self, batch_size, train=True):

        self.batch_size = batch_size
        self.target = tf.placeholder(tf.float32, [None, 227, 227, 3])
        self.image = tf.placeholder(tf.float32, [None, 227, 227, 3])
        self.parameters = {}
        self.outdim_cord = POLICY['num'] * 4
        self.outdim_obj = POLICY['side'] * POLICY['side'] * POLICY['num'] * 1
        self.train = train
        self.wd = 0.0005
        if train:
            self.bbox = tf.placeholder(tf.float32, [None, 4])
            self.confs = tf.placeholder(tf.float32, [None, POLICY['side'] * POLICY['side'], POLICY['num']])
            self.coord = tf.placeholder(tf.float32, [None, POLICY['num'], 4])

    def build(self):
        ########### for target ###########
        # [filter_height, filter_width, in_channels, out_channels]
        tf.summary.image("image", self.image, max_outputs=4)
        tf.summary.image("target", self.target, max_outputs=4)

        self.target_conv1 = self._conv_relu_layer(bottom=self.target, filter_size=[11, 11, 3, 96],
                                                  strides=[1, 4, 4, 1], name="target_conv_1")

        # now 55 x 55 x 96
        self.target_pool1 = tf.nn.max_pool(self.target_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                           padding='VALID', name='target_pool1')
        # now 27 x 27 x 96
        self.target_lrn1 = tf.nn.local_response_normalization(self.target_pool1, depth_radius=2, alpha=0.0001,
                                                              beta=0.75, name="target_lrn1")
        # now 27 x 27 x 96

        self.target_conv2 = self._conv_relu_layer(bottom=self.target_lrn1, filter_size=[5, 5, 48, 256],
                                                  strides=[1, 1, 1, 1], pad=2, bias_init=1.0, group=2,
                                                  name="target_conv_2")
        # now 27 x 27 x 256

        self.target_pool2 = tf.nn.max_pool(self.target_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                           padding='VALID', name='target_pool2')
        # now 13 x 13 x 256
        self.target_lrn2 = tf.nn.local_response_normalization(self.target_pool2, depth_radius=2, alpha=0.0001,
                                                              beta=0.75, name="target_lrn2")
        # now 13 x 13 x 256
        self.target_conv3 = self._conv_relu_layer(bottom=self.target_lrn2, filter_size=[3, 3, 256, 384],
                                                  strides=[1, 1, 1, 1], pad=1, name="target_conv_3")
        # now 13 x 13 x 384
        self.target_conv4 = self._conv_relu_layer(bottom=self.target_conv3, filter_size=[3, 3, 192, 384], bias_init=1.0,
                                                  strides=[1, 1, 1, 1], pad=1, group=2, name="target_conv_4")
        # now 13 x 13 x 384
        self.target_conv5 = self._conv_relu_layer(bottom=self.target_conv4, filter_size=[3, 3, 192, 256], bias_init=1.0,
                                                  strides=[1, 1, 1, 1], pad=1, group=2, name="target_conv_5")
        # now 13 x 13 x 256
        self.target_pool5 = tf.nn.max_pool(self.target_conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                           padding='VALID', name='target_pool5')
        # now 6 x 6 x 256

        ########### for image ###########
        # [filter_height, filter_width, in_channels, out_channels]
        self.image_conv1 = self._conv_relu_layer(bottom=self.image, filter_size=[11, 11, 3, 96],
                                                 strides=[1, 4, 4, 1], name="image_conv_1")

        # now 55 x 55 x 96
        self.image_pool1 = tf.nn.max_pool(self.image_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                          padding='VALID', name='image_pool1')

        # now 27 x 27 x 96
        self.image_lrn1 = tf.nn.local_response_normalization(self.image_pool1, depth_radius=2, alpha=0.0001,
                                                             beta=0.75, name="image_lrn1")

        # now 27 x 27 x 96

        self.image_conv2 = self._conv_relu_layer(bottom=self.image_lrn1, filter_size=[5, 5, 48, 256],
                                                 strides=[1, 1, 1, 1], pad=2, bias_init=1.0, group=2,
                                                 name="image_conv_2")

        # now 27 x 27 x 256

        self.image_pool2 = tf.nn.max_pool(self.image_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                          padding='VALID', name='image_pool2')

        # now 13 x 13 x 256
        self.image_lrn2 = tf.nn.local_response_normalization(self.image_pool2, depth_radius=2, alpha=0.0001,
                                                             beta=0.75, name="image_lrn2")

        # now 13 x 13 x 256
        self.image_conv3 = self._conv_relu_layer(bottom=self.image_lrn2, filter_size=[3, 3, 256, 384],
                                                 strides=[1, 1, 1, 1], pad=1, name="image_conv_3")

        # now 13 x 13 x 384
        self.image_conv4 = self._conv_relu_layer(bottom=self.image_conv3, filter_size=[3, 3, 192, 384],
                                                 strides=[1, 1, 1, 1], pad=1, group=2, name="image_conv_4")

        # now 13 x 13 x 384
        self.image_conv5 = self._conv_relu_layer(bottom=self.image_conv4, filter_size=[3, 3, 192, 256], bias_init=1.0,
                                                 strides=[1, 1, 1, 1], pad=1, group=2, name="image_conv_5")

        # now 13 x 13 x 256
        self.image_pool5 = tf.nn.max_pool(self.image_conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                          padding='VALID', name='image_pool5')

        # now 6 x 6 x 256
        # tensorflow layer: n * w * h * c
        # but caffe layer is: n * c * h * w

        ########### Concatnate two layers ###########
        self.concat = tf.concat([self.target_pool5, self.image_pool5], axis=3)  # 0, 1, 2, 3 - > 2, 3, 1, 0

        # important, since caffe has different layer dimension order
        self.concat = tf.transpose(self.concat, perm=[0, 3, 1, 2])

        # important, since caffe has different layer dimension order
        self.image_pool5 = tf.transpose(self.image_pool5, perm=[0, 3, 1, 2])

        ########### fully connencted layers ###########

        # for object only using image
        self.fc1_image = self._fc_relu_layers(self.concat, dim=4096, name="fc1_image")
        if (self.train):
            self.fc1_image = tf.nn.dropout(self.fc1_image, 0.5)

        self.fc2_image = self._fc_relu_layers(self.fc1_image, dim=4096, name="fc2_image")
        if (self.train):
            self.fc2_image = tf.nn.dropout(self.fc2_image, 0.5)

        self.fc3_image = self._fc_relu_layers(self.fc2_image, dim=4096, name="fc3_image")
        if (self.train):
            self.fc3_image = tf.nn.dropout(self.fc3_image, 0.5)

        self.fc4_image = self._fc_layers(self.fc3_image, dim=self.outdim_obj, name="fc4_image")
        self.re_fc4_image = tf.reshape(self.fc4_image, shape=[-1, POLICY['side'], POLICY['side'], 1])
        tf.summary.image("objectness", expit_tensor(self.re_fc4_image[:, :, :, 0:]), max_outputs=4)

        # coordinate using image + target
        self.fc1_adj = self._fc_relu_layers(self.concat, dim=4096, name="fc1_adj")
        if (self.train):
            self.fc1_adj = tf.nn.dropout(self.fc1_adj, 0.5)

        self.fc2_adj = self._fc_relu_layers(self.fc1_adj, dim=4096, name="fc2_adj")
        if (self.train):
            self.fc2_adj = tf.nn.dropout(self.fc2_adj, 0.5)

        self.fc3_adj = self._fc_relu_layers(self.fc2_adj, dim=4096, name="fc3_adj")
        if (self.train):
            self.fc3_adj = tf.nn.dropout(self.fc3_adj, 0.5)

        self.fc4_adj = self._fc_layers(self.fc3_adj, dim=self.outdim_cord, name="fc4_adj")


        self.print_shapes()

        if (self.train):
            self.loss = self.loss_grid(self.re_fc4_image, POLICY, name="loss_grid") + \
                        self.loss_coord(self.fc4_adj, POLICY ,name="loss_coord")
            tf.summary.scalar('total loss', self.loss)

            l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='l2_weight_loss')
            self.loss_wdecay = self.loss + l2_loss

    def loss_coord(self, net_out, training_schedule, name=None):
        m = training_schedule
        B = m['num']
        anchors = m['anchors']
        scoord = m['coord_scale']
        H, W = m['side'], m['side']

        net_out_reshape = tf.reshape(net_out, [-1, B, 4])
        adjusted_coords_xy = expit_tensor(net_out_reshape[:, :, 0:2])
        adjusted_coords_wh = tf.sqrt(
            tf.exp(net_out_reshape[:, :, 2:4]) * tf.reshape(anchors, [1, B, 2]) / np.reshape([W, H], [1, 1, 2]) + 1e-8)
        coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 2)

        diff = tf.subtract(coords, self.coord)
        diff_flat = tf.abs(tf.reshape(diff, [-1, B * 4]))
        loss = tf.reduce_sum(diff_flat, 1)
        loss = tf.reduce_mean(loss, name=name)
        loss = scoord * loss
        tf.summary.scalar('loss_coord', loss)

        return loss

    def loss_grid(self, net_out, training_schedule, name=None):
        """
        from YOLOv2, link: https://github.com/thtrieu/darkflow
        """
        # meta
        m = training_schedule
        H, W = m['side'], m['side']
        B = m['num']
        HW = H * W  # number of grid cells
        sconf = float(m['object_scale'])
        snoob = float(m['noobject_scale'])


        # Extract the coordinate prediction from net.out
        net_out_reshape = tf.reshape(net_out, [-1, H, W, B])
        adjusted_c = expit_tensor(net_out_reshape)
        adjusted_c = tf.reshape(adjusted_c, [-1, H * W, B])
        adjusted_net_out = adjusted_c

        conid = snoob * (1. - self.confs) + sconf * self.confs
        loss = tf.pow(adjusted_net_out - self.confs, 2)
        loss = tf.multiply(loss, conid)
        loss = tf.reshape(loss, [-1, H * W * B])
        loss = tf.reduce_sum(loss, 1)
        loss = tf.reduce_mean(loss, name=name)
        tf.summary.scalar('loss_grid', loss)

        return loss

    def _batch(self, bboxes, training_schedule):
        """
        Takes a chunk of parsed annotations
        returns value for placeholders of net's
        input & loss layer correspond to this chunk
        :param box: box.x1, box.y1, box.x2, box.y2
        """
        error_box_index = []

        meta = training_schedule
        S, B = meta['side'], meta['num']
        w, h = 10, 10  # 10 is self.bbox's width, height
        # Calculate regression target
        cellx = 1. * w / S
        celly = 1. * h / S

        count = 0
        for idx, bbox in enumerate(bboxes):
            obj = [0, 0, 0, 0]

            centerx = .5 * (bbox[0] + bbox[2])  # xmin, xmax
            centery = .5 * (bbox[1] + bbox[3])  # ymin, ymax
            cx = centerx / cellx
            cy = centery / celly

            # this is implemented in example_generator, def check_center
            # if cx >= S or cy >= S:
            #     raise ('center point error')
            #     return None, None
            obj[2] = (bbox[2] - bbox[0]) / w
            obj[3] = (bbox[3] - bbox[1]) / h
            obj[2] = np.sqrt(obj[2])
            obj[3] = np.sqrt(obj[3])
            obj[0] = cx - np.floor(cx)  # centerx
            obj[1] = cy - np.floor(cy)  # centery
            obj += [int(np.floor(cy) * S + np.floor(cx))]

            # debug
            # print(cx, cy)

            # show(im, allobj, S, w, h, cellx, celly) # unit test

            # Calculate placeholders' values
            confs = np.zeros([S * S, B])
            coord = np.zeros([B, 4])
            prear = np.zeros([S * S, 4])

            try:
                coord[:, :] = [obj[0:4]] * B
                confs[obj[4], :] = [1.] * B

            except IndexError:
                logger.error(str(idx) + ' is not boundary')
                logger.error('cx is ' + str(np.floor(cx)))
                logger.error('cy is ' + str(np.floor(cy)))
                error_box_index.append(idx)

            # Finalise the placeholders' values
            confs = np.expand_dims(confs, 0)
            coord = np.expand_dims(coord, 0)
            if not count == 0:
                batch_confs = np.concatenate([batch_confs, confs], axis=0)
                batch_coord = np.concatenate([batch_coord, coord], axis=0)
            else:
                batch_confs = confs
                batch_coord = coord
            count += 1




        feed_val = {
            'confs': batch_confs, 'coord': batch_coord
        }
        return feed_val, error_box_index

    def _conv_relu_layer(self, bottom, filter_size, strides, pad=0, bias_init=0.0, group=1, trainable=False, name=None):
        with tf.name_scope(name) as scope:

            if (pad > 0):
                paddings = [[0, 0], [pad, pad], [pad, pad], [0, 0]]
                bottom = tf.pad(bottom, paddings, "CONSTANT")
            kernel = tf.Variable(tf.truncated_normal(filter_size, dtype=tf.float32,
                                                     stddev=1e-2), trainable=trainable, name='weights')
            biases = tf.Variable(tf.constant(bias_init, shape=[filter_size[3]], dtype=tf.float32), trainable=trainable,
                                 name='biases')
            self.parameters[name] = [kernel, biases]
            if (group == 1):
                conv = tf.nn.conv2d(bottom, kernel, strides, padding='VALID')
                out = tf.nn.bias_add(conv, biases)
            elif (group == 2):
                kernel1, kernel2 = tf.split(kernel, num_or_size_splits=group, axis=3)
                bottom1, bottom2 = tf.split(bottom, num_or_size_splits=group, axis=3)
                conv1 = tf.nn.conv2d(bottom1, kernel1, strides, padding='VALID')
                conv2 = tf.nn.conv2d(bottom2, kernel2, strides, padding='VALID')
                conv = tf.concat([conv1, conv2], axis=3)
                out = tf.nn.bias_add(conv, biases)
            else:
                raise TypeError("number of groups not supported")

            # if not tf.get_variable_scope().reuse:
            #     weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd,
            #                            name='kernel_loss')
            #     tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
            #                      weight_decay)

            out2 = tf.nn.relu(out)
            _activation_summary(out2)
            out2 = tf.Print(out2, [tf.shape(out2)], message='Shape of %s' % name, first_n=1, summarize=4)
            return out2

    def _fc_relu_layers(self, bottom, dim, name=None):
        with tf.name_scope(name) as scope:
            shape = int(np.prod(bottom.get_shape()[1:]))
            weights = tf.Variable(tf.truncated_normal([shape, dim],
                                                      dtype=tf.float32, stddev=0.005), name='weights')
            bias = tf.Variable(tf.constant(1.0, shape=[dim], dtype=tf.float32), name='biases')
            bottom_flat = tf.reshape(bottom, [-1, shape])
            fc_weights = tf.nn.bias_add(tf.matmul(bottom_flat, weights), bias)
            self.parameters[name] = [weights, bias]

            if not tf.get_variable_scope().reuse:
                weight_decay = tf.multiply(tf.nn.l2_loss(weights), self.wd,
                                           name='fc_relu_weight_loss')
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                     weight_decay)

            top = tf.nn.relu(fc_weights, name=scope)
            _activation_summary(top)
            top = tf.Print(top, [tf.shape(top)], message='Shape of %s' % name, first_n=1, summarize=4)
            return top

    def _fc_layers(self, bottom, dim, name=None):
        with tf.name_scope(name) as scope:
            shape = int(np.prod(bottom.get_shape()[1:]))
            weights = tf.Variable(tf.truncated_normal([shape, dim],
                                                      dtype=tf.float32, stddev=0.005), name='weights')
            bias = tf.Variable(tf.constant(1.0, shape=[dim], dtype=tf.float32), name='biases')
            bottom_flat = tf.reshape(bottom, [-1, shape])
            top = tf.nn.bias_add(tf.matmul(bottom_flat, weights), bias, name=scope)
            self.parameters[name] = [weights, bias]

            if not tf.get_variable_scope().reuse:
                weight_decay = tf.multiply(tf.nn.l2_loss(weights), self.wd,
                                           name='fc_weight_loss')
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                     weight_decay)

            _activation_summary(top)
            top = tf.Print(top, [tf.shape(top)], message='Shape of %s' % name, first_n=1, summarize=4)
            return top

    def _add_wd_and_summary(self, var, wd, collection_name=None):
        if collection_name is None:
            collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.multiply(
                tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection(collection_name, weight_decay)
        _variable_summaries(var)
        return var

    def print_shapes(self):
        print("%s:" % (self.image_conv1), self.image_conv1.get_shape().as_list())
        print("%s:" % (self.image_pool1), self.image_pool1.get_shape().as_list())
        print("%s:" % (self.image_lrn1), self.image_lrn1.get_shape().as_list())
        print("%s:" % (self.image_conv2), self.image_conv2.get_shape().as_list())
        print("%s:" % (self.image_pool2), self.image_pool2.get_shape().as_list())
        print("%s:" % (self.image_lrn2), self.image_lrn2.get_shape().as_list())
        print("%s:" % (self.image_conv3), self.image_conv3.get_shape().as_list())
        print("%s:" % (self.image_conv4), self.image_conv4.get_shape().as_list())
        print("%s:" % (self.image_conv5), self.image_conv5.get_shape().as_list())
        print("%s:" % (self.image_pool5), self.image_pool5.get_shape().as_list())
        print("%s:" % (self.concat), self.concat.get_shape().as_list())
        print("%s:" % (self.fc1_image), self.fc1_image.get_shape().as_list())
        print("%s:" % (self.fc2_image), self.fc2_image.get_shape().as_list())
        print("%s:" % (self.fc3_image), self.fc3_image.get_shape().as_list())
        print("%s:" % (self.fc4_image), self.fc4_image.get_shape().as_list())
        print("%s:" % (self.fc1_adj), self.fc1_adj.get_shape().as_list())
        print("%s:" % (self.fc2_adj), self.fc2_adj.get_shape().as_list())
        print("%s:" % (self.fc3_adj), self.fc3_adj.get_shape().as_list())
        print("%s:" % (self.fc4_adj), self.fc4_adj.get_shape().as_list())
        # \print("%s:" % (self.net_out), self.net_out.get_shape().as_list())
        print("kernel_sizes:")
        for key in self.parameters:
            print("%s:" % (key), self.parameters[key][0].get_shape().as_list())

    def load_weight_from_dict(self, weights_dict, sess):
        # for convolutional layers
        sess.run(self.parameters['target_conv_1'][0].assign(weights_dict['conv1']['weights']))
        sess.run(self.parameters['target_conv_2'][0].assign(weights_dict['conv2']['weights']))
        sess.run(self.parameters['target_conv_3'][0].assign(weights_dict['conv3']['weights']))
        sess.run(self.parameters['target_conv_4'][0].assign(weights_dict['conv4']['weights']))
        sess.run(self.parameters['target_conv_5'][0].assign(weights_dict['conv5']['weights']))
        sess.run(self.parameters['image_conv_1'][0].assign(weights_dict['conv1_p']['weights']))
        sess.run(self.parameters['image_conv_2'][0].assign(weights_dict['conv2_p']['weights']))
        sess.run(self.parameters['image_conv_3'][0].assign(weights_dict['conv3_p']['weights']))
        sess.run(self.parameters['image_conv_4'][0].assign(weights_dict['conv4_p']['weights']))
        sess.run(self.parameters['image_conv_5'][0].assign(weights_dict['conv5_p']['weights']))

        sess.run(self.parameters['target_conv_1'][1].assign(weights_dict['conv1']['bias']))
        sess.run(self.parameters['target_conv_2'][1].assign(weights_dict['conv2']['bias']))
        sess.run(self.parameters['target_conv_3'][1].assign(weights_dict['conv3']['bias']))
        sess.run(self.parameters['target_conv_4'][1].assign(weights_dict['conv4']['bias']))
        sess.run(self.parameters['target_conv_5'][1].assign(weights_dict['conv5']['bias']))
        sess.run(self.parameters['image_conv_1'][1].assign(weights_dict['conv1_p']['bias']))
        sess.run(self.parameters['image_conv_2'][1].assign(weights_dict['conv2_p']['bias']))
        sess.run(self.parameters['image_conv_3'][1].assign(weights_dict['conv3_p']['bias']))
        sess.run(self.parameters['image_conv_4'][1].assign(weights_dict['conv4_p']['bias']))
        sess.run(self.parameters['image_conv_5'][1].assign(weights_dict['conv5_p']['bias']))

        # for fully connected layers
        sess.run(self.parameters['fc1'][0].assign(weights_dict['fc6-new']['weights']))
        sess.run(self.parameters['fc2'][0].assign(weights_dict['fc7-new']['weights']))
        sess.run(self.parameters['fc3'][0].assign(weights_dict['fc7-newb']['weights']))
        sess.run(self.parameters['fc4'][0].assign(weights_dict['fc8-shapes']['weights']))

        sess.run(self.parameters['fc1'][1].assign(weights_dict['fc6-new']['bias']))
        sess.run(self.parameters['fc2'][1].assign(weights_dict['fc7-new']['bias']))
        sess.run(self.parameters['fc3'][1].assign(weights_dict['fc7-newb']['bias']))
        sess.run(self.parameters['fc4'][1].assign(weights_dict['fc8-shapes']['bias']))

    def test(self):
        sess = tf.Session()
        a = np.full((self.batch_size, 227, 227, 3), 1)  # numpy.full(shape, fill_value, dtype=None, order='C')
        b = np.full((self.batch_size, 227, 227, 3), 2)
        sess.run(tf.global_variables_initializer())

        sess.run([self.fc4_concat], feed_dict={self.image: a, self.target: b})


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_summaries(var):
    """Attach a lot of summaries to a Tensor."""
    if not tf.get_variable_scope().reuse:
        name = var.op.name
        logging.debug("Creating Summary for: %s" % name)
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar(name + '/mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.summary.scalar(name + '/sttdev', stddev)
            tf.summary.scalar(name + '/max', tf.reduce_max(var))
            tf.summary.scalar(name + '/min', tf.reduce_min(var))
            tf.summary.histogram(name, var)


if __name__ == "__main__":
    tracknet = TRACKNET(10)
    tracknet.build()
    sess = tf.Session()
    a = np.full((tracknet.batch_size, 227, 227, 3), 1)
    b = np.full((tracknet.batch_size, 227, 227, 3), 2)
    sess.run(tf.global_variables_initializer())
    sess.run([tracknet.image_pool5], feed_dict={tracknet.image: a, tracknet.target: b})



