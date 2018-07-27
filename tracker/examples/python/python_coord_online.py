import vot
import sys

sys.path.append('/usr/local/lib')
import os
import glob

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from helper.image_proc import cropPadImage
from helper.BoundingBox import BoundingBox, calculate_box
import goturn_net_coord
import numpy as np
import cv2
from helper.config import POLICY
from logger.logger import setup_logger
from example_generator import example_generator, check_center


def data_generator(image_prev, bbox_prev_tight, kGeneratedExamplesPerImage):
    objExampleGen = example_generator(float(POLICY['lamda_shift']), float(POLICY['lamda_scale']),
                                      float(POLICY['min_scale']), float(POLICY['max_scale']), logger)

    images = []
    targets = []
    bbox_gt_scaleds = []

    center_bool = False
    while not center_bool:
        # img_prev, img_curr, bbox_prev, bbox_curr = train_video(train_vid_videos)
        objExampleGen.reset(bbox_prev_tight, bbox_prev_tight, image_prev, image_prev)
        image, target, bbox_gt_scaled = objExampleGen.make_true_example()
        center_bool = check_center(bbox_gt_scaled)
        if center_bool:
            images.append(image)
            targets.append(target)
            bbox_gt_scaleds.append(bbox_gt_scaled)
            images, targets, bbox_gt_scaleds = objExampleGen.make_training_examples(kGeneratedExamplesPerImage,
                                                                                    images, targets, bbox_gt_scaleds)

    # debug
    # show_images(images, targets, bbox_gt_scaleds)

    for idx, (img, tag, box) in enumerate(zip(images, targets, bbox_gt_scaleds)):
        images[idx] = cv2.resize(img, (POLICY['HEIGHT'], POLICY['WIDTH']), interpolation=cv2.INTER_CUBIC)
        targets[idx] = cv2.resize(tag, (POLICY['HEIGHT'], POLICY['WIDTH']), interpolation=cv2.INTER_CUBIC)
        bbox_gt_scaleds[idx] = np.array([box.x1, box.y1, box.x2, box.y2], dtype=np.float32)

    images = np.reshape(np.array(images), (len(images), 227, 227, 3))
    targets = np.reshape(np.array(targets), (len(targets), 227, 227, 3))
    bbox_gt_scaled = np.reshape(np.array(bbox_gt_scaleds), (len(bbox_gt_scaleds), 4))

    return [images, targets, bbox_gt_scaled]


class bbox_estimator:
    """tracker class"""

    def __init__(self, show_intermediate_output, logger):
        """TODO: to be defined. """
        self.show_intermediate_output = show_intermediate_output
        self.logger = logger

    def init(self, image_curr, region):
        """ initializing the first frame in the video
        """
        left = max(region.x, 0)
        top = max(region.y, 0)
        right = min(region.x + region.width, image.shape[1] - 1)
        bottom = min(region.y + region.height, image.shape[0] - 1)
        bbox_gt = BoundingBox(left, top, right, bottom)
        self.image_prev = image_curr
        self.bbox_prev_tight = bbox_gt
        self.bbox_curr_prior_tight = bbox_gt
        self.DeltaBox = np.array([0., 0.])
        self.lambdaBox = 0.3
        self.prevBoxeffect = 0
        self.occlusion_flag = 0
        self.frist_frame = True

    def preprocess(self, image):
        """TODO: Docstring for preprocess.

        :arg1: TODO
        :returns: TODO

        """
        image_out = image
        if image_out.shape != (POLICY['HEIGHT'], POLICY['WIDTH'], POLICY['channels']):
            image_out = cv2.resize(image_out, (POLICY['WIDTH'], POLICY['HEIGHT']), interpolation=cv2.INTER_CUBIC)

        image_out = np.float32(image_out)
        return image_out

    # def online_training(self, tracknet, train_step, kGeneratedExamplesPerImage):
    #     cur_batch = data_generator(self.image_prev, self.bbox_prev_tight, kGeneratedExamplesPerImage)
    #     feed_val, error_box_index = tracknet._batch(cur_batch[2], POLICY)
    #
    #     for i in xrange(kGeneratedExamplesPerImage):
    #         [_, loss] = sess.run([train_step, tracknet.loss_wdecay], feed_dict={tracknet.image: cur_batch[0][i+1, ...],
    #                                                                             tracknet.target: cur_batch[1][i+1, ...],
    #                                                                             tracknet.bbox: cur_batch[2][i+1, ...],
    #                                                                             tracknet.confs: feed_val['confs'][i+1, ...],
    #                                                                             tracknet.coord: feed_val['coord'][i+1, ...]})
    #         logger.debug('Train: iteration: %.3fs, average_loss: %f' % (i, loss))

    def online_training(self, tracknet, train_step, step):
        cur_batch = data_generator(self.image_prev, self.bbox_prev_tight, 10)
        feed_val, error_box_index = tracknet._batch(cur_batch[2], POLICY)

        for i in xrange(step):
            [_, loss] = sess.run([train_step, tracknet.loss_wdecay], feed_dict={tracknet.image: cur_batch[0],
                                                                                tracknet.target: cur_batch[1],
                                                                                tracknet.bbox: cur_batch[2],
                                                                                tracknet.confs: feed_val['confs'],
                                                                                tracknet.coord: feed_val['coord']})

            logger.debug('Train: iteration: %.3fs, average_loss: %f' % (i, loss))

    def track(self, image_curr, tracknet, train_step, step, sess):
        """TODO: Docstring for tracker.
        :returns: TODO

        """
        if self.frist_frame:
            self.online_training(tracknet, train_step, step)
            self.frist_frame = False

        # occlusion is learned
        # else:
        #     self.online_training(tracknet, train_step, 1)

        target_pad, _, _, _ = cropPadImage(self.bbox_prev_tight, self.image_prev)
        cur_search_region, search_location, edge_spacing_x, edge_spacing_y = cropPadImage(self.bbox_curr_prior_tight,
                                                                                          image_curr)

        # image, BGR(training type)
        cur_search_region_resize = self.preprocess(cur_search_region)
        target_pad_resize = self.preprocess(target_pad)

        cur_search_region_expdim = np.expand_dims(cur_search_region_resize, axis=0)
        target_pad_expdim = np.expand_dims(target_pad_resize, axis=0)

        re_fc4_image, fc4_adj = sess.run([tracknet.re_fc4_image, tracknet.fc4_adj],
                                         feed_dict={tracknet.image: cur_search_region_expdim,
                                                    tracknet.target: target_pad_expdim})
        bbox_estimate, object_bool, objectness = calculate_box(re_fc4_image, fc4_adj)

        print('objectness_s is: ', objectness)

        ########### original method ############
        # this box is NMS result, TODO, all bbox check

        if not len(bbox_estimate) == 0:
            bbox_estimate = BoundingBox(bbox_estimate[0][0], bbox_estimate[0][1], bbox_estimate[0][2],
                                        bbox_estimate[0][3])

            # Inplace correction of bounding box
            bbox_estimate.unscale(cur_search_region)
            bbox_estimate.uncenter(image_curr, search_location, edge_spacing_x, edge_spacing_y)

            # self.image_prev = image_curr
            # self.bbox_prev_tight = bbox_estimate
            self.bbox_curr_prior_tight = bbox_estimate
        else:
            # self.image_prev = self.image_prev
            # self.bbox_prev_tight = self.bbox_prev_tight
            self.bbox_curr_prior_tight = self.bbox_curr_prior_tight
            bbox_estimate = self.bbox_curr_prior_tight

        ########### original method ############

        ############ trick method ############

        # if object_bool:
        # # if not len(bbox_estimate) == 0:
        #     # current_box_wh = np.array([(bbox_estimate.[0][2] - bbox_estimate.[0][0]), (bbox_estimate.[0][3] - bbox_estimate.[0][1])], dtype=np.float32)
        #     # prev_box_wh = np.array([5., 5.], dtype=np.float32)
        #
        #     bbox_estimate = BoundingBox(bbox_estimate[0][0], bbox_estimate[0][1], bbox_estimate[0][2], bbox_estimate[0][3])
        #
        #     # relative distance from center point [5. 5.]
        #     relative_current_box = np.array([(bbox_estimate.x2 + bbox_estimate.x1) / 2,
        #                             (bbox_estimate.y2 + bbox_estimate.y1) / 2],
        #                            dtype=np.float32)
        #     relative_distance = np.linalg.norm(relative_current_box - np.array([5., 5.]))
        #
        #     # Inplace correction of bounding box
        #     bbox_estimate.unscale(cur_search_region)
        #     bbox_estimate.uncenter(image_curr, search_location, edge_spacing_x, edge_spacing_y)
        #
        #     # image's width height , center point
        #     current_box = np.array([(bbox_estimate.x2 + bbox_estimate.x1) / 2, (bbox_estimate.y2 + bbox_estimate.y1) / 2], dtype=np.float32)
        #     prev_box = np.array([(self.bbox_curr_prior_tight.x2 + self.bbox_curr_prior_tight.x1) / 2, (self.bbox_curr_prior_tight.y2 + self.bbox_curr_prior_tight.y1) / 2],
        #                         dtype=np.float32)
        #
        #     if relative_distance < 2:
        #         self.DeltaBox = self.lambdaBox * (current_box - prev_box) + (1 - self.lambdaBox) * self.DeltaBox
        #
        #
        #         self.image_prev = image_curr
        #         self.bbox_prev_tight = bbox_estimate
        #         self.bbox_curr_prior_tight = bbox_estimate
        #         print(self.DeltaBox)
        #     else:
        #         # under prev img, box block is no update
        #         self.image_prev = self.image_prev
        #         self.bbox_prev_tight = self.bbox_prev_tight
        #         # self.bbox_curr_prior_tight = self.bbox_prev_tight
        #         self.bbox_curr_prior_tight = BoundingBox(self.bbox_curr_prior_tight.x1 + self.DeltaBox[0],
        #                                                  self.bbox_curr_prior_tight.y1 + self.DeltaBox[1],
        #                                                  self.bbox_curr_prior_tight.x2 + self.DeltaBox[0],
        #                                                  self.bbox_curr_prior_tight.y2 + self.DeltaBox[1])
        #         bbox_estimate = self.bbox_curr_prior_tight
        #         print('distance is {:>3}'.format(relative_distance))
        #         print(self.DeltaBox)
        # else:
        #     # under prev img, box block is no update
        #     self.image_prev = self.image_prev
        #     self.bbox_prev_tight = self.bbox_prev_tight
        #     # self.bbox_curr_prior_tight = self.bbox_prev_tight
        #     self.bbox_curr_prior_tight = BoundingBox(self.bbox_curr_prior_tight.x1 + self.DeltaBox[0],
        #                                              self.bbox_curr_prior_tight.y1 + self.DeltaBox[1],
        #                                              self.bbox_curr_prior_tight.x2 + self.DeltaBox[0],
        #                                              self.bbox_curr_prior_tight.y2 + self.DeltaBox[1])
        #     bbox_estimate = self.bbox_curr_prior_tight
        #     print('occlusion is detected')
        #     print(self.DeltaBox)
        #
        # ############ trick method ############

        left_x = bbox_estimate.x1
        left_y = bbox_estimate.y1
        width = bbox_estimate.x2 - bbox_estimate.x1
        height = bbox_estimate.y2 - bbox_estimate.y1
        return vot.Rectangle(left_x, left_y, width, height)
        # return bbox_estimate


handle = vot.VOT("rectangle")
selection = handle.region()
BATCH_SIZE = 1
logger = setup_logger(logfile=None)
# ckpt = '/datahdd/workdir/jaehyuk/code/experiment/vgg19/grid13_box1_vgg19_alltraining/checkpoints/checkpoint.ckpt-277261'
# ckpt = '/datahdd/workdir/jaehyuk/checkpoints/VIDtracker/180712_adj/checkpoint.ckpt-560836'
# ckpt = '/datahdd/workdir/jaehyuk/code/experiment/VIDtracker/checkpoints/checkpoint.ckpt-551566'
# ckpt = "/home/jaehyuk/code/github/vot-toolkit/tracker/examples/python/checkpoints/checkpoint.ckpt-111241"
# ckpt_dir = "./checkpoints"
ckpt = '/home/jaehyuk/code/experiment/VIDonline/checkpoints/checkpoint.ckpt-37081'
selected_weight = True

step_dir = "/home/jaehyuk/code/github/vot-toolkit/tracker/examples/python/step"
#  descend!
stlist = ['/home/jaehyuk/code/github/vot-toolkit/tracker/examples/python/step/step.100.txt',
          '/home/jaehyuk/code/github/vot-toolkit/tracker/examples/python/step/step.50.txt',
          '/home/jaehyuk/code/github/vot-toolkit/tracker/examples/python/step/step.40.txt',
          '/home/jaehyuk/code/github/vot-toolkit/tracker/examples/python/step/step.25.txt',
          '/home/jaehyuk/code/github/vot-toolkit/tracker/examples/python/step/step.10.txt',
          '/home/jaehyuk/code/github/vot-toolkit/tracker/examples/python/step/step.5.txt',
          '/home/jaehyuk/code/github/vot-toolkit/tracker/examples/python/step/step.1.txt']

step = 1
for st in stlist:
    if os.path.exists(st):
        step = int(st.split('.')[1])
        break

# debug
step = 50

imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

bbox_estim = bbox_estimator(False, logger)
tracknet = goturn_net_coord.TRACKNET(BATCH_SIZE, train=True, online=True)
tracknet.build()

# TODO check trainiable_variables
tvars = tf.trainable_variables()
g_vars = [var for var in tvars if 'fc1_image' or 'fc2_image' or 'fc3_image' or 'fc4_image' in var.name]
# g_vars = [var for var in tvars if 'fc1_image' in var.name]
train_step = tf.train.AdamOptimizer(1e-4).minimize(tracknet.loss)


sess = tf.Session()
init = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()
sess.run(init)
sess.run(init_local)

coord = tf.train.Coordinator()
# start the threads
tf.train.start_queue_runners(sess=sess, coord=coord)

# # it is too long
# all_ckpt_meta = glob.glob(os.path.join(ckpt_dir, '*.meta'))
# num = []
# for ckpt_meta in all_ckpt_meta:
#     num.append(int(ckpt_meta.split('-')[1].split('.')[0]))
#
# max_num = max(num)
# ckpt = os.path.join(ckpt_dir, 'checkpoint.ckpt-' + str(max_num))


if ckpt:
    if not selected_weight:
        saver = tf.train.Saver()
        saver.restore(sess, ckpt)
        logger.info(str(ckpt) + " is restored")

    # configurate weight to be initialized
    else:
        restore = {}
        from tensorflow.contrib.framework.python.framework.checkpoint_utils import list_variables

        slim = tf.contrib.slim
        for scope in list_variables(ckpt):
            if 'conv' or 'fc1_adj' or 'fc2_adj' or 'fc3_adj' or 'fc4_adj' in scope[0]:
                variables_to_restore = slim.get_variables(scope=scope[0])
                if variables_to_restore:
                    restore[scope[0]] = variables_to_restore[0]  # variables_to_restore is list : [op]
        saver = tf.train.Saver(restore)
        saver.restore(sess, ckpt)
        logger.info("model is restored selected weight using " + str(ckpt))
else:
    raise()


image = cv2.imread(imagefile)
bbox_estim.init(image, selection)
while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile)
    bbox = bbox_estim.track(image, tracknet, train_step, step, sess)
    handle.report(bbox)

