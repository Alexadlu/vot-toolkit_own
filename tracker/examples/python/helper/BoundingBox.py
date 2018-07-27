# Date: Tuesday 06 June 2017 03:54:55 PM
# Email: nrupatunga@whodat.com
# Name: Nrupatunga
# Description: bounding box class

import numpy as np
import sys
sys.path.append('/home/jaehyuk/code/github/vot-toolkit/tracker/examples/python/helper')
from helper import sample_exp_two_sides, sample_rand_uniform
from config import POLICY


class BoundingBox:
    """Docstring for BoundingBox. """

    def __init__(self, x1, y1, x2, y2):
        """bounding box """

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.frame_num = 0
        self.kContextFactor = 2
        self.kScaleFactor = 10

    def get_center_x(self):
        """TODO: Docstring for get_center_x.
        :returns: TODO

        """
        return (self.x1 + self.x2) / 2.

    def get_center_y(self):
        """TODO: Docstring for get_center_y.
        :returns: TODO

        """
        return (self.y1 + self.y2) / 2.

    def compute_output_height(self):
        """TODO: Docstring for compute_output_height.
        :returns: TODO

        """
        bbox_height = self.y2 - self.y1
        output_height = self.kContextFactor * bbox_height

        return max(1.0, output_height)

    def compute_output_width(self):
        """TODO: Docstring for compute_output_width.
        :returns: TODO

        """
        bbox_width = self.x2 - self.x1
        output_width = self.kContextFactor * bbox_width

        return max(1.0, output_width)

    def edge_spacing_x(self):
        """TODO: Docstring for edge_spacing_x.
        :returns: TODO

        """
        output_width = self.compute_output_width()
        bbox_center_x = self.get_center_x()

        return max(0.0, (output_width / 2) - bbox_center_x)

    def edge_spacing_y(self):
        """TODO: Docstring for edge_spacing_y.
        :returns: TODO

        """
        output_height = self.compute_output_height()
        bbox_center_y = self.get_center_y()

        return max(0.0, (output_height / 2) - bbox_center_y)

    def unscale(self, image):
        """TODO: Docstring for unscale.
        :returns: TODO

        """
        height = image.shape[0]
        width = image.shape[1]

        self.x1 = self.x1 / self.kScaleFactor
        self.x2 = self.x2 / self.kScaleFactor
        self.y1 = self.y1 / self.kScaleFactor
        self.y2 = self.y2 / self.kScaleFactor

        self.x1 = self.x1 * width
        self.x2 = self.x2 * width
        self.y1 = self.y1 * height
        self.y2 = self.y2 * height

    def uncenter(self, raw_image, search_location, edge_spacing_x, edge_spacing_y):
        """TODO: Docstring for uncenter.
        :returns: TODO

        """
        self.x1 = max(0.0, self.x1 + search_location.x1 - edge_spacing_x)
        self.y1 = max(0.0, self.y1 + search_location.y1 - edge_spacing_y)
        self.x2 = min(raw_image.shape[1], self.x2 + search_location.x1 - edge_spacing_x)
        self.y2 = min(raw_image.shape[0], self.y2 + search_location.y1 - edge_spacing_y)

    def recenter(self, search_loc, edge_spacing_x, edge_spacing_y, bbox_gt_recentered):
        """TODO: Docstring for recenter.
        :returns: TODO

        """
        bbox_gt_recentered.x1 = self.x1 - search_loc.x1 + edge_spacing_x
        bbox_gt_recentered.y1 = self.y1 - search_loc.y1 + edge_spacing_y
        bbox_gt_recentered.x2 = self.x2 - search_loc.x1 + edge_spacing_x
        bbox_gt_recentered.y2 = self.y2 - search_loc.y1 + edge_spacing_y

        return bbox_gt_recentered

    def scale(self, image):
        """TODO: Docstring for scale.
        :returns: TODO

        """
        height = image.shape[0]
        width = image.shape[1]

        self.x1 = self.x1 / width
        self.y1 = self.y1 / height
        self.x2 = self.x2 / width
        self.y2 = self.y2 / height

        self.x1 = self.x1 * self.kScaleFactor
        self.y1 = self.y1 * self.kScaleFactor
        self.x2 = self.x2 * self.kScaleFactor
        self.y2 = self.y2 * self.kScaleFactor

    def get_width(self):
        """TODO: Docstring for get_width.
        :returns: TODO

        """
        return (self.x2 - self.x1)

    def get_height(self):
        """TODO: Docstring for get_width.
        :returns: TODO

        """
        return (self.y2 - self.y1)

    def shift(self, image, lambda_scale_frac, lambda_shift_frac, min_scale, max_scale, shift_motion_model, bbox_rand):
        """TODO: Docstring for shift.
        :returns: TODO

        """
        width = self.get_width()
        height = self.get_height()

        center_x = self.get_center_x()
        center_y = self.get_center_y()

        kMaxNumTries = 10

        new_width = -1
        num_tries_width = 0
        while ((new_width < 0) or (new_width > image.shape[1] - 1)) and (num_tries_width < kMaxNumTries):
            if shift_motion_model:
                width_scale_factor = max(min_scale, min(max_scale, sample_exp_two_sides(lambda_scale_frac)))
            else:
                rand_num = sample_rand_uniform()
                width_scale_factor = rand_num * (max_scale - min_scale) + min_scale

            new_width = width * (1 + width_scale_factor)
            new_width = max(1.0, min((image.shape[1] - 1), new_width))
            num_tries_width = num_tries_width + 1

        new_height = -1
        num_tries_height = 0
        while ((new_height < 0) or (new_height > image.shape[0] - 1)) and (num_tries_height < kMaxNumTries):
            if shift_motion_model:
                height_scale_factor = max(min_scale, min(max_scale, sample_exp_two_sides(lambda_scale_frac)))
            else:
                rand_num = sample_rand_uniform()
                height_scale_factor = rand_num * (max_scale - min_scale) + min_scale

            new_height = height * (1 + height_scale_factor)
            new_height = max(1.0, min((image.shape[0] - 1), new_height))
            num_tries_height = num_tries_height + 1

        first_time_x = True
        new_center_x = -1
        num_tries_x = 0

        while (first_time_x or (new_center_x < center_x - width * self.kContextFactor / 2)
               or (new_center_x > center_x + width * self.kContextFactor / 2)
               or ((new_center_x - new_width / 2) < 0)
               or ((new_center_x + new_width / 2) > image.shape[1])
               and (num_tries_x < kMaxNumTries)):

            if shift_motion_model:
                new_x_temp = center_x + width * sample_exp_two_sides(lambda_shift_frac)
            else:
                rand_num = sample_rand_uniform()
                new_x_temp = center_x + rand_num * (2 * new_width) - new_width

            new_center_x = min(image.shape[1] - new_width / 2, max(new_width / 2, new_x_temp))
            first_time_x = False
            num_tries_x = num_tries_x + 1

        first_time_y = True
        new_center_y = -1
        num_tries_y = 0

        while (first_time_y or (new_center_y < center_y - height * self.kContextFactor / 2)
               or (new_center_y > center_y + height * self.kContextFactor / 2)
               or ((new_center_y - new_height / 2) < 0)
               or ((new_center_y + new_height / 2) > image.shape[0])
               and (num_tries_y < kMaxNumTries)):

            if shift_motion_model:
                new_y_temp = center_y + height * sample_exp_two_sides(lambda_shift_frac)
            else:
                rand_num = sample_rand_uniform()
                new_y_temp = center_y + rand_num * (2 * new_height) - new_height

            new_center_y = min(image.shape[0] - new_height / 2, max(new_height / 2, new_y_temp))
            first_time_y = False
            num_tries_y = num_tries_y + 1

        bbox_rand.x1 = new_center_x - new_width / 2
        bbox_rand.x2 = new_center_x + new_width / 2
        bbox_rand.y1 = new_center_y - new_height / 2
        bbox_rand.y2 = new_center_y + new_height / 2

        return bbox_rand


def expit_tensor(x):
	return 1. / (1. + np.exp(-x))


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list

    boxes = np.array(boxes)
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # float data type
    return boxes[pick]

def IOU_suppresion(boxes, prev_box, overlapThresh):
    # if there are no boxes, return an empty list

    boxes = np.array(boxes)
    if len(boxes) == 0:
        return [], 0.

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box

    prev_box_area = (prev_box[2] - prev_box[0]) * (prev_box[3] - prev_box[1])


    # find the largest (x, y) coordinates for the start of
    # the bounding box and the smallest (x, y) coordinates
    # for the end of the bounding box
    xx1 = np.maximum(x1, prev_box[0])
    yy1 = np.maximum(y1, prev_box[1])
    xx2 = np.minimum(x2, prev_box[2])
    yy2 = np.minimum(y2, prev_box[3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    # compute the ratio of overlap
    overlap = (w * h) / prev_box_area
    overlap_score = np.max(overlap)
    IOU_boxes = boxes[np.argmax(overlap)]


    # return only the bounding boxes that were picked using the
    # float data type
    return IOU_boxes, overlap_score

def calculate_box(re_fc4_image, fc4_adj):
    """

    :param re_fc4_image: [1, H, W, 1]
    :param fc4_adj: [1, 4]
    :return:
    """
    H, W = POLICY['side'], POLICY['side']
    B = POLICY['num']
    anchors = POLICY['anchors']
    w, h = 10, 10
    # TODO, compare with tf implementation
    # calculate box
    fc4_adj = np.reshape(fc4_adj, [4])
    re_fc4_image = np.reshape(re_fc4_image, [H, W, 1])

    adjusted_coords_xy = expit_tensor(fc4_adj[0:2])
    adjusted_coords_wh = np.exp(fc4_adj[2:4]) * np.reshape(anchors, [2]) / np.reshape([W, H], [2])

    adjusted_c = expit_tensor(re_fc4_image)

    # find max objscore box TODO, if you need NMS, add it
    # top_obj_indexs = np.where(adjusted_c > POLICY['thresh'])
    top_obj_indexs = np.where(adjusted_c == np.max(adjusted_c))

    # debug, print top 10 objectscore

    objectness_s = adjusted_c[top_obj_indexs]

    pred_box=[]
    object_bool = False;

    if objectness_s.any() > POLICY['thresh']:
        object_bool = True

    for idx, objectness in np.ndenumerate(objectness_s):
        pred_cx = (float(top_obj_indexs[1][idx]) + adjusted_coords_xy[0]) / W * w
        pred_cy = (float(top_obj_indexs[0][idx]) + adjusted_coords_xy[1]) / H * h
        pred_w = adjusted_coords_wh[0] * w
        pred_h = adjusted_coords_wh[1] * h
        pred_obj = objectness

        pred_xl = pred_cx - pred_w / 2
        pred_yl = pred_cy - pred_h / 2
        pred_xr = pred_cx + pred_w / 2
        pred_yr = pred_cy + pred_h / 2

        pred_box.append([pred_xl, pred_yl, pred_xr, pred_yr, pred_obj])

    NMS_pred_box = non_max_suppression_fast(pred_box, POLICY['thresh_IOU'])

    # prev_box = [2.5, 2.5, 7.5, 7.5]
    # IOU_pred_box, overlap_score = IOU_suppresion(pred_box, prev_box, POLICY['thresh_IOU'])
    # if not len(IOU_pred_box) == 0:
        # IOU_pred_box = np.expand_dims(IOU_pred_box, axis=0)

    # type float, array
    return NMS_pred_box, object_bool, objectness_s# IOU_pred_box
