import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN

from model.roi_layers import ROIAlign, ROIPool

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
import numpy.random as npr
import random

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

    def forward(self, im_data, im_info, gt_boxes, num_boxes, th1=None, th2=None, epoch=None):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        #  --------------------------------------------------------------------------------------------
        count_num = [0, 0, 0, 0]
        if self.training:
            self.eval()
            pooled_feat_tmp = pooled_feat.clone().detach()
            pooled_feat_tmp = Variable(pooled_feat_tmp.data, requires_grad=True)
            pooled_feat_new = self._head_to_tail(pooled_feat_tmp)
            output = self.RCNN_cls_score(pooled_feat_new)
            class_num = output.shape[1]
            index = rois_label
            num_rois = pooled_feat_tmp.shape[0]
            num_channel = pooled_feat_tmp.shape[1]
            one_hot = torch.zeros((1), dtype=torch.float32).cuda()
            one_hot = Variable(one_hot, requires_grad=False)
            sp_i = torch.ones([2, num_rois]).long()
            sp_i[0, :] = torch.arange(num_rois)
            sp_i[1, :] = index
            sp_v = torch.ones([num_rois])
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
            one_hot_sparse = Variable(one_hot_sparse, requires_grad=False)  # [n, 21]
            one_hot = torch.sum(output * one_hot_sparse)
            self.zero_grad()
            one_hot.backward()
            grads_val = pooled_feat_tmp.grad.clone().detach()
            grad_channel_mean = torch.mean(grads_val.view(num_rois, num_channel, -1), dim=2)
            cam_all = torch.sum(pooled_feat_tmp * grad_channel_mean.view(num_rois, num_channel, 1, 1), 1)
            cam_all = cam_all.view(num_rois, 49)
            self.zero_grad()

            # -------------------------IA ----------------------------
            num_s = 18
            th_mask_value = torch.sort(cam_all, dim=1, descending=True)[0][:, num_s]
            th_mask_value = th_mask_value.view(num_rois, 1).expand(num_rois, 49)
            mask_all_cuda = torch.where(cam_all > th_mask_value, torch.zeros(cam_all.shape).cuda(), torch.ones(cam_all.shape).cuda())
            mask_all = mask_all_cuda.reshape(num_rois, 7, 7).view(num_rois, 1, 7, 7)

            # ------------------------ batch ---------------------
            pooled_feat_before_after = torch.cat((pooled_feat_tmp, pooled_feat_tmp * mask_all), dim=0)
            pooled_feat_before_after = self._head_to_tail(pooled_feat_before_after)
            cls_score_before_after = self.RCNN_cls_score(pooled_feat_before_after)
            cls_prob_before_after = F.softmax(cls_score_before_after, dim=1)
            cls_prob_before = cls_prob_before_after[0: num_rois]
            cls_prob_after = cls_prob_before_after[num_rois: num_rois * 2]

            prepare_mask_fg_num = rois_label.nonzero().size(0)
            prepare_mask_bg_num = num_rois - prepare_mask_fg_num

            sp_i = torch.ones([2, num_rois]).long()
            sp_i[0, :] = torch.arange(num_rois)
            sp_i[1, :] = rois_label
            sp_v = torch.ones([num_rois])
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()

            before_vector = torch.sum(one_hot_sparse * cls_prob_before, dim=1)
            after_vector = torch.sum(one_hot_sparse * cls_prob_after, dim=1)
            change_vector = before_vector - after_vector - 0.01
            change_vector = torch.where(change_vector > 0, change_vector, torch.zeros(change_vector.shape).cuda())

            fg_index = torch.where(rois_label > 0, torch.ones(change_vector.shape).cuda(), torch.zeros(change_vector.shape).cuda())
            bg_index = 1 - fg_index
            if fg_index.nonzero().shape[0] != 0:
                not_01_fg_index = fg_index.nonzero()[:, 0].long()
            else:
                not_01_fg_index = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).cuda().long()
            not_01_bg_index = bg_index.nonzero()[:, 0].long()
            change_vector_fg = change_vector[not_01_fg_index]
            change_vector_bg = change_vector[not_01_bg_index]

            for_fg_change_vector = change_vector.clone()
            for_bg_change_vector = change_vector.clone()
            for_fg_change_vector[not_01_bg_index] = -10000
            for_bg_change_vector[not_01_fg_index] = -10000

            th_fg_value = torch.sort(change_vector_fg, dim=0, descending=True)[0][int(round(float(prepare_mask_fg_num) / 5))]
            drop_index_fg = for_fg_change_vector.gt(th_fg_value)
            th_bg_value = torch.sort(change_vector_bg, dim=0, descending=True)[0][int(round(float(prepare_mask_bg_num) / 30))]
            drop_index_bg = for_bg_change_vector.gt(th_bg_value)
            drop_index_fg_bg = drop_index_fg + drop_index_bg
            ignore_index_fg_bg = 1 - drop_index_fg_bg
            not_01_ignore_index_fg_bg = ignore_index_fg_bg.nonzero()[:, 0]
            mask_all[not_01_ignore_index_fg_bg.long(), :] = 1

            # ---------------------------------------------------------
            self.train()
            mask_all = Variable(mask_all, requires_grad=True)
            pooled_feat = pooled_feat * mask_all

        #-------------------------------------------------------------------------------------------------------

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
