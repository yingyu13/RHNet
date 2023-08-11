import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import ConfigDict
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmcv.ops import batched_nms
from mmcv.ops import DeformConv2d
from mmdet.core import (MlvlPointGenerator, distance2bbox,
                        filter_scores_and_topk, select_single_mlvl)
from mmdet.core import multi_apply, reduce_mean
from mmdet.models.dense_heads.paa_head import levels_to_images
# from mmdet.models import HEADS, AnchorFreeHead
from mmdet.models.utils import sigmoid_geometric_mean
from mmcv.ops import nms_rotated
from mmrotate.models.builder import ROTATED_HEADS
from mmrotate.core import build_bbox_coder, multiclass_nms_rotated
from mmrotate.models.dense_heads.rotated_anchor_free_head import RotatedAnchorFreeHead
from mmcv.ops import nms_rotated
from mmrotate.models.detectors.utils import align_tensor
from ..builder import ROTATED_HEADS, build_loss
from mmcv.ops import diff_iou_rotated_2d
from mmrotate.core import build_bbox_coder
from mmrotate.core.bbox.builder import ROTATED_BBOX_ASSIGNERS
from mmrotate.core.bbox.iou_calculators import build_iou_calculator
INF = 1e8
eps=1e-6

@ROTATED_HEADS.register_module()
class DynamicQueryGen(RotatedAnchorFreeHead):
    def __init__(
            self,
            num_classes,
            in_channels,
            strides=(8, 16, 32, 64, 128),
            regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                            (512, INF)),
            center_sampling=False,
            center_sample_radius=1.5,
            bbox_coder=dict(type='DistancePointBBoxCoder'),
            num_queries=300,
            loss_cls=dict(
                    type='QualityFocalLoss',
                    use_sigmoid=True,
                    activated=True,
                    beta=2.0,
                    loss_weight=1.0),
            loss_bbox=dict(type='RotatedIoULoss', loss_weight=2.0),
            scale_angle=True,
            nms_cfg=dict(iou_thr=0.2, nms_pre=1000),
            offset=0.5,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            init_cfg=dict(
                type='Normal',
                layer='Conv2d',
                std=0.01,
                override=dict(
                    type='Normal',
                    name='conv_cls',
                    std=0.01,
                    bias_prob=0.01)),
            **kwargs):

        self.is_scale_angle = scale_angle

        super().__init__(num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg)

        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.num_queries = num_queries
        self.nms_cfg = nms_cfg

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        self.init_weights()
        self.prior_generator = MlvlPointGenerator(strides, offset=offset)

    def init_weights(self):
        """Initialize weights of the head."""
        bias_cls = bias_init_with_prob(0.01)
        for m in self.cls_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.reg_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for layer in self.objectness.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)
            if isinstance(layer, nn.GroupNorm):
                torch.nn.init.constant_(layer.weight, 1)
                torch.nn.init.constant_(layer.bias, 0)

        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_reg, std=0.01)
        normal_init(self.conv_angle, std=0.01)

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs-1):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(chn,
                           self.feat_channels,
                           3,
                           stride=1,
                           padding=3 // 2,
                           conv_cfg=self.conv_cfg,
                           norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(chn,
                           self.feat_channels,
                           3,
                           stride=1,
                           padding=3 // 2,
                           conv_cfg=self.conv_cfg,
                           norm_cfg=self.norm_cfg))

        self.objectness = nn.Sequential(
            nn.Conv2d(self.feat_channels, self.feat_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_channels // 4, 1, 3, padding=3 // 2))

        cls_out_channels = self.num_classes

        self.conv_cls = nn.Conv2d(self.feat_channels,
                                  self.num_base_priors * cls_out_channels,
                                  3,
                                  padding=3 // 2)

        self.conv_reg = nn.Conv2d(self.feat_channels,
                                  self.num_base_priors * 4,
                                  3,
                                  padding=3 // 2)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides])

        cls_out_channels = self.num_classes

        self.conv_angle = nn.Conv2d(self.feat_channels, self.num_base_priors * 1, 3, padding=1)      
        if self.is_scale_angle:
            self.scale_angle = Scale(1.0)

        self.compress = nn.Linear(self.feat_channels * 2, self.feat_channels)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales, self.strides)

    def forward_single(self, x, scale, stride):
        cls_feat = x
        reg_feat = x
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)

        rbox_pred = scale(self.conv_reg(reg_feat).exp()).float()
        angle_pred = self.conv_angle(reg_feat)
        if self.is_scale_angle:
            angle_pred = self.scale_angle(angle_pred).float()

        rbox_pred = torch.cat([rbox_pred, angle_pred],
                                        dim=1)

        cls_logits = self.conv_cls(cls_feat)
        object_nesss = self.objectness(reg_feat)
        cls_score = sigmoid_geometric_mean(cls_logits, object_nesss)

        return cls_score, rbox_pred, cls_feat, reg_feat

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        outs = self.forward(x)

        ori_cls_scores, ori_bbox_preds, all_query_bboxes, all_object_feats, mlvl_priors = self.query_select(*outs, img_metas)

        return ori_cls_scores, ori_bbox_preds, all_query_bboxes, all_object_feats, mlvl_priors

    def query_select(self, cls_scores, bbox_preds, cls_feats, reg_feats, img_metas=None, **kwargs):

        num_levels = len(cls_scores)
        num_imgs = len(img_metas)
        channels = cls_feats[0].shape[1]
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)
        all_object_feats = []
        all_query_bboxes = []
        ori_cls_scores = []
        ori_bbox_preds = []
        for img_id in range(len(img_metas)):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            rbbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]

            cls_feats_list = [cls_feats[i].permute(0, 2, 3, 1).view(num_imgs, -1, channels).contiguous()[img_id].detach() for i in range(num_levels)]
            reg_feats_list = [reg_feats[i].permute(0, 2, 3, 1).view(num_imgs, -1, channels).contiguous()[img_id].detach() for i in range(num_levels)]
            cls_feats_list = torch.cat(cls_feats_list, dim=0)
            reg_feats_list = torch.cat(reg_feats_list, dim=0)

            ori_bboxes_single, ori_scores_single, query_bboxes_single, query_scores_single, query_inds_single = self.query_select_single(
                cls_score_list, rbbox_pred_list, mlvl_priors, img_metas[img_id])

            ori_cls_scores.append(ori_scores_single)
            ori_bbox_preds.append(ori_bboxes_single)

            query_scores_single = query_scores_single.max(-1).values

            object_feats = torch.cat([cls_feats_list, reg_feats_list],
                                     dim=-1)

            object_feats = object_feats.detach()
            query_bboxes_single = query_bboxes_single.detach()

            object_feats = self.compress(object_feats)

            select_ids = torch.sort(query_scores_single,
                                    descending=True).indices[:self.num_queries]
            query_inds_single = query_inds_single[select_ids]
            query_bboxes_single = query_bboxes_single[select_ids]

            object_feats = object_feats[query_inds_single]

            all_object_feats.append(object_feats)
            all_query_bboxes.append(query_bboxes_single)

        all_object_feats = align_tensor(all_object_feats)
        all_query_bboxes = align_tensor(all_query_bboxes)

        return ori_cls_scores, ori_bbox_preds, all_query_bboxes, all_object_feats, mlvl_priors

    def query_select_single(self, cls_score_list, bbox_pred_list, mlvl_priors, img_meta,
                  **kwargs):
        ori_bboxes = []
        ori_scores = []
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_query_inds = []
        start_inds = 0
        for level_idx, (cls_score, bbox_pred, priors, stride) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                     mlvl_priors, \
                        self.prior_generator.strides)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.num_classes)
            ori_bboxes.append(bbox_pred)
            ori_scores.append(cls_score)
            binary_cls_score = cls_score.max(-1).values.reshape(-1, 1)
            nms_pre = self.nms_cfg.pop('nms_pre', 1000)
            bbox_pred = self.bbox_coder.decode(priors, bbox_pred)
            results = filter_scores_and_topk(
                binary_cls_score, 0, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors, cls_score=cls_score))
            scores, labels, keep_idxs, filtered_results = results
            keep_idxs = keep_idxs + start_inds
            start_inds = start_inds + len(cls_score)
            bbox_pred = filtered_results['bbox_pred']
            cls_score = filtered_results['cls_score']

            mlvl_bboxes.append(bbox_pred)
            mlvl_scores.append(cls_score)
            mlvl_query_inds.append(keep_idxs)

        ori_scores = torch.cat(ori_scores)
        ori_bboxes = torch.cat(ori_bboxes)

        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_query_inds = torch.cat(mlvl_query_inds)

        det_bboxes, keep_idxs = nms_rotated(mlvl_bboxes,
                                        mlvl_scores.max(-1).values, self.nms_cfg['iou_thr'])
        return ori_bboxes, ori_scores, mlvl_bboxes[keep_idxs], mlvl_scores[keep_idxs], mlvl_query_inds[keep_idxs]

    def loss(self, cls_scores, bbox_preds, all_priors, per_img_prior, gt_bboxes, gt_labels, img_metas,
                 **kwargs):
                 
        flatten_cls_scores = cls_scores
        flatten_bbox_preds = bbox_preds

        cls_reg_targets = self.get_targets(
            all_priors,
            gt_bboxes,
            img_metas,
            gt_labels_list=gt_labels,
        )
        (labels_list, bbox_targets_list) = cls_reg_targets

        losses_cls, losses_bbox,\
            cls_avg_factors, bbox_avg_factors = multi_apply(
                self.loss_single,
                per_img_prior,
                flatten_cls_scores,
                flatten_bbox_preds,
                labels_list,
                bbox_targets_list,
                )
        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

        bbox_avg_factor = reduce_mean(
            sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def loss_single(self, per_img_prior, cls_score, bbox_pred, labels, 
                    bbox_targets):

        bbox_targets = bbox_targets.reshape(-1, 5)
        labels = labels.reshape(-1)
        cls_loss_func = self.loss_cls
        score = labels.new_zeros(labels.shape, dtype=torch.float32)
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = cls_score.size(-1)
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_pred.device)
        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]

            pos_decode_bbox_pred = pos_bbox_pred

            pos_points = per_img_prior[pos_inds]
            pos_decode_bbox_targets = self.bbox_coder.decode(
                pos_points, pos_bbox_targets)

            with torch.no_grad():
                iou = diff_iou_rotated_2d(pos_decode_bbox_pred.unsqueeze(0), pos_decode_bbox_targets.unsqueeze(0))
                iou = iou.squeeze(0).clamp(min=eps)

            score[pos_inds] = iou
            pos_bbox_weight = iou

            loss_bbox = self.loss_bbox(pos_decode_bbox_pred,
                                       pos_decode_bbox_targets,
                                       weight=pos_bbox_weight,
                                       avg_factor=1.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            pos_bbox_weight = bbox_targets.new_tensor(0.)

        targets = (labels, score)
        loss_cls = cls_loss_func(cls_score,
                                 targets,
                                 avg_factor=1.0)

        return loss_cls, loss_bbox, num_pos, pos_bbox_weight.sum()

    def get_targets(self, 
                    points,                     
                    gt_bboxes_list,
                    img_metas,
                    gt_labels_list=None):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
                concat_lvl_angle_targets (list[Tensor]): Angle targets of \
                    each level.
        """
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]
        
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        return (labels_list, bbox_targets_list)

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression, classification and angle targets for a single
        image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((num_points, 1))

        areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)

        cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
        rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                               dim=-1).reshape(num_points, num_gts, 2, 2)
        offset = points - gt_ctr
        offset = torch.matmul(rot_matrix, offset[..., None])
        offset = offset.squeeze(-1)

        w, h = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            stride = offset.new_zeros(offset.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            inside_center_bbox_mask = (abs(offset) < stride).all(dim=-1)
            inside_gt_bbox_mask = torch.logical_and(inside_center_bbox_mask,
                                                    inside_gt_bbox_mask)

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        angle_targets = gt_angle[range(num_points), min_area_inds]
        bbox_targets = torch.cat([bbox_targets, angle_targets], dim=-1)

        return labels, bbox_targets
