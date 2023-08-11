from mmrotate.models.builder import ROTATED_DETECTORS
# from mmdet.models.detectors.two_stage import TwoStageDetector
from mmrotate.models.detectors.two_stage import RotatedTwoStageDetector
from mmrotate.core import rbbox2result
import torch

@ROTATED_DETECTORS.register_module()
class RHNet(RotatedTwoStageDetector):
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        losses = dict()
        x = self.extract_feat(img)
        # remove p2 for rpn
        rpn_x = x[1:]
        roi_x = x

        cls_scores, bbox_preds, proposals, object_feats, mlvl_priors = \
            self.rpn_head.forward_train(
                rpn_x,
                img_metas,
                gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore)

        imgs_whwh = []
        per_img_prior = []
        for meta in img_metas:
            h, w, _ = meta['img_shape']
            imgs_whwh.append(x[0].new_tensor([[w, h, w, h, 1]]))
            per_img_prior.append(torch.cat(mlvl_priors))
        imgs_whwh = torch.cat(imgs_whwh, dim=0)
        imgs_whwh = imgs_whwh[:, None, :]

        loss_inputs = (cls_scores, bbox_preds, mlvl_priors, per_img_prior, gt_bboxes, gt_labels, img_metas)
        rpn_losses = self.rpn_head.loss(*loss_inputs)

        for k, v in rpn_losses.items():
            losses[f'rpn_{k}'] = v

        roi_losses = self.roi_head.forward_train(
            roi_x,
            proposals,
            object_feats,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            imgs_whwh=imgs_whwh)
        losses.update(roi_losses)

        return losses

    def simple_test(self, img, img_metas, rescale=True):
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        x = self.extract_feat(img)
        # remove p2 for rpn
        rpn_x = x[1:]
        roi_x = x

        outs, proposals, object_feats, mlvl_priors = \
            self.rpn_head.simple_test_rpn(
                rpn_x,
                img_metas,
                )

        imgs_whwh = []
        for meta in img_metas:
            h, w, _ = meta['img_shape']
            imgs_whwh.append(x[0].new_tensor([[w, h, w, h, 1]]))
        imgs_whwh = torch.cat(imgs_whwh, dim=0)
        imgs_whwh = imgs_whwh[:, None, :]

        results = self.roi_head.simple_test(roi_x,
                                            proposals,
                                            object_feats,
                                            img_metas,
                                            imgs_whwh=imgs_whwh,
                                            rescale=rescale)
        return results
