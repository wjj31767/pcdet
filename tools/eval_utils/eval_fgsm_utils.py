import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])

def fgsm(epsilon, batch_dict, model, ord, iterations, pgd=False, momentum=None, clip_norm=None):
    global original_voxels
    alpha = epsilon / iterations
    if ord == "inf":
        ord_fn = torch.sign
    elif ord == "1":
        ord_fn = lambda x: x / torch.sum(torch.abs(x), dim=2, keepdim=True)
    elif ord == "2":
        ord_fn = lambda x: x / torch.sqrt(torch.sum(x ** 2, dim=2, keepdim=True))
    elif ord == "2.5":
        def ord_fn(x):
            norm = torch.norm(x, dim=-1, keepdim=True)
            return torch.where(
                torch.eq(norm, torch.zeros(norm.shape).float().cuda()) & torch.ones(x.shape).bool().cuda(),
                torch.zeros_like(x).cuda(), x / norm)
    else:
        raise ValueError("Only L-inf, L1, L2, and normalized L2 norms are supported!")
    if pgd:
        original_voxels = batch_dict['voxels'].clone()
        example_voxels_adv = original_voxels + (torch.rand(batch_dict['voxels'].size(), dtype=batch_dict['voxels'].dtype,
                                                              device=batch_dict['voxels'].device) - 0.5) * epsilon / 2
    if momentum:
        prev_grad = torch.zeros_like(batch_dict['voxels'])
    for _ in range(iterations):
        if pgd:
            batch_dict['voxels'] = example_voxels_adv
        batch_dict['voxels'].requires_grad_(True)
        for cur_module in model.module_list:
            batch_dict = cur_module(batch_dict)
        model.dense_head.forward_ret_dict.update(model.dense_head.assign_targets(gt_boxes=batch_dict['gt_boxes']))
        loss, _, _ = model.get_training_loss()
        model.zero_grad()
        loss.mean().backward()
        grad = batch_dict['voxels'].grad.data.clone()
        if momentum:
            grad = grad / torch.mean(torch.abs(grad), dim=1, keepdim=True)
            grad = momentum * prev_grad + grad
            prev_grad = grad.clone()
        perturb = alpha * ord_fn(grad)

        batch_dict['voxels'] = batch_dict['voxels'].detach()
        true_tensor = torch.ones(batch_dict['voxels'].shape, dtype=batch_dict['voxels'].dtype).cuda()
        false_tensor = torch.zeros(batch_dict['voxels'].shape, dtype=batch_dict['voxels'].dtype).cuda()
        mask = torch.where(batch_dict['voxels'] != 0., true_tensor, false_tensor).bool()
        batch_dict['voxels'][mask.any(-1)] += perturb[mask.any(-1)]
        if pgd:
            batch_dict['voxels'] = original_voxels+torch.clamp(batch_dict['voxels']-original_voxels,-alpha,alpha)
def rectification_pointcloud(example, original_voxels):
    true_tensor = torch.ones(example['voxels'].shape, dtype=example['voxels'].dtype).cuda()
    false_tensor = torch.zeros(example['voxels'].shape, dtype=example['voxels'].dtype).cuda()
    mask = torch.where(example['voxels'] != 0., true_tensor, false_tensor).bool()
    example['voxels'][mask.any(-1)][:, :3] = original_voxels[mask.any(-1)][:, :3] * torch.sum(
        example['voxels'][mask.any(-1)][:, :3] * original_voxels[mask.any(-1)][:, :3], dim=1, keepdim=True) / torch.sum(
        original_voxels[mask.any(-1)][:, :3] * original_voxels[mask.any(-1)][:, :3], dim=1, keepdim=True)
    example['voxels'][:, 3] = original_voxels[:, 3]


def rectification_angle(example, original_voxels):
    example['voxels'][:, :3] = original_voxels[:, :3]


def rectification(example, original_voxels):
    true_tensor = torch.ones(example['voxels'].shape, dtype=example['voxels'].dtype).cuda()
    false_tensor = torch.zeros(example['voxels'].shape, dtype=example['voxels'].dtype).cuda()
    mask = torch.where(example['voxels'] != 0., true_tensor, false_tensor).bool()
    example['voxels'][mask.any(-1)][:, :3] = original_voxels[mask.any(-1)][:, :3] * torch.sum(
        example['voxels'][mask.any(-1)][:, :3] * original_voxels[mask.any(-1)][:, :3], dim=1, keepdim=True) / torch.sum(
        original_voxels[mask.any(-1)][:, :3] * original_voxels[mask.any(-1)][:, :3], dim=1, keepdim=True)

def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, epsilon, ord, iterations, rec_type, pgd=True,momentum=False,dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        original_voxels = batch_dict['voxels'].clone()
        fgsm(epsilon, batch_dict, model, ord, iterations, pgd=True,momentum=False)
        # rectification on both
        if rec_type=="both":
            rectification(batch_dict, original_voxels)
        # rectification just on pointcloud
        elif rec_type=="points":
            rectification_pointcloud(batch_dict, original_voxels)
        # rectification just on intencity
        elif rec_type=="intencity":
            rectification_angle(batch_dict, original_voxels)
        else:
            raise ValueError("3 type of rectification is accepted")
        with torch.no_grad():
            pred_dicts,ret_dict = model(batch_dict)

        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    # logger.info(result_str)
    print(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
