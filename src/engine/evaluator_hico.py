import math
import os
import sys
from typing import Iterable
import numpy as np
import copy
import itertools
import matplotlib.pyplot as plt
import torch

import src.util.misc as utils
import src.util.logger as loggers
from src.data.evaluators.hico_eval import HICOEvaluator
from src.models.stip_utils import check_annotation, plot_cross_attention, plot_hoi_results

@torch.no_grad()
def hico_evaluate(model, recon_model, resnet, centroids, postprocessors, data_loader, device, thr, args):
    model.eval()

    metric_logger = loggers.MetricLogger(mode="test", delimiter="  ")
    header = 'Evaluation Inference (HICO-DET)'

    preds = []
    gts = []
    indices = []
    hoi_recognition_time = []

    for samples, targets in metric_logger.log_every(data_loader, 50, header):
        samples = samples.to(device)
        targets = [{k: (v.to(device) if k != 'id' else v) for k, v in t.items()} for t in targets]

        # # register hooks to obtain intermediate outputs
        # dec_selfattn_weights, dec_crossattn_weights = [], []
        # if 'HOTR' in type(model).__name__:
        #     hook_self = model.interaction_transformer.decoder.layers[-1].self_attn.register_forward_hook(lambda self, input, output: dec_selfattn_weights.append(output[1]))
        #     hook_cross = model.interaction_transformer.decoder.layers[-1].multihead_attn.register_forward_hook(lambda self, input, output: dec_crossattn_weights.append(output[1]))
        # else:
        #     hook_self = model.interaction_decoder.layers[-1].self_attn.register_forward_hook(lambda self, input, output: dec_selfattn_weights.append(output[1]))
        #     hook_cross = model.interaction_decoder.layers[-1].multihead_attn.register_forward_hook(lambda self, input, output: dec_crossattn_weights.append(output[1]))

        outputs = model(recon_model, resnet, centroids, samples, targets)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](outputs, orig_target_sizes, threshold=thr, dataset='hico-det')
        hoi_recognition_time.append(results[0]['hoi_recognition_time'] * 1000)

        # # visualize
        # if targets[0]['id'] in [57]: # [47, 57, 81, 30, 46, 97]: # 30, 46, 97
        #     # check_annotation(samples, targets, mode='eval', rel_num=20)
        #
        #     # visualize cross-attentioa
        #     if 'HOTR' in type(model).__name__:
        #         outputs['pred_actions'] = outputs['pred_actions'][:, :, :args.num_actions]
        #         outputs['pred_rel_pairs'] = [x.cpu() for x in torch.stack([outputs['pred_hidx'].argmax(-1), outputs['pred_oidx'].argmax(-1)], dim=-1)]
        #     topk_qids, q_name_list = plot_hoi_results(samples, outputs, targets, args=args)
        #     plot_cross_attention(samples, outputs, targets, dec_crossattn_weights, topk_qids=topk_qids)
        #     print(f"image_id={targets[0]['id']}")
        #
        #     # visualize self attention
        #     print('visualize self-attention')
        #     q_num = len(dec_selfattn_weights[0][0])
        #     plt.figure(figsize=(10,4))
        #     plt.imshow(dec_selfattn_weights[0][0].cpu().numpy(), vmin=0, vmax=0.4)
        #     plt.xticks(np.arange(q_num), [f"{i}" for i in range(q_num)], rotation=90, fontsize=12)
        #     plt.yticks(np.arange(q_num), [f"({q_name_list[i]})={i}" for i in range(q_num)], fontsize=12)
        #     plt.gca().xaxis.set_ticks_position('top')
        #     plt.grid(alpha=0.4, linestyle=':')
        #     plt.show()
        # hook_self.remove(); hook_cross.remove()

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        # For avoiding a runtime error, the copy is used
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))


    print(f"[stats] HOI Recognition Time (avg) : {sum(hoi_recognition_time)/len(hoi_recognition_time):.4f} ms")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]
    
    evaluator = HICOEvaluator(preds, gts, data_loader.dataset.rare_triplets,
                                  data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat)

    stats = evaluator.evaluate()

    return stats

