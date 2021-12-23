#!/bin/env python
import os
import shutil
import logging
import torch
import wandb
import nltk

from train import Trainer, parse_args, setup_logging  # noqa:E402
from generate import generate_predictions  # noqa:E402
from data.evaluation.multiwoz import MultiWozEvaluator, compute_bmr_remove_reference, compute_delexicalized_bmr  # noqa: E402
from data import load_dataset  # noqa:E402
# import data.evaluation.convlab  # noqa:E402
# from evaluation_utils import compute_delexicalized_bleu  # noqa:E402
from data.utils import wrap_dataset_with_cache  # noqa:E402

rank_to_device = [0, 1, 2, 3]


class MultiWozTrainer(Trainer):
    @torch.no_grad()
    def run_convlab_evaluation(self):
        self.logger.info('running convlab evaluation')
        self.model.eval()
        analyzer = data.evaluation.multiwoz.convlab.ConvLabAnalyzer()
        result = analyzer(self.prediction_pipeline, num_dialogs=self.args.evaluation_dialogs)

        # Move the results from evaluator script to wandb
        shutil.move('results', os.path.join(wandb.run.dir, 'results'))

        # Fix synchronize metrics when using different run for other metrics
        online_run = wandb.Api().run(self.wandb_runid)
        evaluation_keys = {'test_inform', 'test_success', 'test_bleu', 'test_delex_bleu'}
        summary = {k: v for k, v in online_run.summary.items() if k in evaluation_keys}
        wandb.run.summary.update(summary)

        # Update results from the analyzer
        wandb.run.summary.update(result)

    @torch.no_grad()
    def run_test_evaluation(self, epoch):
        self.logger.info('running multiwoz evaluation for epoch ' + str(epoch))
        # self.logger.info('generating responses')
        self.model.eval()
        dataset = load_dataset('multiwoz-2.1-test', use_goal=True)
        dataset = wrap_dataset_with_cache(dataset)
        beliefs, databases, documents, responses, gold_responses, delex_responses, delex_gold_responses = \
            generate_predictions(self.prediction_pipeline, dataset, 'test-predictions.txt')
        evaluator = MultiWozEvaluator(dataset, is_multiwoz_eval=True, logger=self.logger)
        self.logger.info('computing metrics')
        joint_goal, db_correct, mmr5, r1, r5, matches, success, domain_results, dp, dr, df = \
            evaluator.evaluate(beliefs, databases, documents, delex_responses, progressbar=True)
        bleu, meteor, rouge = compute_bmr_remove_reference(responses, gold_responses)
        delex_bleu, delex_meteor, delex_rouge = compute_delexicalized_bmr(delex_responses, delex_gold_responses)
        self.logger.info('evaluation finished for epoch ' + str(epoch))
        self.logger.info(f'joint goal: {joint_goal:.4f}, db acc: {db_correct:.4f}')
        self.logger.info(f'detect_p: {dp:.4f}, detect_r: {dr:.4f}, detect_f: {df:.4f}')
        self.logger.info(f'MRR@5: {mmr5:.4f}, R@1: {r1:.4f}, R@5: {r5:.4f}')
        self.logger.info(f'inform: {matches:.4f}, success: {success:.4f}')
        self.logger.info(f'bleu: {bleu:.4f}, meteor: {meteor:.4f}, rouge: {rouge:.4f}')
        self.logger.info(f'delex_bleu: {delex_bleu:.4f}, delex_meteor: {delex_meteor:.4f}, delex_rouge: {delex_rouge:.4f}')

        # We will use external run to run in a separate process
        run = wandb.run
        shutil.copy('test-predictions.txt', run.dir)
        run.summary.update(dict(
            test_joint_goal=joint_goal,
            test_db_acc=db_correct,
            test_mmr5=mmr5,
            test_r1=r1,
            test_r5=r5,
            test_inform=matches,
            test_success=success,
            test_bleu=bleu,
            test_delex_bleu=delex_bleu,
            test_meteor=meteor,
            test_delex_meteor=delex_meteor,
            test_rouge=rouge,
            test_delex_rouge=delex_rouge
        ))

    def run_evaluation(self, epoch):
        if self.is_master():
            # self.run_convlab_evaluation()
            self.run_test_evaluation(epoch)
        # if self.args.local_rank == -1 or torch.distributed.get_rank() == 1 or torch.distributed.get_world_size() == 1:
            # self.run_test_evaluation()
        torch.distributed.barrier()

        '''
        if self.is_master() and self.args.local_rank != -1 and torch.distributed.get_world_size() > 1:
            # Fix synchronize metrics when using different run for other metrics
            online_run = wandb.Api().run(self.wandb_runid)
            evaluation_keys = {'test_inform', 'test_success', 'test_bleu', 'test_delex_bleu'}
            summary = {k: v for k, v in online_run.summary.items() if k in evaluation_keys}
            wandb.run.summary.update(summary)
        '''


if __name__ == '__main__':
    # Update punkt
    # nltk.download('punkt')

    args = parse_args()
    setup_logging()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if args.local_rank != -1:
        assert args.device == torch.device('cuda'), "CUDA must be available in distributed training"
        device_idx = (args.local_rank + args.device_shift) % len(rank_to_device)
        torch.cuda.set_device(rank_to_device[device_idx])
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        logger.info('initialized distributed training with {} nodes, local-rank: {}'.format(
            torch.distributed.get_world_size(), args.local_rank))

    # Start training
    trainer = MultiWozTrainer(args, logger)
    trainer.train()
    if args.local_rank != -1:
        torch.distributed.destroy_process_group()
