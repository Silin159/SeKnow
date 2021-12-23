#!/bin/env python
import os
import argparse
import logging
import torch
import transformers
import nltk
from collections import OrderedDict

from utils import setup_logging, pull_model  # noqa:E402
from data.utils import BeliefParser, DatabaseParser, wrap_dataset_with_cache  # noqa: E402
from data import load_dataset  # noqa: E402
from data.evaluation.multiwoz import MultiWozEvaluator, compute_bmr_remove_reference, compute_delexicalized_bmr  # noqa: E402
from generate import generate_predictions  # noqa:E402
# from evaluation_utils import compute_delexicalized_bleu  # noqa:E402


def parse_predictions(dataset, filename):
    gts, bfs, ds, rs = [], [], [], []
    delexrs, delexgts = [], []
    bf_parser = BeliefParser()
    d_paser = DatabaseParser()
    with open(filename, 'r') as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('GT:'):
                gts.append(line[len('GT:'):])
            elif line.startswith('GTD:'):
                delexgts.append(line[len('GTD:'):])
            elif line.startswith('BF:'):
                bf = line[len('BF:'):]
                bf = bf_parser(bf)
                assert bf is not None
                bfs.append(bf)
            elif line.startswith('RD:'):
                delexrs.append(line[len('RD:'):])
            elif line.startswith('R:'):
                r = line[len('R:'):]
                rs.append(r)
            elif line.startswith('D:'):
                d = line[len('D:'):]
                d = d_paser(d)
                assert d is not None
                ds.append(d)
    # assert len(gts) == len(bfs) == len(rs) == len(delexrs) == len(delexgts)
    return rs, bfs, ds, gts, delexrs, delexgts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None)
    parser.add_argument('--file', default=None)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--dataset', default='multiwoz-2.1-test')
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    if args.resume is not None and args.model is None:
        args.model = f'wandb:{args.resume}'
    assert args.model is not None or args.file is not None

    # Update punkt
    # nltk.download('punkt')

    setup_logging()
    logger = logging.getLogger()
    if args.resume:
        import wandb
        # Resume run and fill metrics
        os.environ.pop('WANDB_NAME', None)
        wandb.init(resume=args.resume)
    elif args.wandb:
        import wandb
        # It is an artifact
        # Start a new evaluate run
        wandb.init(job_type='evaluation')
    else:
        wandb = None

    dataset = load_dataset(args.dataset, use_goal=True)
    dataset = wrap_dataset_with_cache(dataset)

    if args.file is None or not os.path.exists(args.file):
        args.model = pull_model(args.model)

    if args.file is not None:
        path = args.file
        if not os.path.exists(path):
            path = os.path.join(args.model, args.file)
        responses, beliefs, databases, gold_responses, delex_responses, delex_gold_responses = \
            parse_predictions(dataset, path)
    else:
        logger.info('generating responses')
        pipeline = transformers.pipeline('augpt-conversational', args.model, device=0 if torch.cuda.is_available() else -1)
        beliefs, databases, documents, responses, gold_responses, delex_responses, delex_gold_responses = \
            generate_predictions(pipeline, dataset, os.path.join(wandb.run.dir if wandb and wandb.run else '.', 'test-predictions.txt'))
    logger.info('evaluation started')
    evaluator = MultiWozEvaluator(dataset, is_multiwoz_eval=True, logger=logger)
    joint_goal, db_correct, matches, success, domain_results = \
        evaluator.evaluate(beliefs, databases, delex_responses, progressbar=True)
    logger.info('evaluation finished')
    logger.info(f'joint goal: {joint_goal:.4f}, db acc: {db_correct:.4f}')
    logger.info(f'match: {matches:.4f}, success: {success:.4f}')
    if wandb and wandb.run:
        wandb.run.summary.update(dict(
            test_joint_goal=joint_goal,
            test_db_acc=db_correct,
        ))
        wandb.run.summary.update(dict(
            test_inform=matches,
            test_success=success,
        ))
    logger.info('computing language quality metrics:')
    if dataset.lexicalizer is not None:
        bleu, meteor, rouge = compute_bmr_remove_reference(responses, gold_responses)
        logger.info(f'test bleu: {bleu:.4f}, test meteor: {meteor:.4f}, test rouge: {rouge:.4f}')
        if wandb and wandb.run:
            wandb.run.summary.update(dict(test_bleu=bleu, test_meteor=meteor, test_rouge=rouge))

    delex_bleu, delex_meteor, delex_rouge = compute_delexicalized_bmr(delex_responses, delex_gold_responses)
    logger.info(f'test delex bleu: {delex_bleu:.4f}, test delex meteor: {delex_meteor:.4f}, test delex rouge: {delex_rouge:.4f}')
    if wandb and wandb.run:
        wandb.run.summary.update(dict(test_delex_bleu=delex_bleu, test_delex_meteor=delex_meteor, test_delex_rouge=delex_rouge))
