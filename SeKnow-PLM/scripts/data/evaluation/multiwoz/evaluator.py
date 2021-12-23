import re
from functools import partial
from tqdm import tqdm
from collections import defaultdict, Counter
from nltk.util import ngrams
import math
from data.loader import load_dataset
from data.utils import BeliefParser, DatabaseParser, format_belief, format_database
import nltk
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from fuzzywuzzy import fuzz


class BLEUScorer(object):
    def __init__(self):
        pass

    def score(self, hyps_ls, refs_ls):
        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]

        # accumulate ngram statistics
        for (hyps, refs) in zip(hyps_ls, refs_ls):
            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]
            for hyp in hyps:
                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0: break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)

        # computing bleu score
        p0 = 1e-10
        bp = 1 if c > r else math.exp(1 - float(r) / (float(c) + p0))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 for i in range(4)]
        s = math.fsum(w * math.log(p_n) for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu


class METEORScorer(object):
    def __init__(self):
        self.invalid = (None, [], "")
        pass

    def score(self, hyps, refs):
        sample = 1e-10
        score = 0.0
        for (hyp, ref) in zip(hyps, refs):
            if (hyp not in self.invalid) and (ref not in self.invalid):
                score += meteor_score(ref, hyp)
                sample += 1
        meteor = score/sample
        return meteor


class ROUGEScorer(object):
    def __init__(self):
        self.invalid = (None, [], "")
        pass

    def score(self, hyps, refs):
        sample = 1e-10
        score = 0.0
        rougef = Rouge()
        for hyp, ref in zip(hyps, refs):
            if (hyp not in self.invalid) and (ref not in self.invalid):
                score += rougef.get_scores(hyp, ref)[0]['rouge-l']['f']  # consider ROUGE-L
                sample += 1
        rouge = score/sample
        return rouge


class BaseEvaluator(object):
    def initialize(self):
        raise NotImplementedError

    def add_example(self, ref, hyp):
        raise NotImplementedError

    def get_report(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _get_prec_recall(tp, fp, fn):
        precision = tp / (tp + fp + 10e-20)
        recall = tp / (tp + fn + 10e-20)
        f1 = 2 * precision * recall / (precision + recall + 1e-20)
        return precision, recall, f1

    @staticmethod
    def _get_tp_fp_fn(label_list, pred_list):
        tp = len([t for t in pred_list if t in label_list])
        fp = max(0, len(pred_list) - tp)
        fn = max(0, len(label_list) - tp)
        return tp, fp, fn


def get_logger():
    import logging  # noqa:F811
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger


class MultiWozEvaluator(BaseEvaluator):
    def __init__(self, dataset, is_multiwoz_eval=False, logger=None):
        self.db = dataset.database
        self.doc_base = dataset.docbase
        self.dataset = dataset
        self.labels = list()
        self.hyps = list()
        self.belief_parser = BeliefParser()
        self.database_parser = DatabaseParser()
        self.is_multiwoz_eval = is_multiwoz_eval
        self.logger = logger or get_logger()
        self.label_regex = re.compile(r'\[([\w\d\s]+)\]')

    def _query_original_db(self, domain, belief):
        belief = {domain: belief}
        return self.db(belief, return_results=True)[domain][1]

    def _get_requestables_and_venues(self, items, beliefs, responses, dialog_booked_domains):
        # for computing corpus success
        requestables = {'phone', 'address', 'postcode', 'reference', 'id'}
        provided_requestables = defaultdict(lambda: set())
        venue_offered = defaultdict(lambda: [])
        for i, (item, belief, response, booked_domains) in enumerate(zip(items, beliefs, responses, dialog_booked_domains)):
            if item.uk_based:
                continue
            database_results = self.db(belief, return_results=True)
            current_requestables = set(self.label_regex.findall(response))
            self.logger.debug(response)
            current_domain = next(iter(belief.keys())) if belief else None
            self.logger.debug(f'domain: {current_domain}, r: {current_requestables}')
            self.logger.debug(f"belief: {belief.get('hotel', None)}")
            self.logger.debug(f"db: {database_results.get('hotel', None)}")

            # Parse multiwoz style requestables first
            legacy_requestables = {x for x in current_requestables if '_' in x}
            current_requestables.difference_update(legacy_requestables)
            for requestable_candidate in legacy_requestables:
                domain, slot = requestable_candidate.split('_')
                if slot not in requestables:
                    continue

                # https://github.com/budzianowski/multiwoz/blob/a24d299fafa00371d03880bce34cb3b0923518fa/evaluate.py#L248
                # if db pointer was allowing for that
                if slot == 'reference' and domain in {'restaurant', 'hotel', 'train'}:
                    if domain not in booked_domains:
                        continue

                provided_requestables[domain].add(slot)

            # New style delexicalization
            for domain, (num_results, results) in database_results.items():
                if not self.is_multiwoz_eval:
                    current_delex_requestables = set(belief.get(domain, dict()).keys())
                    if num_results > 0:
                        current_delex_requestables.update(results[0].keys())

                    matched_requestables = current_requestables.intersection(current_delex_requestables)
                    if 'reference' in matched_requestables and domain in {'restaurant', 'hotel', 'train'}:
                        # https://github.com/budzianowski/multiwoz/blob/a24d299fafa00371d03880bce34cb3b0923518fa/evaluate.py#L248
                        # if db pointer was allowing for that
                        if domain not in booked_domains:
                            matched_requestables.remove('reference')

                    current_requestables -= matched_requestables
                    provided_requestables[domain].update(matched_requestables.intersection(requestables))

                # Venues offered
                if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                    venues = results
                    if len(venue_offered[domain]) == 0 and venues:
                        venue_offered[domain] = venues
                    else:
                        flag = False
                        for ven in venues:
                            if venue_offered[domain][0].get('id') == ven.get('id'):
                                flag = True
                                break
                        if not flag and venues:  # sometimes there are no results so sample won't work
                            venue_offered[domain] = venues
                else:
                    venue_offered[domain] = domain + '_name'

            # These slots are not lexicalised back, but its is not a concern
            # in multiwoz evaluation which does not take it into account
            if current_domain and self.is_multiwoz_eval:
                provided_requestables[current_domain].update(current_requestables)
        return provided_requestables, venue_offered

    def _evaluate_generated_dialogue(self, real_requestables, provided_requestables,
                                     venue_offered, goal, stats):
        # if name was given in the task
        for domain in goal.keys():
            # if name was provided for the user, the match is being done automatically
            # if realDialogue['goal'][domain].has_key('info'):
            if 'informable' in goal[domain]:
                # if realDialogue['goal'][domain]['info'].has_key('name'):
                if 'name' in goal[domain]['informable']:
                    venue_offered[domain] = domain + '_name'

            # special domains - entity does not need to be provided
            if domain in ['taxi', 'police', 'hospital']:
                venue_offered[domain] = domain + '_name'

            # if id was not requested but train was found we dont want
            # to override it to check if we booked the right train
            if domain == 'train' and (not venue_offered[domain] and 'id' not in goal['train']['requestable']):
                venue_offered[domain] = domain + '_name'

        """
        Given all inform and requestable slots
        we go through each domain from the user goal
        and check whether right entity was provided and
        all requestable slots were given to the user.
        The dialogue is successful if that's the case for all domains.
        """
        stat_domain_total, stat_domain_match, stat_domain_success = stats

        # MATCH
        match = 0.0
        for domain in goal.keys():
            domain_success = False
            if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                goal_venues = self._query_original_db(domain, goal[domain]['informable'])

                if isinstance(venue_offered[domain], str):
                    if '_name' in venue_offered[domain]:
                        domain_success = True
                elif len(venue_offered[domain]) > 0:
                    reference = venue_offered[domain][0]['id']
                    if any(isinstance(x, dict) and x.get('id') == reference for x in goal_venues):
                        domain_success = True
            else:
                if domain + '_name' in venue_offered[domain]:
                    domain_success = True
            match += domain_success
            stat_domain_total[domain] += 1
            stat_domain_match[domain] += domain_success

        if match == len(goal.keys()):
            match = 1.0
        else:
            match = 0.0

        # SUCCESS
        success = 0.0
        if match == 1.0:
            for domain in goal.keys():
                # if values in sentences are super set of requestables
                prov_req = provided_requestables[domain].intersection(real_requestables[domain])
                domain_success = len(prov_req) == len(real_requestables[domain])
                # if not domain_success:
                #    # print('HUPS', domain, provided_requestables[domain], real_requestables[domain])
                success += domain_success
                stat_domain_success[domain] += domain_success

            if success >= len(real_requestables):
                success = 1.0
            else:
                success = 0.0

        if success == 0:
            # print((real_requestables, provided_requestables))
            pass
        return success, match

    def _get_goal_and_requestables(self, gt_belief, goal):
        # compute corpus success
        real_requestables = {}
        provided_requestables = {}
        venue_offered = {}
        for domain in goal.keys():
            provided_requestables[domain] = []
            venue_offered[domain] = []
            real_requestables[domain] = goal[domain]['requestable']
        return goal, real_requestables

    def _parse_entities(self, tokens):
        entities = []
        for t in tokens:
            if '[' in t and ']' in t:
                entities.append(t)
        return entities

    def pack_dialogues(self, dataset, beliefs, databases, documents, responses):
        def batch_dialog(dialog):
            (items, goals, beliefs, databases, documents, responses, booked_domains) = tuple(zip(*dialog))
            return items, goals[0], beliefs, databases, documents, responses, booked_domains

        if isinstance(dataset, str):
            dataset = load_dataset(dataset, goal=True)
        current_dialogue = []
        for item, belief, database, document, response in zip(dataset, beliefs, databases, documents, responses):
            if len(item.context) == 1:
                if current_dialogue:
                    yield batch_dialog(current_dialogue)
                current_dialogue = []
            # print(item, item.goal)
            current_dialogue.append((item, item.goal, belief, database, document, response, item.booked_domains))
            # current_dialogue.append((item, item.goal, item.raw_belief, item.response))
        yield batch_dialog(current_dialogue)

    def reciprocal_rank(self, k_ref, k_pre, pre_num=5):
        match = []
        for idx in range(pre_num):
            match.append(fuzz.ratio(k_ref, k_pre[idx]) >= 95)
        if True in match:
            rank = match.index(True)
            return 1.0/(rank+1)
        else:
            return 0.0

    def recall_at_k(self, k_ref, k_pre, topk=5):
        match = []
        for idx in range(topk):
            match.append(fuzz.ratio(k_ref, k_pre[idx]) >= 95)
        if True in match:
            return 1.0
        else:
            return 0.0

    def compute_prf(self, score_sum, tp, fp, fn):
        if tp + fp > 0.0:
            score_p = score_sum/(tp + fp)
        else:
            score_p = 0.0

        if tp + fn > 0.0:
            score_r = score_sum/(tp + fn)
        else:
            score_r = 0.0

        if score_p + score_r > 0.0:
            score_f = 2*score_p*score_r/(score_p+score_r)
        else:
            score_f = 0.0

        return score_p, score_r, score_f

    def evaluate(self, beliefs, databases, documents, responses, progressbar=False):
        dialogues = self.pack_dialogues(self.dataset, beliefs, databases, documents, responses)
        successes, matches = 0.0, 0.0
        jg_acc, db_acc = 0.0, 0.0
        stats = tuple(Counter() for _ in range(3))
        domain_total, domain_match, domain_success = stats
        total = 0.0
        bf_total = 0.0
        detect_tp, detect_fp, detect_tn, detect_fn = 0.0, 0.0, 0.0, 0.0
        sum_mrr5, sum_r1, sum_r5 = 0.0, 0.0, 0.0

        offset = 0
        progress = tqdm(total=len(self.dataset),
                        desc=progressbar if isinstance(progressbar, str) else 'evaluating',
                        disable=not progressbar)
        for idx, (items, goal, beliefs, databases, documents, responses, booked_domains) in enumerate(dialogues):

            for turn_idx, (item, belief, database, document) in enumerate(zip(items, beliefs, databases, documents)):
                uk_based_pre = "ruk" in belief.get(item.active_domain, {}).keys()
                gold_doc = item.document
                '''debug
                if idx < 20 and item.uk_based:
                    print(item.active_domain)
                    print(belief)
                    print(database)
                    print(item.document)
                    print(document[0])
                '''
                if item.uk_based:
                    if uk_based_pre:
                        detect_tp += 1
                        sum_mrr5 += self.reciprocal_rank(gold_doc, document, 5)
                        sum_r1 += self.recall_at_k(gold_doc, document, 1)
                        sum_r5 += self.recall_at_k(gold_doc, document, 5)
                    else:
                        detect_fn += 1
                else:
                    if uk_based_pre:
                        detect_fp += 1
                    else:
                        detect_tn += 1
                    user_utter = item.context[-1]
                    gold_belief_str = format_belief(item.raw_belief)
                    gold_database_str = format_database(item.database)
                    if ('thank' not in user_utter) and ('bye' not in user_utter) and (gold_belief_str is not ""):
                        bf_total += 1
                        if format_belief(belief) == gold_belief_str:
                            jg_acc += 1
                        if format_database(database) == gold_database_str:
                            db_acc += 1

            goal, real_requestables = self._get_goal_and_requestables(items[-1].raw_belief, goal)
            self.logger.debug(f'rr: {real_requestables}, g: {goal}')
            provided_requestables, venue_offered = self._get_requestables_and_venues(items, beliefs, responses, booked_domains)
            success, match = self._evaluate_generated_dialogue(
                real_requestables, provided_requestables, venue_offered, goal, stats)
            successes += success
            matches += match
            total += 1
            offset += len(items)
            progress.update(len(items))

        jg_acc = jg_acc / float(bf_total)
        db_acc = db_acc / float(bf_total)
        mrr5_p, mrr5_r, mrr5_f = self.compute_prf(sum_mrr5, detect_tp, detect_fp, detect_fn)
        r5_p, r5_r, r5_f = self.compute_prf(sum_r5, detect_tp, detect_fp, detect_fn)
        r1_p, r1_r, r1_f = self.compute_prf(sum_r1, detect_tp, detect_fp, detect_fn)
        dp, dr, df = self.compute_prf(detect_tp, detect_tp, detect_fp, detect_fn)
        matches = matches / float(total)
        successes = successes / float(total)
        domain_results = dict()
        for key in domain_total.keys():
            domain_results[key] = domain_match[key] / \
                float(domain_total[key]), domain_success[key] / float(domain_total[key])

        return jg_acc, db_acc, mrr5_f, r1_f, r5_f, matches, successes, domain_results, dp, dr, df


def compute_bmr_remove_reference(responses, gold_responses):
    responses = list(map(lambda x: x.lower(), responses))
    gold_responses = list(map(lambda x: x.lower(), gold_responses))

    reference_regex = re.compile(
        r'(?:^|[^a-zA-Z0-9])(?=[A-Z0-9]{8}(?:[^a-zA-Z0-9]|$))([A-Z0-9]*[A-Z][A-Z0-9]*|0{4}\d{4})')
    reference_sub = partial(reference_regex.sub, lambda x: x.group(0).replace(x.group(1), 'REFERENCE'))
    responses = list(map(reference_sub, responses))
    gold_responses = list(map(reference_sub, gold_responses))

    responses = list(map(nltk.tokenize.word_tokenize, responses))
    gold_responses = list(map(nltk.tokenize.word_tokenize, gold_responses))
    responses = list(map(lambda x: " ".join(x), responses))
    gold_responses = list(map(lambda x: " ".join(x), gold_responses))
    warp_responses = list(map(lambda x: [x], responses))
    wrap_gold_responses = list(map(lambda x: [x], gold_responses))

    meteor = METEORScorer().score(responses, wrap_gold_responses)
    rouge = ROUGEScorer().score(warp_responses, wrap_gold_responses)
    bleu = BLEUScorer().score(warp_responses, wrap_gold_responses)
    return bleu, meteor, rouge


def compute_delexicalized_bmr(responses, gold_responses):
    responses = list(map(lambda x: x.lower(), responses))
    gold_responses = list(map(lambda x: x.lower(), gold_responses))

    token_regex = re.compile(r'\[([\w\s\d]+)\]')
    token_sub = partial(token_regex.sub, lambda x: x.group(1).upper().replace(' ', ''))
    responses = list(map(token_sub, responses))
    gold_responses = list(map(token_sub, gold_responses))

    responses = list(map(nltk.tokenize.word_tokenize, responses))
    gold_responses = list(map(nltk.tokenize.word_tokenize, gold_responses))
    responses = list(map(lambda x: " ".join(x), responses))
    gold_responses = list(map(lambda x: " ".join(x), gold_responses))
    warp_responses = list(map(lambda x: [x], responses))
    wrap_gold_responses = list(map(lambda x: [x], gold_responses))

    meteor = METEORScorer().score(responses, wrap_gold_responses)
    rouge = ROUGEScorer().score(warp_responses, wrap_gold_responses)
    bleu = BLEUScorer().score(warp_responses, wrap_gold_responses)
    return bleu, meteor, rouge
