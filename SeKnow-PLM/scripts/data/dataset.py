import bisect
import os
import json
from .utils import BlacklistItemsWrapper
from collections import OrderedDict
from utils import AutoDatabase, AutoLexicalizer, AutoDocbase
from .utils import DialogDataset, DialogDatasetItem, split_name

DATASETS_PATH = os.path.expanduser(os.environ.get('DATASETS_PATH', '~/datasets'))


def build_blacklist(items, domains=None):
    for i, (dialogue, items) in enumerate(items):
        if domains is not None and set(dialogue['domains']).difference(domains):  # unwanted domain
            yield i
        elif items[-1]['speaker'] != 'system':  # do not end with a system response
            yield i


def load_dataset(name, use_goal=False, context_window_size=None, domains=None, **kwargs) -> DialogDataset:
    name, split = split_name(name)
    path = os.path.join(DATASETS_PATH, name)
    with open(os.path.join(path, f'{split}.json'), 'r') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)
    dialogues = data['dialogues']
    items = DialogueItems(dialogues)  # important definition, for each iteration, return: whole dialog, turns to current
    items = BlacklistItemsWrapper(items, list(build_blacklist(items, domains)))  # skip bad iteration results

    def transform(x):
        dialogue, items = x  # every element in DialogueItems() contains two parts: whole dialog, turns to current
        context = [s['text'] for s in items[:-1]]  # dialog context
        if context_window_size is not None and context_window_size > 0:
            context = context[-context_window_size:]

        belief = items[-1]['belief']
        database = items[-1]['database']
        document = items[-1]['document']

        item = DialogDatasetItem(context, raw_belief=belief, database=database, document=document,
                                 response=items[-1]['delexicalised_text'], raw_response=items[-1]['text'])
        if use_goal:
            setattr(item, 'goal', dialogue['goal'])
            # MultiWOZ evaluation uses booked domains property
            if 'booked_domains' in items[-1]:
                setattr(item, 'booked_domains', items[-1]['booked_domains'])
            setattr(item, 'dialogue_act', items[-1]['dialogue_act'])
        setattr(item, 'active_domain', items[-1]['active_domain'])
        setattr(item, 'uk_based', items[-1]['uk_based'])
        return item

    dataset = DialogDataset(items, transform=transform, domains=data['domains'])
    if os.path.exists(os.path.join(path, 'database.zip')):
        dataset.database = AutoDatabase.load(path)

    if os.path.exists(os.path.join(path, 'lexicalizer.zip')):
        dataset.lexicalizer = AutoLexicalizer.load(path)

    if os.path.exists(os.path.join(path, 'document_base.json')):
        dataset.docbase = AutoDocbase.load(path)

    return dataset


class DialogueItems:
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            r.append(e + s)
            s += e
        return r

    def __init__(self, dialogues):
        lengths = [len(x['items']) for x in dialogues]
        self.cumulative_sizes = DialogueItems.cumsum(lengths)
        self.dialogues = dialogues

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dialogue_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dialogue_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dialogue_idx - 1]
        # whole dialog sample, turns to current
        return self.dialogues[dialogue_idx], self.dialogues[dialogue_idx]['items'][:sample_idx + 1]

    def __len__(self):
        if not self.cumulative_sizes:
            return 0
        return self.cumulative_sizes[-1]
