import os


def punctuation_standardization(string: str):
    punctuation_dict = {
        '\u201c': "\"",
        '\u201d': "\"",
        '\u2019': "'",
        '\u2018': "'",
        '\u2013': '-'
    }
    for key, value in punctuation_dict.items():
        string = string.replace(key, value)
    return string


def gigaword_detokenize(string, is_target=False):
    _tok_dict = {
        '(': '-lrb-',
        ')': '-rrb-',
        '[': '-lsb-',
        ']': '-rsb-',
        '{': '-lcb-',
        '}': '-rcb-',
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;'
    }
    string = string.replace('UNK', '[UNK]')
    string = string.replace('<unk>', '[UNK]')
    for key, value in _tok_dict.items():
        string = string.replace(value, key)
    # string = string.replace("''", "\"")
    # string = string.replace("``", "\"")
    # string = string.replace("`", "'")
    # string = string.replace(" n't", "n't")
    # string = string.replace(" 's", "'s")
    # string = string.replace(" 'd", "'d")
    # string = string.replace(" 'll", "'ll")
    return string


def cnndm_detokenize(string, is_target=False):
    _tok_dict = {
        '(': '-LRB-',
        ')': '-RRB-',
        '[': '-LSB-',
        ']': '-RSB-',
        '{': '-LCB-',
        '}': '-RCB-'
    }
    if not is_target:
        string = string.replace('<S_SEP>', '')
    else:
        string = string.replace('<S_SEP>', '[SEP]')
    for key, value in _tok_dict.items():
        string = string.replace(value, key)
    string = string.replace("''", "\"")
    string = string.replace('``', "\"")
    string = string.replace('`', "'")
    string = string.replace(" n't", "n't")
    string = string.replace(" 's", "'s")
    string = string.replace(" 'd", "'d")
    string = string.replace(" 'll", "'ll")
    return string


def blanklm_detokenize(string, is_target=False):
    string = string.replace('_UNK', '[UNK]')
    string = string.replace('<blank>', '[MASK]')
    return string


class SummmaryProcessor:
    def __init__(self, task, data_dir, tokenizer):
        self.task = task
        self.data_dir = data_dir
        self.tokenizer = tokenizer

    def create_examples(self, split):
        if split == 'train':
            filename = 'train'
        elif split == 'dev':
            filename = 'val'
        elif split == 'test':
            filename = 'test'
        else:
            raise NotImplementedError(split)
        if self.task == 'gigaword':
            detokenizer = gigaword_detokenize
        elif self.task == 'cnn_dm':
            detokenizer = cnndm_detokenize
        else:
            detokenizer = None
        source_texts, target_texts = [], []
        with open(os.path.join(self.data_dir, f'{filename}.source'),
                  encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                line = punctuation_standardization(line)
                line = detokenizer(line) if detokenizer else line
                source_texts.append(line)
        with open(os.path.join(self.data_dir, f'{filename}.target'),
                  encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                line = punctuation_standardization(line)
                line = detokenizer(line,
                                   is_target=True) if detokenizer else line
                target_texts.append(line)
        assert len(source_texts) == len(target_texts)
        example_list = []
        for idx, (source_text,
                  target_text) in enumerate(zip(source_texts, target_texts)):
            guid = '%s-%s' % (split, idx)
            example = {
                'guid': guid,
                'text_a': source_text,
                'text_b': target_text
            }
            example_list.append(example)
        return example_list
