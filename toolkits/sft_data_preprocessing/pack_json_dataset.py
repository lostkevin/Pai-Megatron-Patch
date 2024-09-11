from transformers import AutoTokenizer
import json
import pdb
tokenizer = AutoTokenizer.from_pretrained("/mnt/qwen-ckpts/Qwen2-0.5B")
seqlen = 2048
example_len_map = dict()
with open("/mnt/qwen-datasets/qwen_sft.json") as f:
    for line in f:
        line = line.strip()
        example = json.loads(line)
        input = example["instruction"]+example['input']
        output = example['output']
        input_ids = tokenizer(input, add_special_tokens=False)['input_ids']
        output_ids = tokenizer(output, add_special_tokens=False)['input_ids']
        example_len_map[line] = len(input_ids) + len(output_ids) + 3

example_group = []
len_group = []
with open("/mnt/qwen-datasets/packed_qwen_sft.json", 'w', encoding='utf8') as f:
    for example, len in example_len_map.items():
        example = json.loads(example)
        if sum(len_group) <= seqlen:
            example_group.append(example)
            len_group.append(len)
        else:
            last_example = example_group[-1]
            last_example_len = len_group[-1]
            assert last_example_len <= seqlen
            json_string = json.dumps(example_group[:-1], ensure_ascii=False)
            f.write(json_string+"\n")
            example_group = [last_example]+[example]
            len_group = [last_example_len]+[len]

