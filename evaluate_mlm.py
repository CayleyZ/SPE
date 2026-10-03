"""Evaluate masked-LM perplexity or one-token-at-a-time pseudo-perplexity."""

import argparse
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from spikingjelly.activation_based import functional
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from checkpoint_utils import load_mlm


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--mode', choices=['pseudo', 'masked'], default='pseudo')
    parser.add_argument('--dataset_name', default='nyu-mll/glue')
    parser.add_argument('--dataset_config_name', default='sst2')
    parser.add_argument('--dataset_path', help='Raw or tokenized dataset saved with save_to_disk().')
    parser.add_argument('--text_file', help='One sentence per line; overrides dataset options.')
    parser.add_argument('--text_column', default='sentence')
    parser.add_argument('--split', default='validation')
    parser.add_argument('--cache_dir')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--mlm_probability', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--neuron_backend', choices=['cupy', 'torch'], default='cupy')
    parser.add_argument('--output', default='results/mlm_validation.json')
    return parser.parse_args()


def load_examples(args, tokenizer):
    if args.text_file:
        lines = [line.strip() for line in Path(args.text_file).read_text(encoding='utf-8').splitlines() if line.strip()]
        rows = [{args.text_column: line} for line in lines]
    elif args.dataset_path:
        dataset = load_from_disk(args.dataset_path)
        rows = dataset[args.split] if hasattr(dataset, 'keys') else dataset
    else:
        rows = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir)[args.split]
    indices = range(args.start_index, min(len(rows), args.start_index + args.num_samples))
    encoded = []
    for index in indices:
        row = rows[index]
        if 'input_ids' in row:
            item = {key: list(row[key])[:args.max_length] for key in ['input_ids', 'attention_mask', 'token_type_ids'] if key in row}
            if 'attention_mask' not in item:
                item['attention_mask'] = [1] * len(item['input_ids'])
            item = tokenizer.pad(item, padding='max_length', max_length=args.max_length)
            item['special_tokens_mask'] = tokenizer.get_special_tokens_mask(item['input_ids'], already_has_special_tokens=True)
        else:
            column = args.text_column if args.text_column in row else 'text'
            item = tokenizer(row[column], padding='max_length', truncation=True, max_length=args.max_length, return_special_tokens_mask=True)
        encoded.append(dict(item))
    if not encoded:
        raise ValueError('No evaluation examples selected.')
    return encoded


@torch.inference_mode()
def evaluate_pseudo(model, examples, tokenizer, device):
    results = []
    special = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.mask_token_id}
    total_nll, total_tokens, total_hits = 0.0, 0, 0
    for index, example in enumerate(examples):
        ids = torch.tensor(example['input_ids'], device=device)
        attention = torch.tensor(example['attention_mask'], device=device)
        types = torch.tensor(example.get('token_type_ids', [0] * len(ids)), device=device)
        positions = [i for i, (token, valid) in enumerate(zip(ids.tolist(), attention.tolist())) if valid and token not in special]
        nll, hits = 0.0, 0
        for position in positions:
            masked = ids.clone()
            masked[position] = tokenizer.mask_token_id
            try:
                logits = model(input_ids=masked.unsqueeze(0), attention_mask=attention.unsqueeze(0), token_type_ids=types.unsqueeze(0)).logits[0, position]
                loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), ids[position].reshape(1))
                nll += float(loss)
                hits += int(logits.argmax() == ids[position])
            finally:
                functional.reset_net(model)
        if not positions:
            continue
        total_nll += nll
        total_tokens += len(positions)
        total_hits += hits
        sample = {'sample_index': index, 'n_tokens': len(positions), 'mean_nll': nll / len(positions),
                  'pseudo_ppl': math.exp(nll / len(positions)), 'top1_acc': hits / len(positions)}
        results.append(sample)
        print(json.dumps(sample), flush=True)
    if not total_tokens:
        raise ValueError('No non-special tokens to evaluate.')
    return {'n_tokens': total_tokens, 'mean_nll': total_nll / total_tokens,
            'pseudo_ppl': math.exp(total_nll / total_tokens), 'top1_acc': total_hits / total_tokens, 'samples': results}


@torch.inference_mode()
def evaluate_masked(model, examples, tokenizer, device, args):
    collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=args.mlm_probability, seed=args.seed)
    loader = DataLoader(examples, batch_size=args.batch_size, collate_fn=collator)
    nll, count, hits = 0.0, 0, 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        try:
            logits = model(**batch).logits
            labels = batch['labels']
            valid = labels != -100
            nll += float(torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels, reduction='sum'))
            count += int(valid.sum())
            hits += int(((logits.argmax(-1) == labels) & valid).sum())
        finally:
            functional.reset_net(model)
    if count == 0:
        raise ValueError('No masked tokens were sampled; increase num_samples.')
    return {'masked_tokens': count, 'mean_nll': nll / count, 'masked_ppl': math.exp(nll / count), 'top1_acc': hits / count}


def main():
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError('num_samples must be positive.')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. For CPU use --device cpu --neuron_backend torch.')
    if device.type == 'cpu' and args.neuron_backend == 'cupy':
        raise ValueError('The cupy backend requires CUDA; use --neuron_backend torch on CPU.')
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model, report = load_mlm(args.model_name_or_path, args.neuron_backend, args.max_length)
    model.to(device).eval()
    print('Checkpoint load:', json.dumps(report), flush=True)
    examples = load_examples(args, tokenizer)
    result = evaluate_pseudo(model, examples, tokenizer, device) if args.mode == 'pseudo' else evaluate_masked(model, examples, tokenizer, device, args)
    if not math.isfinite(result['mean_nll']):
        raise FloatingPointError('Non-finite evaluation loss.')
    payload = {'args': vars(args), 'load_report': report, 'metrics': result}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'samples'}, indent=2))


if __name__ == '__main__':
    main()
