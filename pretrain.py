"""MLM pre-training of the released SPE SpikingBERT architecture."""

import argparse
import json
import math
from itertools import chain
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from spikingjelly.activation_based import functional
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, get_scheduler

from checkpoint_utils import load_config
from spikingbert_pelif_simsig_stable import BertForMaskedLM


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument('--tokenized_dataset', help='DatasetDict saved with datasets.save_to_disk().')
    data.add_argument('--dataset_name', nargs='+', help='Raw dataset directories or Hugging Face dataset IDs.')
    data.add_argument('--train_file', help='UTF-8 text, JSON, or CSV file.')
    parser.add_argument('--dataset_config_name')
    parser.add_argument('--validation_file')
    parser.add_argument('--config_name', default='configs/spe_110m.json')
    parser.add_argument('--tokenizer_name', default='bert-base-uncased')
    parser.add_argument('--cache_dir')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--per_device_train_batch_size', type=int, default=64)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=64)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--max_train_steps', type=int, default=650000)
    parser.add_argument('--num_warmup_steps', type=int, default=5000)
    parser.add_argument('--mlm_probability', type=float, default=0.15)
    parser.add_argument('--checkpointing_steps', type=int, default=50000)
    parser.add_argument('--eval_steps', type=int, default=10000)
    parser.add_argument('--preprocessing_num_workers', type=int, default=1)
    parser.add_argument('--max_train_samples', type=int)
    parser.add_argument('--max_eval_samples', type=int)
    parser.add_argument('--resume_from_checkpoint', help='Accelerate state directory, e.g. step_50000.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--neuron_backend', choices=['cupy', 'torch'], default='cupy')
    parser.add_argument('--mixed_precision', choices=['no', 'fp16', 'bf16'], default='no')
    return parser.parse_args()


def prepare_data(args, tokenizer, accelerator):
    if args.tokenized_dataset:
        result = load_from_disk(args.tokenized_dataset)
        if not isinstance(result, DatasetDict) or not {'train', 'validation'} <= set(result):
            raise ValueError('Tokenized data must contain train and validation splits.')
        return result
    corpora = []
    if args.dataset_name:
        for name in args.dataset_name:
            if Path(name).is_dir():
                corpus = load_from_disk(name)
            else:
                corpus = load_dataset(name, args.dataset_config_name, cache_dir=args.cache_dir)
            if not isinstance(corpus, DatasetDict):
                corpus = DatasetDict(train=corpus)
            corpora.append(corpus)
    else:
        files = {'train': args.train_file}
        if args.validation_file:
            files['validation'] = args.validation_file
        extension = Path(args.train_file).suffix.lstrip('.')
        corpora = [load_dataset('text' if extension == 'txt' else extension, data_files=files, cache_dir=args.cache_dir)]
    train, validation = [], []
    for corpus in corpora:
        if 'validation' not in corpus:
            split = corpus['train'].train_test_split(test_size=min(1000, max(1, len(corpus['train']) // 100)), seed=args.seed)
            corpus = DatasetDict(train=split['train'], validation=split['test'])
        for split_name, destination in [('train', train), ('validation', validation)]:
            part = corpus[split_name]
            if 'text' not in part.column_names:
                raise ValueError('Raw corpora must have a text column.')
            destination.append(part.select_columns(['text']))
    raw = DatasetDict(train=concatenate_datasets(train), validation=concatenate_datasets(validation))

    def tokenize(examples):
        return tokenizer(examples['text'], return_special_tokens_mask=True)

    def group(examples):
        flat = {key: list(chain.from_iterable(values)) for key, values in examples.items()}
        total = len(flat['input_ids']) // args.max_seq_length * args.max_seq_length
        return {key: [values[i:i + args.max_seq_length] for i in range(0, total, args.max_seq_length)]
                for key, values in flat.items()}

    with accelerator.main_process_first():
        encoded = raw.map(tokenize, batched=True, num_proc=args.preprocessing_num_workers, remove_columns=['text'])
        encoded = encoded.map(group, batched=True, num_proc=args.preprocessing_num_workers)
    return encoded


@torch.no_grad()
def evaluate(model, loader, accelerator):
    model.eval()
    totals = torch.zeros(2, device=accelerator.device, dtype=torch.float64)
    for batch in loader:
        try:
            logits = model(**batch).logits
            labels = batch['labels']
            losses = torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels, reduction='none')
            per_sample = torch.stack((losses.sum(1), (labels != -100).sum(1)), dim=1).double()
            totals += accelerator.gather_for_metrics(per_sample).sum(0)
        finally:
            functional.reset_net(model)
    if totals[1] == 0:
        raise ValueError('The evaluation set contains no masked tokens.')
    nll = (totals[0] / totals[1]).item()
    return {'masked_nll': nll, 'masked_ppl': math.exp(nll), 'masked_tokens': int(totals[1].item())}


def main():
    args = parse_args()
    if args.max_train_steps <= 0 or args.max_seq_length <= 0:
        raise ValueError('Training steps and sequence length must be positive.')
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision)
    set_seed(args.seed)
    output = Path(args.output_dir)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'training_args.json').write_text(json.dumps(vars(args), indent=2))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    config = load_config(args.config_name, args.neuron_backend, args.max_seq_length)
    if args.max_seq_length > config.max_position_embeddings:
        raise ValueError('Increase max_position_embeddings in the config for longer pre-training sequences.')
    model = BertForMaskedLM(config)
    data = prepare_data(args, tokenizer, accelerator)
    train, validation = data['train'], data['validation']
    if args.max_train_samples:
        train = train.select(range(min(args.max_train_samples, len(train))))
    if args.max_eval_samples:
        validation = validation.select(range(min(args.max_eval_samples, len(validation))))
    if not len(train) or not len(validation):
        raise ValueError('Both tokenized splits must be nonempty.')
    train_collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=args.mlm_probability)
    eval_collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=args.mlm_probability, seed=args.seed)
    train_loader = DataLoader(train, shuffle=True, batch_size=args.per_device_train_batch_size, collate_fn=train_collator)
    eval_loader = DataLoader(validation, batch_size=args.per_device_eval_batch_size, collate_fn=eval_collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_scheduler('linear', optimizer, num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
                              num_training_steps=args.max_train_steps * accelerator.num_processes)
    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, eval_loader, scheduler)
    step = 0
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        step = int(Path(args.resume_from_checkpoint).name.removeprefix('step_'))
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    start_epoch = step // updates_per_epoch
    skipped_batches = (step % updates_per_epoch) * args.gradient_accumulation_steps
    progress = tqdm(total=args.max_train_steps, initial=step, disable=not accelerator.is_local_main_process)
    last_metrics = {}
    for epoch in range(start_epoch, start_epoch + math.ceil(max(0, args.max_train_steps - step) / updates_per_epoch) + 1):
        if hasattr(train_loader, 'set_epoch'):
            train_loader.set_epoch(epoch)
        active = accelerator.skip_first_batches(train_loader, skipped_batches) if epoch == start_epoch and skipped_batches else train_loader
        model.train()
        for batch in active:
            if step >= args.max_train_steps:
                break
            with accelerator.accumulate(model):
                try:
                    loss = model(**batch).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite training loss.')
                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                finally:
                    functional.reset_net(model)
            if not accelerator.sync_gradients:
                continue
            step += 1
            progress.update(1)
            progress.set_postfix(loss=float(loss.detach()))
            if args.eval_steps and step % args.eval_steps == 0:
                last_metrics = evaluate(model, eval_loader, accelerator)
                accelerator.print(json.dumps({'step': step, **last_metrics}))
                model.train()
            if args.checkpointing_steps and step % args.checkpointing_steps == 0:
                state_dir = output / f'step_{step}'
                accelerator.save_state(str(state_dir))
                if accelerator.is_main_process:
                    accelerator.unwrap_model(model).config.save_pretrained(state_dir)
                    tokenizer.save_pretrained(state_dir)
        if step >= args.max_train_steps:
            break
    progress.close()
    last_metrics = evaluate(model, eval_loader, accelerator)
    accelerator.wait_for_everyone()
    accelerator.unwrap_model(model).save_pretrained(output, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output)
        (output / 'all_results.json').write_text(json.dumps({'step': step, **last_metrics}, indent=2))
    accelerator.print(json.dumps(last_metrics))


if __name__ == '__main__':
    main()
