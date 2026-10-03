"""Fine-tuning SPE on GLUE and the paper's text classification tasks."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset, load_from_disk
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from spikingjelly.activation_based import functional
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding, get_scheduler

from checkpoint_utils import load_classifier

TASK_KEYS = {'cola': ('sentence',), 'sst2': ('sentence',), 'mrpc': ('sentence1', 'sentence2'),
             'stsb': ('sentence1', 'sentence2'), 'qqp': ('question1', 'question2'),
             'mnli': ('premise', 'hypothesis'), 'qnli': ('question', 'sentence'),
             'rte': ('sentence1', 'sentence2'), 'mr': ('text',), 'sst5': ('text',),
             'subj': ('text',), 'ag_news': ('text',), 'imdb': ('text',)}
TEXT_DATASETS = {'mr': 'rotten_tomatoes', 'sst5': 'SetFit/sst5', 'subj': 'SetFit/subj',
                 'ag_news': 'SetFit/ag_news', 'imdb': 'SetFit/imdb'}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task_name', choices=TASK_KEYS, required=True)
    parser.add_argument('--model_name_or_path', required=True, help='Downloaded SPE checkpoint with config and tokenizer.')
    parser.add_argument('--dataset_path', help='Optional local DatasetDict saved with save_to_disk().')
    parser.add_argument('--cache_dir')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_length', type=int, help='Default: 1024 for AGNEWS/IMDB, 128 for other tasks.')
    parser.add_argument('--per_device_train_batch_size', type=int, default=32)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--max_train_steps', type=int)
    parser.add_argument('--num_warmup_steps', type=int, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_train_samples', type=int)
    parser.add_argument('--max_eval_samples', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--neuron_backend', choices=['cupy', 'torch'], default='cupy')
    parser.add_argument('--mixed_precision', choices=['no', 'fp16', 'bf16'], default='no')
    args = parser.parse_args()
    args.max_length = args.max_length or (1024 if args.task_name in ['ag_news', 'imdb'] else 128)
    return args


def metrics(task, predictions, references):
    if task == 'cola':
        return {'matthews_correlation': float(matthews_corrcoef(references, predictions))}
    if task == 'stsb':
        return {'pearson': float(pearsonr(references, predictions).statistic),
                'spearmanr': float(spearmanr(references, predictions).statistic)}
    result = {'accuracy': float(accuracy_score(references, predictions))}
    if task in ['mrpc', 'qqp']:
        result['f1'] = float(f1_score(references, predictions, zero_division=0))
    return result


def main():
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision)
    set_seed(args.seed)
    output = Path(args.output_dir)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'training_args.json').write_text(json.dumps(vars(args), indent=2))
    if args.dataset_path:
        raw = load_from_disk(args.dataset_path)
    elif args.task_name in TEXT_DATASETS:
        raw = load_dataset(TEXT_DATASETS[args.task_name], cache_dir=args.cache_dir)
    else:
        raw = load_dataset('nyu-mll/glue', args.task_name, cache_dir=args.cache_dir)
    regression = args.task_name == 'stsb'
    label_values = sorted(raw['train'].unique('label')) if not regression else []
    label_map = {value: i for i, value in enumerate(label_values)}
    num_labels = 1 if regression else len(label_values)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model, load_report = load_classifier(args.model_name_or_path, num_labels, args.neuron_backend, args.max_length)
    accelerator.print('Checkpoint load:', json.dumps(load_report))
    model.config.problem_type = 'regression' if regression else 'single_label_classification'
    if not regression:
        feature = raw['train'].features['label']
        names = getattr(feature, 'names', None)
        names = [names[v] for v in label_values] if names else [str(v) for v in label_values]
        model.config.label2id = {name: i for i, name in enumerate(names)}
        model.config.id2label = {i: name for i, name in enumerate(names)}

    def tokenize(examples):
        encoded = tokenizer(*(examples[key] for key in TASK_KEYS[args.task_name]), truncation=True, max_length=args.max_length)
        encoded['labels'] = examples['label'] if regression else [label_map.get(value, -100) for value in examples['label']]
        return encoded

    # Unlabelled official test splits are omitted; GLUE uses development splits.
    evaluation_splits = ['validation_matched', 'validation_mismatched'] if args.task_name == 'mnli' else [
        'test' if args.task_name in TEXT_DATASETS else 'validation']
    selected = ['train', *evaluation_splits]
    with accelerator.main_process_first():
        processed = {split: raw[split].map(tokenize, batched=True, remove_columns=raw[split].column_names) for split in selected}
    if args.max_train_samples:
        processed['train'] = processed['train'].select(range(min(args.max_train_samples, len(processed['train']))))
    if args.max_eval_samples:
        for split in evaluation_splits:
            processed[split] = processed[split].select(range(min(args.max_eval_samples, len(processed[split]))))
    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(processed['train'], shuffle=True, batch_size=args.per_device_train_batch_size, collate_fn=collator)
    eval_loaders = {split: DataLoader(processed[split], batch_size=args.per_device_eval_batch_size, collate_fn=collator) for split in evaluation_splits}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    eval_loaders = {split: accelerator.prepare(loader) for split, loader in eval_loaders.items()}
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = args.max_train_steps or args.num_train_epochs * updates_per_epoch
    scheduler = get_scheduler('linear', optimizer, num_warmup_steps=args.num_warmup_steps,
                              num_training_steps=total_steps)
    accelerator.register_for_checkpointing(scheduler)
    epochs = math.ceil(total_steps / updates_per_epoch)
    progress = tqdm(total=total_steps, disable=not accelerator.is_local_main_process)
    primary = {'cola': 'matthews_correlation', 'stsb': 'spearmanr', 'mrpc': 'f1', 'qqp': 'f1'}.get(args.task_name, 'accuracy')
    best_score, best_epoch, step, history = -float('inf'), None, 0, []
    for epoch in range(epochs):
        if hasattr(train_loader, 'set_epoch'):
            train_loader.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            with accelerator.accumulate(model):
                try:
                    loss = model(**batch).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite fine-tuning loss.')
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                    if accelerator.sync_gradients:
                        scheduler.step()
                finally:
                    functional.reset_net(model)
            if accelerator.sync_gradients:
                step += 1
                progress.update(1)
                progress.set_postfix(loss=float(loss.detach()))
            if step >= total_steps:
                break
        model.eval()
        epoch_metrics = {}
        for split, loader in eval_loaders.items():
            predictions, references = [], []
            for batch in loader:
                try:
                    with torch.no_grad():
                        logits = model(**batch).logits
                    predicted = logits.squeeze(-1) if regression else logits.argmax(-1)
                    predicted, labels = accelerator.gather_for_metrics((predicted, batch['labels']))
                    predictions.extend(predicted.cpu().tolist())
                    references.extend(labels.cpu().tolist())
                finally:
                    functional.reset_net(model)
            epoch_metrics[split] = metrics(args.task_name, np.asarray(predictions), np.asarray(references))
        history.append({'epoch': epoch + 1, 'step': step, **epoch_metrics})
        accelerator.print(json.dumps(history[-1]))
        score = epoch_metrics[evaluation_splits[0]][primary]
        if score > best_score:
            best_score, best_epoch = score, epoch + 1
            accelerator.wait_for_everyone()
            accelerator.unwrap_model(model).save_pretrained(output / 'best', is_main_process=accelerator.is_main_process, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output / 'best')
        if accelerator.is_main_process:
            (output / 'all_results.json').write_text(json.dumps({'best_epoch': best_epoch, 'best_metric': primary,
                'best_score': best_score, 'history': history}, indent=2))
        if step >= total_steps:
            break
    progress.close()


if __name__ == '__main__':
    main()
