"""Strict checkpoint loading shared by inference and fine-tuning."""

from pathlib import Path

from safetensors.torch import load_file
from transformers import AutoConfig

from spikingbert_pelif_simsig_stable import BertForMaskedLM, BertForSequenceClassification


def load_config(model_dir, neuron_backend='cupy', max_length=None):
    config = AutoConfig.from_pretrained(model_dir)
    config.T = getattr(config, 'T', 4)
    config.pelif_tau = getattr(config, 'pelif_tau', 2.0)
    config.pelif_k = getattr(config, 'pelif_k', 0.3)
    config.pelif_token_num = max(getattr(config, 'pelif_token_num', 128), max_length or 128)
    config.simsig_alpha = getattr(config, 'simsig_alpha', 4.0)
    config.simsig_eps = getattr(config, 'simsig_eps', 1e-6)
    config.neuron_backend = neuron_backend
    config._attn_implementation = 'eager'
    config.use_cache = False
    return config


def load_checkpoint(model, checkpoint_dir, classification=False, extend_positions=False):
    state_path = Path(checkpoint_dir) / 'model.safetensors'
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    saved = load_file(str(state_path), device='cpu')
    expected = model.state_dict()
    ignored = []
    unexpected = []
    extended = []
    for name, value in list(saved.items()):
        if classification and name.startswith('cls.'):
            ignored.append(name)
            del saved[name]
            continue
        if name not in expected:
            unexpected.append(name)
            continue
        if value.shape != expected[name].shape:
            if (extend_positions and name == 'bert.embeddings.position_embeddings.weight'
                    and value.shape[1:] == expected[name].shape[1:]
                    and value.shape[0] < expected[name].shape[0]):
                replacement = expected[name].clone()
                replacement[:value.shape[0]] = value
                replacement[value.shape[0]:] = value[-1:]
                saved[name] = replacement
                extended.append(name)
            else:
                raise ValueError(f'Checkpoint shape mismatch for {name}: {value.shape} vs {expected[name].shape}')
    if unexpected:
        raise ValueError(f'Unexpected checkpoint keys: {unexpected}')
    missing = set(expected) - set(saved)
    if classification:
        allowed = {'classifier.weight', 'classifier.bias', 'bert.pooler.dense.weight', 'bert.pooler.dense.bias'}
    else:
        allowed = {'cls.predictions.decoder.weight', 'cls.predictions.decoder.bias'}
        if model.cls.predictions.decoder.weight is not model.bert.embeddings.word_embeddings.weight:
            raise ValueError('The MLM decoder weight is not tied to word embeddings.')
        if model.cls.predictions.decoder.bias is not model.cls.predictions.bias:
            raise ValueError('The MLM decoder bias is not tied to the prediction bias.')
    if missing - allowed:
        raise ValueError(f'Missing learned checkpoint tensors: {sorted(missing - allowed)}')
    model.load_state_dict(saved, strict=False)
    return {'checkpoint_keys': len(saved) + len(ignored), 'loaded_keys': len(saved),
            'missing_keys': sorted(missing), 'ignored_mlm_head_keys': ignored,
            'extended_position_embeddings': extended, 'shape_mismatches': []}


def load_mlm(model_dir, neuron_backend='cupy', max_length=None):
    model = BertForMaskedLM(load_config(model_dir, neuron_backend, max_length))
    report = load_checkpoint(model, model_dir)
    return model, report


def load_classifier(model_dir, num_labels, neuron_backend='cupy', max_length=128):
    config = load_config(model_dir, neuron_backend, max_length)
    config.num_labels = num_labels
    extend = max_length > config.max_position_embeddings
    if extend:
        config.max_position_embeddings = max_length
    model = BertForSequenceClassification(config)
    report = load_checkpoint(model, model_dir, classification=True, extend_positions=extend)
    return model, report
