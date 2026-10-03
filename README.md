# SPE: Positional Encoding for Spiking Transformers

Code for **Positional Encoding for Spiking Transformers** (ICML 2026), by Zijian
Zhou, Yu Liang, Honglin Cao, Ammar Belatreche, Jieyuan Zhang, Wenjie Wei, Shuai
Wang, Malu Zhang, Yang Yang, and Haizhou Li.

SPE encodes positions through position-dependent thresholds in the Q/K PE-LIF
neurons. Its Membrane Potential Regularization Surrogate Gradient (MPR-SG)
regularizes the discrepancy between batch-mean membrane potentials and spikes.
This repository contains the main pre-training and fine-tuning experiments.

## Released checkpoint

Download the pre-trained weights from **[ModelScope: kailai1104/SPE](https://modelscope.cn/models/kailai1104/SPE)**.

| Setting | Value |
| --- | --- |
| Source checkpoint | `snn_pe2_110M/step_650000` |
| Encoder | 12 layers, hidden size 768, intermediate size 3072, 12 heads |
| Vocabulary | BERT uncased, 30,522 tokens |
| Simulation steps | 4 |
| PE-LIF | tau 2.0, base threshold 1.0, sinusoidal amplitude 0.3 |
| MPR-SG | sigmoid slope 4.0, numerical epsilon 1e-6 |
| Pre-training sequence length | 128 |
| Checkpoint size | 438,310,944 bytes |
| SHA256 | `1a69961b10bde0c1c6d7e35e4d7ee0789fd33df588d78fedd375c298ace11c45` |

The public ModelScope download was verified against the SHA256 above.

The checkpoint contains the original, unmodified weights plus the matching
configuration and tokenizer. Optimizer, scheduler, and random-state files are
not distributed. The custom model is `spikingbert_pelif_simsig_stable.py`; the
PE-LIF/MPR-SG implementation is `pelif_simsig_stable.py`.

## Installation

Validated on Python 3.10, PyTorch 2.8.0, Transformers 4.57.0, SpikingJelly
0.0.0.0.14, CuPy 13.0.0, and one NVIDIA H100. The model uses Transformers
internals, so keep its pinned version.

```bash
git clone https://github.com/CayleyZ/SPE.git
cd SPE
pip install -r requirements.txt
# For a CUDA 12.x runtime:
pip install cupy-cuda12x==13.0.0
pip install modelscope_hub
python -c "from modelscope_hub import HubApi; HubApi().download_repo('kailai1104/SPE', 'model', local_dir='checkpoints/SPE')"
```

CuPy must match the CUDA runtime. CPU inference and small training runs can use
`--neuron_backend torch`; evaluation also needs `--device cpu`.

## Verify the weights

For this bidirectional masked language model, `pseudo_ppl` masks **one token at
a time**; `masked_ppl` evaluates the **15% MLM masking** used in pre-training.
These are different protocols and are not autoregressive language-model PPL.

```bash
# SST-2 development rows 0-4; mask every non-special token separately.
python evaluate_mlm.py \
  --model_name_or_path checkpoints/SPE \
  --mode pseudo --dataset_name nyu-mll/glue --dataset_config_name sst2 \
  --split validation --num_samples 5 --max_length 128 \
  --output results/pseudo_ppl.json

# A local tokenized pre-training validation split.
python evaluate_mlm.py \
  --model_name_or_path checkpoints/SPE --mode masked \
  --dataset_path /path/to/128_tokenized_data \
  --num_samples 256 --batch_size 8 --seed 42 \
  --output results/masked_ppl.json
```

| Check | Result |
| --- | --- |
| Original model: SST-2 pseudo-PPL, 5 samples / 77 tokens | 12.1288614031 |
| Released model: same inputs and weights | 12.1288614031 |
| MLM PPL: 256 local validation blocks / 4,835 masked tokens | 7.5602651847 |
| MLM weight loading | All 274 stored tensors match; zero shape mismatches |
| Pre-training and fine-tuning | Two optimizer steps, evaluation, and checkpoint saving passed |

The local corpus MLM value is specific to the available validation blocks and
seed, not a universal benchmark score. The five-sample pseudo-PPL check is a
checkpoint compatibility check. Full downstream training was not repeated for
this release. See [validation/checkpoint_validation.json](validation/checkpoint_validation.json)
for the recorded checks.

The two absent MLM decoder aliases are tied to the stored input embedding
weight and prediction bias. The loader verifies these ties and rejects any
missing independent learned tensor or shape mismatch. During fine-tuning, the
MLM head is discarded; the classification head and BERT pooler are initialized
from scratch, while every encoder tensor is loaded from the checkpoint.

## Pre-training

The main configuration uses MLM probability 0.15, sequence length 128, AdamW
with learning rate 2e-4 and zero weight decay, 5,000 warm-up steps followed by
linear decay, and 650,000 optimizer steps. Eight GPUs with batch size 64 each
give a global batch size of 512 and approximately 42.6 billion processed tokens.

Prepare the paper's five corpora (TinyStories, BookCorpus, CC-News, OpenWebText,
and Wikipedia) with Hugging Face Datasets. `--dataset_name` accepts directories
saved with `save_to_disk()` or dataset IDs; each raw corpus needs a `text`
column. When a corpus lacks validation data, a held-out split is created
without reusing training examples. JSON/CSV/text files are also supported.

To use the preprocessed corpus from the original experiment:

```bash
TOKENIZED_DATASET=/path/to/128_tokenized_data NUM_GPUS=8 \
  bash scripts/pretrain.sh
```

That directory must be a `DatasetDict` with `train` and `validation` splits
containing `input_ids`, `attention_mask`, and optionally `token_type_ids` and
`special_tokens_mask`. Raw corpora can be processed directly:

```bash
torchrun --standalone --nproc_per_node=8 pretrain.py \
  --dataset_name /path/to/stories /path/to/bookcorpus /path/to/cc_news \
                 /path/to/openwebtext /path/to/wikipedia \
  --config_name configs/spe_110m.json --tokenizer_name bert-base-uncased \
  --output_dir checkpoints/spe_110m
```

`pretrain.py` initializes a new SPE model; its tokenizer/config arguments do
not load ANN BERT weights. Saved `step_*` directories contain Accelerate state,
configuration, and tokenizer assets. Resume a local training run with
`--resume_from_checkpoint checkpoints/spe_110m/step_50000`. The distributed
launch configuration is provided; the release checks ran on one GPU.

## Fine-tuning

Supported tasks: CoLA, SST-2, MRPC, STS-B, QQP, MNLI (matched and mismatched),
QNLI, RTE, MR, SST-5, Subj, AGNEWS, and IMDB. GLUE is evaluated on development
splits. Other text classification tasks use their provided test splits, as in
the original scripts. `--dataset_path` can point to a local `DatasetDict` with
the matching text/label columns.

```bash
MODEL_PATH=checkpoints/SPE bash scripts/finetune.sh sst2
MODEL_PATH=checkpoints/SPE bash scripts/finetune.sh mnli
MODEL_PATH=checkpoints/SPE bash scripts/finetune.sh ag_news
MODEL_PATH=checkpoints/SPE bash scripts/finetune.sh imdb
```

The defaults are batch size 32, AdamW learning rate 2e-5, zero weight decay,
linear decay without warm-up, and no gradient accumulation. Maximum length is
128 for short-text/GLUE tasks and 1024 for AGNEWS/IMDB. The script's epoch
defaults come from the existing task scripts and can be overridden. The best
checkpoint is written to `<output_dir>/best`, with metrics in `all_results.json`.

For 1024-token tasks, PE-LIF thresholds extend to that length and the learned
absolute positional embedding table is extended by repeating its final row,
matching the source fine-tuning implementation. The pre-training checkpoint's
absolute embedding table and attention evaluation order (`QK^T`, then `V`) are
preserved for numerical compatibility. SPE itself also permits reassociation
to `Q(K^T V)`; this release does not implement that alternative evaluation order.

## Direct inference

```python
import torch
from transformers import AutoTokenizer
from spikingjelly.activation_based import functional
from checkpoint_utils import load_mlm

path = 'checkpoints/SPE'
tokenizer = AutoTokenizer.from_pretrained(path)
model, report = load_mlm(path)
model.cuda().eval()
batch = tokenizer('Spiking positional encoding is [MASK].', return_tensors='pt')
with torch.no_grad():
    logits = model(**{key: value.cuda() for key, value in batch.items()}).logits
functional.reset_net(model)
```

Use the custom SPE class/loader. `AutoModelForMaskedLM` would instantiate an
ordinary BERT. Always reset spiking neuron states between independent batches;
the included training and evaluation scripts do this automatically.

## Repository layout

```text
spikingbert_pelif_simsig_stable.py  Checkpoint-compatible spiking BERT model
pelif_simsig_stable.py            PE-LIF and MPR-SG
checkpoint_utils.py              Validated loading for MLM and classification
pretrain.py                      MLM pre-training
finetune.py                      GLUE and text classification fine-tuning
evaluate_mlm.py                  Masked PPL / pseudo-PPL checks
configs/spe_110m.json             Released architecture and neuron parameters
scripts/                        Main experiment launchers
validation/                     Checkpoint verification record
```

## Citation

```bibtex
@inproceedings{zhou2026spe,
  title={Positional Encoding for Spiking Transformers},
  author={Zhou, Zijian and Liang, Yu and Cao, Honglin and Belatreche, Ammar and
          Zhang, Jieyuan and Wei, Wenjie and Wang, Shuai and Zhang, Malu and
          Yang, Yang and Li, Haizhou},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}
```

## License

Apache 2.0. The BERT implementation retains its upstream copyright notices;
see [LICENSE](LICENSE) and [NOTICE](NOTICE).
