
# DR-FET Codebase

This repository contains the implementation of **DR-FET**, a descriptor-based retrieval framework for fine-grained entity typing with distant supervision.

The pipeline is controlled via a single YAML configuration file and executed through `main.py`.

---

## 1. Environment Setup

* Python ≥ 3.9
* PyTorch ≥ 2.0
* Required packages are listed in `requirements.txt`

Before running, please make sure all models are downloaded locally or accessible from the specified paths.

---

## 2. Configuration File (`config.yaml`)

All experiments are configured through `config.yaml`. Below is an example:

```yaml
model:
  model_name: "qwen7b"
  model_path: "YOUR_PATH_TO_MODEL\\Qwen2.5-7B-Instruct"
  encoder_name: "bert"
  encoder_path: "YOUR_PATH_TO_MODEL\\bert-base-cased"

stage:
  pre_filter: true
  train: true
  test: true
```

### 2.1 Model Configuration

```yaml
model:
  model_name: "qwen7b"
  model_path: "..."
```

* `model_name`: Identifier of the LLM backend (used for logging and checkpoints).
* `model_path`: Local path to the instruction-tuned LLM used for training and inference.

```yaml
encoder_name: "bert"
encoder_path: "..."
```

* `encoder_name`: Sentence encoder used for descriptor-based retrieval.
* `encoder_path`: Local path to the encoder model (e.g., BERT, E5, BGE).

The encoder is used **only for descriptor embedding and candidate retrieval**, not for final prediction.

---

### 2.2 Pipeline Control (`stage`)

```yaml
stage:
  pre_filter: true
  train: true
  test: true
```

Each flag controls one stage of the DR-FET pipeline:

* `pre_filter`:
  Runs descriptor-based retrieval to filter distant supervision data and construct a high-confidence training set.

* `train`:
  Fine-tunes the LLM using the filtered training data.

* `test`:
  Performs candidate-constrained inference on the test set and reports evaluation metrics.

You may enable or disable stages depending on your needs.
For example, to only run evaluation using an existing checkpoint:

```yaml
stage:
  pre_filter: false
  train: false
  test: true
```

---

## 3. Running the Pipeline (`main.py`)

The entire pipeline is executed via:

```bash
python main.py
```


