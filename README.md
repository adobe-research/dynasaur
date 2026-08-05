<h1 align="center">DynaSaur 🦖: Large Language Agents<br>Beyond Predefined Actions</h1>

<p align="center">
    <a href="https://arxiv.org/abs/2411.01747">
        https://img.shields.io/badge/arXiv-2411.01747-b31b1b.svg
    </a>
    <a href="https://openreview.net/forum?id=lv0cJ2pWVd">
        https://img.shields.io/badge/COLM-2025-blue.svg
    </a>
</p>

DynaSaur is a dynamic LLM-based agent framework published at COLM 2025 that uses a programming language as a universal representation of its actions. At each step, it generates a Python snippet that either calls on existing actions or creates new ones when the current action set is insufficient. These new actions can be developed from scratch or formed by composing existing actions, gradually expanding a reusable library for future tasks.

Empirically, DynaSaur exhibits remarkable versatility, recovering automatically when no relevant actions are available or when existing actions fail due to unforeseen edge cases. As of this writing, it holds the top position on the [GAIA benchmark](https://huggingface.co/spaces/gaia-benchmark/leaderboard), and remains the leading non-ensemble method to date.

# Installation

### 1. Create a `.env` file and add your keys:
```bash

# Required: Main keys for the agent
AZURE_API_KEY=""
AZURE_ENDPOINT=""
AZURE_API_VERSION=""

# Required: Keys for embeddings used in action retrieval
EMBED_MODEL_TYPE="AzureOpenAI"
AZURE_EMBED_MODEL_NAME=""
AZURE_EMBED_API_KEY=""
AZURE_EMBED_ENDPOINT=""
AZURE_EMBED_API_VERSION=""

# Optional: Keys for user-defined actions. 
# You don't need these if you won't use those actions.
SERPAPI_API_KEY=""
AZURE_GPT4V_API_KEY=""
AZURE_GPT4V_ENDPOINT=""
AZURE_GPT4V_API_VERSION=""
```

### 2. Download the GAIA files:
You will need a Hugging Face (HF) access token with write permissions before cloning the GAIA repository. Visit [this page](https://huggingface.co/settings/tokens) to generate your token. Then log in to HF using the following command:
```bash
huggingface-cli login
```
Now we download the files with the commands below:
```bash
mkdir data
git clone https://huggingface.co/datasets/gaia-benchmark/GAIA
mv GAIA/2023 data/gaia/
rm -rf GAIA
```

### 3. Set up the environment:
```bash
conda create -n dynasaur python==3.12
conda activate dynasaur
pip install -r requirements.txt
```

# Let the 🦖 take over
```bash
python dynasaur.py
```

# TODOs
- [ ] Add support for the OpenAI API

# Citation
If you find this work useful, please cite our [paper](https://arxiv.org/pdf/2411.01747):
```bibtex
@inproceedings{nguyen2024dynasaur,
  title   = {DynaSaur: Large Language Agents Beyond Predefined Actions},
  author  = {Dang Nguyen and Viet Dac Lai and Seunghyun Yoon and Ryan A. Rossi and Handong Zhao and Ruiyi Zhang and Puneet Mathur and Nedim Lipka and Yu Wang and Trung Bui and Franck Dernoncourt and Tianyi Zhou},
  year    = {2025},
  booktitle = {Proceedings of the Conference on Language Modeling}
}
```
