<h1 align="center">DynaSaur 🦖: Large Language Agents<br>Beyond Predefined Actions</h1>

DynaSaur is a dynamic LLM-based agent framework that uses a programming language as a universal representation of its actions. At each step, it generates a Python snippet that either calls on existing actions or creates new ones when the current action set is insufficient. These new actions can be developed from scratch or formed by composing existing actions, gradually expanding a reusable library for future tasks.

Empirically, DynaSaur exhibits remarkable versatility, recovering automatically when no relevant actions are available or when existing actions fail due to unforeseen edge cases. As of this writing, it holds the top position on the [GAIA benchmark](https://huggingface.co/spaces/gaia-benchmark/leaderboard), and remains the leading non-ensemble method to date.

# Installation

### 1. Create a `.env` file and add your keys:
```bash

# Required: Main keys for the agent (we also support the OpenAI API)
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

# Citation
If you find this work useful, please cite the following:
```bibtex
@article{nguyen2024dynasaur,
  title   = {DynaSaur: Large Language Agents Beyond Predefined Actions},
  author  = {Dang Nguyen and Viet Dac Lai and Seunghyun Yoon and Ryan A. Rossi and Handong Zhao and Ruiyi Zhang and Puneet Mathur and Nedim Lipka and Yu Wang and Trung Bui and Franck Dernoncourt and Tianyi Zhou},
  year    = {2024},
  journal = {arXiv preprint arXiv:2411.01747}
}
```
