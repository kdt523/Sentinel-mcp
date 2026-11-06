# notebooks

GPU notebooks (Colab/Kaggle) for SFT, GRPO and fast-tier training. Never run on the laptop.

## Notebooks

| Notebook | Purpose | Runtime |
|---|---|---|
| [01_llm_sessions_colab.ipynb](01_llm_sessions_colab.ipynb) | Dataset step 7: paraphrase benign prompts and run LLM-driven sessions with `qwen2.5:7b` (Ollama) against the in-process gateway. Works on **Kaggle** and **Colab**; builds its own Python 3.12 environment | GPU (Kaggle T4/P100, Colab T4) |

Run all cells. Kaggle: turn on a GPU and Internet; results are in `/kaggle/working/llm_sessions/`.
Colab: results go to Google Drive. Pin `COMMIT` for the final run: samples carry a `build_id`
and the dataset builder refuses to mix them. The LLM-session run belongs on a GPU service, not the
laptop (about 34 s per session on laptop CPU, about 6 s on a T4).
