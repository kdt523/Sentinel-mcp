# notebooks

GPU notebooks (Colab/Kaggle) for SFT, GRPO and fast-tier training. Never run on the laptop.

## Notebooks

| Notebook | Purpose | Runtime |
|---|---|---|
| [01_llm_sessions_colab.ipynb](01_llm_sessions_colab.ipynb) | Dataset step 7: paraphrase benign prompts and run LLM-driven sessions with `qwen2.5:7b` (Ollama) against the in-process gateway; results go to Google Drive | Colab GPU (T4) |

The LLM-session run belongs on Colab, not the laptop (about 34 s per session on laptop CPU, a few seconds on a T4). Pin
`COMMIT` to the commit of the final scripted capture: samples carry a `build_id` and the
dataset builder refuses to mix them.
