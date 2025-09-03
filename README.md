# 🎲 Analyzing Gambling Addictions: A Spanish Corpus for Understanding Pathological Behavior

This repository accompanies the paper:  
**"Analyzing Gambling Addictions: A Spanish Corpus for Understanding Pathological Behavior"**  
📍 Accepted at *Findings of EMNLP 2025*.


## 📂 Dataset

The main contribution of this work is a **Spanish sentence retrieval dataset** focused on symptoms associated with pathological gambling.

- **Corpus:** `resources/dataset/corpus.jsonl`  
- **Queries & Qrels:** also available in the same directory. Since 4 assessors were employed, the majority agreement is considered when at least two authors consider the sentence relevant. 
- All files follow the [BEIR](https://github.com/beir-cellar/beir) compatible format, enabling easy use with standard baselines (see Section 4 of the paper).  

Additionally, a **subfolder with pools** is provided, containing the material used by both human annotators and LLMs for dataset labeling.

---

## ⚙️ Code

The `src` folder is structured as follows:

- **`train/`** → Training scripts for our domain-adapted **ludoBETO** model.  
- **`labelling/`** → Statistics and analysis of human vs. automatic label generation. "David Gallego" user is the trained psychologist.

---

## 🤖 Model

We introduce **[ludoBETO](https://huggingface.co/citiusLTL/ludoBETO)**, a BETO-based model adapted to the pathological gambling domain.  
This model is publicly available on HuggingFace for further research and fine-tuning.

🔧 In our paper, we also implemented a **cross-encoder** using the [SimCSE](https://www.sbert.net/examples/sentence_transformer/unsupervised_learning/SimCSE/README.html) strategy with custom parameters over ludoBETO.

---

## 📖 Citation

If you use this resource, please cite:

```bibtex
@inproceedings{couto-etal-2025,
    title = "Analyzing Gambling Addictions: A Spanish Corpus for Understanding Pathological Behavior",
    author = "Couto-Pintos, Manuel and
              Fernández-Pichel, Marcos and
               Aragón, Mario Ezra and
              Losada, David E.",
    booktitle = "Findings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)"
}
