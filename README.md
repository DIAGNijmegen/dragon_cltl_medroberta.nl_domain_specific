# DRAGON CLTL MedRoBERTa.nl Domain-specific Algorithm

Adaptation of [DRAGON baseline](https://github.com/DIAGNijmegen/dragon_baseline) (version `0.2.1`), with pretrained foundational model `CLTL/MedRoBERTa.nl` [1].

For details on the pretrained foundational model, check out HuggingFace: [huggingface.co/CLTL/MedRoBERTa.nl](https://huggingface.co/CLTL/MedRoBERTa.nl).

The following adaptations were made to the DRAGON baseline:

```python
model_name = "CLTL/MedRoBERTa.nl"
```

These settings were kept the same:

```python
per_device_train_batch_size = 4
gradient_accumulation_steps = 2
gradient_checkpointing = False
max_seq_length = 512
learning_rate = 1e-05
```

Additionally, a bug with the tokenizer truncation length was fixed in the `predict_huggingface` function.

**References:**

1. Verkijk, S. & Vossen, P. (2022) MedRoBERTa.nl: A Language Model for Dutch Electronic Health Records. Computational Linguistics in the Netherlands Journal, 11.
