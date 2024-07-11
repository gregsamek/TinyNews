# Tiny News

For a detailed overview of this project from start to finish, check out [GregSamek.github.io/TinyNews](https://GregSamek.github.io/TinyNews)

TinyNews is a collection of one million synthetically generated news bulletins and several language models scratch-trained on this data.  Evaluations suggests that TinyNews retains ~80% of the quality of the training data while using ~1/1000th the number of parameters as the models used to generate it.

Trained models and training data are available on [🤗 Hugging Face](https://huggingface.co/collections/GregSamek/tinynews-668aff540bf195d6e5e0e40f)

This project is essentially a modified reimplementation of the Microsoft Research [TinyStories](https://arxiv.org/abs/2305.07759) project.

TinyNews also borrows heavily from Sebastian Raschka's excellect book [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)

## Scripts

Download one of the models from [🤗 Hugging Face](https://huggingface.co/collections/GregSamek/tinynews-668aff540bf195d6e5e0e40f) and check `demo.ipynb` to generate text in just a few lines of code.

`generate_training_data.py` is used with an [openrouter](https://openrouter.ai/) api key to generate synthetic data using `vocabulary.json` as seeds and `few_shot_examples.json` as in-context examples.

`train_tokenizer.ipynb` is used to clean up the synthetic data and train a custom BPE tokenizer with sentencepiece. Pretrained tokenizers are found in teh `tokenizers` directory and are named after their vocab size.

`train.py` is used to train models. Each model on Huggingface comes with a `config.json` file that is compatible with this script. Example config:
```json
{    
    "model":  {
        "vocab_size": 8192,
        "context_length": 128,
        "d_embedding": 128,
        "d_intermediate": 512,
        "n_heads": 4,
        "n_layers": 4,
        "qkv_bias": false
    },
    "train": {
        "peak_lr": 0.001,
        "warmup_ratio": 0.01,
        "n_epochs": 2,
        "batch_size": 8,
        "weight_decay": 0.1
    }
}
```

`/results/` directory contains loss curves, evaluations by Llama3 70B as a judge, etc. I walk through all of this on the [Project Page](https://GregSamek.github.io/TinyNews)