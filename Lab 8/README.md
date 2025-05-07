The basic premise of this lab is to first fine-tune a transformer-encoder called BERT. We will then show how it and other models can be trained more efficiently by introducing [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). We will finally fine-tune a quantized version of gemma2-2b with the help of LoRa implementation of PEFT, a huggingface library. Notebooks were taken from the links listed below with slight changes: Part 1 notebook is shortened to only cover training in native pytorch, while Part 2 notebook is expanded to cover the case where the rank is switched from 4 to 16 for better accuracy in cat, dog and deer classes while at the same time minimizing catastrophic forgetting.

## Instructions

Apart from part 2, all parts utilize huggingface functions to download the transformer-based models. To download gemma2 at the end you need to create a huggingface access token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and save it inside a .env file in the root directory. (HF_KEY="your_key")

## References and Further Reading

### Part 1: Downloading models from huggingface and fine-tuning them
* https://huggingface.co/docs/transformers/training

### Part 2: LoRa from scratch
* Blog post: https://www.linkedin.com/pulse/more-efficient-finetuning-implementing-lora-from-scratch-george-davis/
* Main repo: https://github.com/davisgcii/basicLoRA
* Another good repo: https://github.com/sunildkumar/lora_from_scratch

### Part 3: PEFT and Quantized LLM Fine-tuning
* PEFT: https://huggingface.co/blog/peft
* Datset used: https://huggingface.co/datasets/tomg-group-umd/CLRS-Text-train
* Supervised Fine-tuning Trainer: https://huggingface.co/docs/trl/main/en/sft_trainer#training-adapters