The basic premise of this lab is to first fine-tune a transformer-encoder called BERT. We will then show how it and other models can be trained more efficiently by introducing [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). We will finally fine-tune a quantized version of gemma2-2b with the help of LoRa implementation of PEFT, a huggingface library.

## Instructions

Apart from part 2, all parts utilize huggingface functions to download the transformer-based models. To download gemma2 at the end you need to create a huggingface access token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and save it inside a .env file in the root directory. (HF_KEY="your_key")

## References and Further Reading

### Part 1:
* https://huggingface.co/docs/transformers/training

### Part 2:
* https://github.com/davisgcii/basicLoRA
* https://www.linkedin.com/pulse/more-efficient-finetuning-implementing-lora-from-scratch-george-davis/
* https://github.com/sunildkumar/lora_from_scratch

### Part 3:
* https://huggingface.co/blog/peft
* https://huggingface.co/docs/trl/main/en/sft_trainer#training-adapters