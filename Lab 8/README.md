The premise of this lab is to showcase common functions in the `Transformers` library to download and fine-tune open-weight LLMs. After getting acquainted with `Transformers` and `Datasets` libraries, we will implement [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) from scratch.  We will then fine-tune a quantized version of TinyLlama with the help of LoRa implementation of PEFT which is another huggingface library. 

First two notebooks were adapted from the links listed below with slight changes: Part 1 notebook is shortened to only cover training in native pytorch, while Part 2 notebook is expanded to cover the case where the rank is switched from 4 to 16 for better accuracy in cat, dog and deer classes.

## References and Further Reading

### Part 1: Downloading models from huggingface and fine-tuning them
* https://huggingface.co/docs/transformers/training

### Part 2: LoRa from scratch
* Blog post: https://www.linkedin.com/pulse/more-efficient-finetuning-implementing-lora-from-scratch-george-davis/
* Main repo: https://github.com/davisgcii/basicLoRA
* Another good repo: https://github.com/sunildkumar/lora_from_scratch

### Part 3: PEFT and Quantized LLM Fine-tuning
* PEFT: https://huggingface.co/blog/peft
* Datset used: https://huggingface.co/datasets/bitext/Bitext-travel-llm-chatbot-training-dataset
* Supervised Fine-tuning Trainer: https://huggingface.co/docs/trl/main/en/sft_trainer#training-adapters