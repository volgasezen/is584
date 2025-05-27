## Instructions

Inside this repository there are functions for building the vector database, asking the RAG model a question and to evaluate it's generation with the RAG triad over 14 example questions using trulens.

To interact with LLM's we will use llamafiles (LLM servers in single files) from [this mozilla repository](https://github.com/Mozilla-Ocho/llamafile?tab=readme-ov-file#other-example-llamafiles). You can download a llamafile and run it in mac, linux or windows by following the instructions in the repository. This will host a server where you can send requests via an API or python libraries. While we can run inference on custom weights using other llama.cpp implementations, or utilities like Ollama, this will be a simple way to run quick inference even on CPU's. 

It is recommended to create a new environment and install the necessary packages with the following command: (some versions could be subject to change)

```console
pip install -r requirements.txt
```