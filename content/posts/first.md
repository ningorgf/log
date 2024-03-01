+++
title = 'First Sample'
date = 2024-02-22T15:23:51+08:00
+++

# Hugging face demo 
Pipeline for a chatbot based on my logs
## get the log files
colab test 

## Hugging face space

Visit https://huggingface.co/new-space and fill in the form.

## LangChain Bug log

Chatbot: LangChain -> generates prompt and process -> LLM -> Generates response.
In principle, for blogs and these short contests, it is sufficient to directly generate a prompt+query using LLM. Using LangChain system that is still being developed, is overly complicated.

```
from langchain_community import embeddings
Chroma.from_documents(documents=doc_splits,collection_name="rag-chroma",embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'),)
```
This line of code can easily generate the bugs in colab and hugging face spaces
```
ValueError: Error raised by inference endpoint: HTTPConnectionPool(host='localhost', port=11434): Max retries exceeded with url: /api/embeddings (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7fd7001d4520>: Failed to establish a new connection: [Errno 111] Connection refused'))
```
Possible reason: Langchain will set the base_url as 'localhost:11434' as the server when creating embeddings.


The method I used: Download Llama 2 and run it locally.

The official LangChain documentation introduces the use of llama.cpp (https://python.langchain.com/docs/integrations/llms/llamacpp) But don't forget to download the model.
https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.llamacpp.LlamaCppEmbeddings.html

The one I use in hf is llama-2-7b.gguf.q4_0.bin (Downloaded q4_0 from https://huggingface.co/TheBloke/Llama-2-7B-GGML and then converted to gguf using convert-llama-ggml-to-gguf.py)

If you want to use this chatbot, you need to spend a long time before you can get feedback, so please be patient if there is no error running.
References:

```
llama_print_timings:        load time =    4325.45 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time =    5808.80 ms /     8 tokens (  726.10 ms per token,     1.38 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time =    5810.46 ms /     9 tokens
WARNING:chromadb.segment.impl.vector.local_hnsw:Number of requested results 4 is greater than number of elements in index 1, updating n_results = 1

llama_print_timings:        load time =    4365.77 ms
llama_print_timings:      sample time =     163.78 ms /   256 runs   (    0.64 ms per token,  1563.04 tokens per second)
llama_print_timings: prompt eval time =  171864.19 ms /   292 tokens (  588.58 ms per token,     1.70 tokens per second)
llama_print_timings:        eval time =  201403.05 ms /   255 runs   (  789.82 ms per token,     1.27 tokens per second)
llama_print_timings:       total time =  374587.81 ms /   547 tokens
```


Wait until I finish updating my notes on the course (TinyML and Efficient Deep Learning Computing 6.5940 • Fall • 2023, [hanlab link](https://hanlab.mit.edu/courses/2023-fall-65940)) before making a new quantized acceleration model (if I still remember).







