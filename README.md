# Comparative study of function calling

## Requirements

```bash
docker model pull ai/qwen2.5:latest
docker model pull ai/qwen3:latest
docker model pull ai/qwen3:0.6B-Q4_K_M
docker model pull ai/gemma3n:latest
docker model pull hf.co/salesforce/llama-xlam-2-8b-fc-r-gguf:q4_k_m
docker model pull hf.co/salesforce/llama-xlam-2-8b-fc-r-gguf:q2_k

docker model pull hf.co/salesforce/xlam-2-3b-fc-r-gguf:q4_k_m
docker model pull hf.co/salesforce/xlam-2-3b-fc-r-gguf:q4_k_s
docker model pull hf.co/salesforce/xlam-2-3b-fc-r-gguf:q4_0
docker model pull hf.co/salesforce/xlam-2-3b-fc-r-gguf:q3_k_l
```


```bash
ollama qwen2.5:latest
ollama pull qwen3:8b
ollama pull qwen3:0.6b
```

The tests are using `ParallelToolCalls: openai.Bool(true)`