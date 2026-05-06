CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-Embedding-0.6B --max_model_len 3000 --runner pooling --host 0.0.0.0 --port 8201 --gpu-memory-utilization 0.8

