# vLLM 部署全量常用参数分类整理
## 一、基础启动参数（服务入口、模型路径）
```bash
# 核心必带
--model /path/to/model               # 模型本地路径/HuggingFace repo id
--served-model-name qwen3-8b         # 对外暴露的模型名，API调用用
--trust-remote-code                  # 加载自定义模型代码必须开
--download-dir ./hf_cache            # 模型缓存目录
--revision main                      # 指定模型分支/版本
--tokenizer /path/to/tokenizer       # 单独指定分词器路径
```

## 二、显存 & 量化相关（最常用）
### 1. 显存分配
```bash
--gpu-memory-utilization 0.9         # GPU显存占用上限0~1，0.9即90%
--max-num-gpu 4                      # 最大使用GPU数量
--tensor-parallel-size 2             # TP张量并行，多卡均分模型权重
--pipeline-parallel-size 1           # PP流水线并行，分层切分（大模型）
```

### 2. 量化加载（低显存必备）
```bash
# 4bit/8bit GPTQ/AWQ
--quantization gptq                  # 量化类型：gptq/awq/squeezellm
--load-format gptq
--wbits 4                            # 量化位宽4/8
--group-size 128

# bitsandbytes 4/8bit
--load-in-8bit
--load-in-4bit

# FP8 量化（支持H100/A100/RTX40系/A5000）
--quantization fp8
--fp8-kv-cache

# KV Cache量化（大幅省显存）
--kv-cache-dtype fp8_e5m2
--enable-prefix-caching              # 前缀缓存，多用户共享KV
```

## 三、推理并发 & 上下文窗口（性能核心）
```bash
--max-model-len 32768                # 模型最大上下文长度
--max-num-seqs 256                   # 同时并行处理的请求数
--max-batch-size-to-trigger-evict 128
--enforce-eager                      # 禁用CUDA graph，兼容性提升
--disable-cuda-graph                 # 关闭CUDA图，适合动态长度场景
--enable-prefix-caching              # 共享前缀KV缓存，提升多轮对话速度
--prefix-cache-nblocks 1024          # 前缀缓存块大小
```

## 四、KV Cache 高级调参
```bash
--block-size 16                      # KV缓存块token粒度，16/32常用
--num-gpu-blocks-override 8192       # 手动指定KV缓存总块数（不自动计算显存）
--swap-space 16                      # CPU交换空间GB，显存不足时缓存KV到内存
--gpu-memory-utilization 0.85        # 预留显存给KV缓存
```

## 五、API 服务参数（OpenAI兼容接口）
```bash
--host 0.0.0.0                       # 监听地址，外网访问必须0.0.0.0
--port 8000                          # 服务端口
--api-key sk-vllm123                 # 接口鉴权密钥
--uvicorn-log-level warning          # 服务日志等级
--ssl-keyfile ./key.pem              # HTTPS证书
--ssl-certfile ./cert.pem
--allow-credentials
```

## 六、采样生成全局默认参数（可被API覆盖）
```bash
--temperature 0.7
--top-p 0.9
--top-k 40
--max-tokens 1024                    # 默认单次生成最大token
--stop "<|endoftext|>", "</s>"       # 全局停止符
--logprobs 5                         # 返回top5 token概率
--echo                               # 返回输入prompt
```

## 七、性能优化参数
```bash
# 连续批处理优化
--continuous-batching
--max-num-batched-tokens 16384       # 单批最大token总数，控显存峰值

# FlashInfer 加速（新版本vLLM）
--enable-flashinfer
--flashinfer-attention

# PagedAttention基础（默认开启）
--disable-paged-attention            # 关闭不推荐，性能暴跌

# 异步调度（v0.4+）
--disable-async-output-proc
```

## 八、分布式多卡/多机部署
```bash
# 单机多卡TP
--tensor-parallel-size 2

# 多机分布式
--distributed-executor-backend ray
--ray-address auto
--pp-size 2
--tp-size 2
```

## 九、日志、调试、监控
```bash
--log-level INFO                     # DEBUG/INFO/WARNING/ERROR
--log-file ./vllm.log
--print-stats                        # 定时输出吞吐、显存、队列统计
--stats-interval 5                   # 统计打印间隔秒
--disable-log-requests               # 不打印每条请求日志，减少IO
--enable-metrics                    # 开启Prometheus监控
--metrics-port 8001
```

## 十、特殊功能参数
### 1. 工具调用 / 函数调用
```bash
--enable-auto-tool-choice
--tool-call-parser hermes
```

### 2. 多模态（LLaVA、Qwen-VL等）
```bash
--model llava-v1.5-7b
--mm-processor-kwargs '{"image_size":336}'
--image-token-size 576
```

### 3. 嵌入模型部署
```bash
--task embed
--trust-remote-code
```

### 4. 推理模式（解码/编码）
```bash
--task generate      # 默认文本生成
--task embed         # 向量嵌入
--task score         # 打分模型
```

## 十一、完整启动示例（单卡Qwen3-8B 4bit量化）
```bash
python -m vllm.entrypoints.openai.api_server \
--model Qwen/Qwen3-8B \
--served-model-name qwen3-8b \
--host 0.0.0.0 \
--port 8000 \
--gpu-memory-utilization 0.9 \
--max-model-len 32768 \
--quantization gptq \
--wbits 4 \
--group-size 128 \
--enable-prefix-caching \
--max-num-seqs 128 \
--trust-remote-code \
--api-key sk-123456
```

## 生产环境常用
1. 显存不足：降低 `gpu-memory-utilization`、开启 `swap-space`、`fp8 kv cache`、4bit量化
2. 并发低：调大 `max-num-seqs`、`max-num-batched-tokens`、开启 `prefix-caching`
3. 多卡大模型：`tensor-parallel-size` 按GPU数量均分
4. 接口外网访问：`--host 0.0.0.0` + `--api-key` 鉴权
5. 长文本：调高 `max-model-len`，配合FP8 KV节省显存
