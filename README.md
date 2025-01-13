
<div align="center">
<h1><img src="assets/umbrella.jpeg" height="50px" align="top"/> UMbreLLa: Deploying LLMs <br>for Single-User Scenarios 
</h1>
</div>

<div align="center">
<i>Using UMbreLLa, 70B-level models can achieve performance comparable to human reading speed on an RTX 4070Ti, delivering exceptional efficiency and responsiveness.</i>
</div>

## 1. Models Supported and Benchmarks

The throughput is measured with a batch size of 1 to directly mirror the user experience.

### 1.1 MT Bench
<table>
  <thead>
    <tr>
      <th>GPU</th>
      <th>Model</th>
      <th>Draft</th>
      <th colspan="2">Throughput (tokens/sec)</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th>Stochastic</th>
      <th>Greedy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">RTX 4090</td>
      <td><a href="https://huggingface.co/hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4">Llama3.1-70B-Instruct-AWQ</td>
      <td><a href="https://huggingface.co/hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4">Llama3.1-8B-Instruct-AWQ</td>
      <td>7.2</td>
      <td>8.6</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/casperhansen/llama-3.3-70b-instruct-awq">Llama3.3-70B-Instruct-AWQ</td>
      <td><a href="https://huggingface.co/hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4">Llama3.1-8B-Instruct-AWQ</td>
      <td>7.0</td>
      <td>7.4</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct">Llama3.1-8B-Instruct</td>
      <td><a href="https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct">Llama3.2-1B-Instruct</td>
      <td>100.7</td>
      <td>108.1</td>
    </tr>
    <tr>
      <td rowspan="2">RTX 4080 SUPER</td>
      <td><a href="https://huggingface.co/hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4">Llama3.1-70B-Instruct-AWQ</td>
      <td><a href="https://huggingface.co/hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4">Llama3.1-8B-Instruct-AWQ</td>
      <td>7.4</td>
      <td>8.4</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/casperhansen/llama-3.3-70b-instruct-awq">Llama3.3-70B-Instruct-AWQ</td>
      <td><a href="https://huggingface.co/hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4">Llama3.1-8B-Instruct-AWQ</td>
      <td>6.7</td>
      <td>7.2</td>
    </tr>
    <tr>
      <td rowspan="2">RTX 4070 Ti</td>
      <td><a href="https://huggingface.co/hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4">Llama3.1-70B-Instruct-AWQ</td>
      <td><a href="https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct">Llama3.2-1B-Instruct</td>
      <td>5.5</td>
      <td>6.1</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/casperhansen/llama-3.3-70b-instruct-awq">Llama3.3-70B-Instruct-AWQ</td>
      <td><a href="https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct">Llama3.2-1B-Instruct</td>
      <td>5.2</td>
      <td>5.5</td>
    </tr>
    <tr>
      <td rowspan="2">L40</td>
      <td><a href="https://huggingface.co/hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4">Llama3.1-70B-Instruct-AWQ</td>
      <td><a href="https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct">Llama3.2-1B-Instruct</td>
      <td>37.0</td>
      <td>38.5</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/casperhansen/llama-3.3-70b-instruct-awq">Llama3.3-70B-Instruct-AWQ</td>
      <td><a href="https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct">Llama3.2-1B-Instruct</td>
      <td>36.3</td>
      <td>37.1</td>
    </tr>
  </tbody>
</table>




### 1.2 Code Completion
<table>
  <thead>
    <tr>
      <th>GPU</th>
      <th>Model</th>
      <th>Draft</th>
      <th>Throughput (tokens/sec)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">RTX 4090</td>
      <td><a href="https://huggingface.co/hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4">Llama3.1-70B-Instruct-AWQ</td>
      <td><a href="https://huggingface.co/hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4">Llama3.1-8B-Instruct-AWQ</td>
      <td>11.4</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/casperhansen/llama-3.3-70b-instruct-awq">Llama3.3-70B-Instruct-AWQ</td>
      <td><a href="https://huggingface.co/hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4">Llama3.1-8B-Instruct-AWQ</td>
      <td>11.2</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct">Llama3.1-8B-Instruct</td>
      <td><a href="https://huggingface.co/InfiniAILab/CodeDrafter-500M">CodeDrafter-500M</td>
      <td>174.8</td>
    </tr>
    <tr>
      <td rowspan="3">RTX 4080 SUPER</td>
      <td><a href="https://huggingface.co/hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4">Llama3.1-70B-Instruct-AWQ</td>
      <td><a href="https://huggingface.co/hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4">Llama3.1-8B-Instruct-AWQ</td>
      <td>12.2</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/casperhansen/llama-3.3-70b-instruct-awq">Llama3.3-70B-Instruct-AWQ</td>
      <td><a href="https://huggingface.co/hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4">Llama3.1-8B-Instruct-AWQ</td>
      <td>12.1</td>
    </tr>
     <tr>
      <td><a href="https://huggingface.co/hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4">Llama3.1-8B-Instruct-AWQ</td>
      <td><a href="https://huggingface.co/InfiniAILab/CodeDrafter-500M">CodeDrafter-500M</td>
      <td>195.3</td>
    </tr>
    <tr>
      <td rowspan="3">RTX 4070 Ti</td>
      <td><a href="https://huggingface.co/hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4">Llama3.1-70B-Instruct-AWQ</td>
      <td><a href="https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct">Llama3.2-1B-Instruct</td>
      <td>9.7</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/casperhansen/llama-3.3-70b-instruct-awq">Llama3.3-70B-Instruct-AWQ</td>
      <td><a href="https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct">Llama3.2-1B-Instruct</td>
      <td>9.6</td>
    </tr>
     <tr>
      <td><a href="https://huggingface.co/hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4">Llama3.1-8B-Instruct-AWQ</td>
      <td><a href="https://huggingface.co/InfiniAILab/CodeDrafter-500M">CodeDrafter-500M</td>
      <td>162.3</td>
    </tr>
    <tr>
      <td rowspan="2">L40</td>
      <td><a href="https://huggingface.co/hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4">Llama3.1-70B-Instruct-AWQ</td>
      <td><a href="https://huggingface.co/InfiniAILab/CodeDrafter-500M">CodeDrafter-500M</td>
      <td>45.6</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/casperhansen/llama-3.3-70b-instruct-awq">Llama3.3-70B-Instruct-AWQ</td>
      <td><a href="https://huggingface.co/InfiniAILab/CodeDrafter-500M">CodeDrafter-500M</td>
      <td>45.0</td>
    </tr>
  </tbody>
</table>

*Offloading experiments heavily rely on the status of PCIE, and may vary across instances.*

## 2 Deploying your LLMs with UMbreLLa

### 2.1 Install
```bash
conda create -n umbrella python=3.10
bash install.sh
```
### 2.2 CLI Chatbot
```bash
cd app
python chatbot.py --configuration ../configs/chat_config_24gb.json
```

Then you can chat with the LLM specified in `chat_config_24gb.json`.

### 2.3 Gradio Chatbot
```bash
cd app
python gradio_chat.py --configuration ../configs/chat_config_24gb.json
```

Then you can chat with the LLM specified in `chat_config_24gb.json` in Gradio.

### 2.4 API Server/Client
#### 2.4.1 Server
```bash
cd app
python api.py --configuration ../configs/chat_config_24gb.json --max_client 1 --port 65432
```

`configuration` specifies the LLM and speculative decoding details.

`max_client` is the maximum clients that can connect to the server.

`port` is the port of the server.

#### 2.4.2 Client
After the server is started, Client can be started and connect to the server by
```python
from umbrella.api.client import APIClient
client = APIClient(port=port) #port should be the same as the server
client.run()
```

To get the LLM output,
```python
input1 = {"context": text1, "max_new_tokens": 512, "temperature": 0.0}
output1 = client.get_output(**input1)
```
## 3 Config the LLM Engine
```json
{
    "model": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4", 
    "draft_model": "meta-llama/Llama-3.2-1B-Instruct",
    "offload": true,
    "cuda_graph": false,
    "max_length": 4096,
    "num_cache_layers": 0,
    "generation_length": 256,
    "max_turns": 12,
    "topk": 32,
    "temperature": 0.6,
    "topp": 0.9,
    "repetition_penalty": 1.05,
    "growmap_path": "../umbrella/trees/sequoia_tree-3x4.json",
    "width": 16,
    "num_beams": 24,
    "depth": 16,
    "engine": "dynamic",
    "template": "meta-llama3"
}
```
`model`: Target LLM to serve.
`draft_model`: Draft model.
`offload`: Whether offload the target model.
`cuda_graph`: Whether apply cuda graph for draft model (AWQ not supported now).
`engine`: static (for on device) or dynamic (for offloading)
`width`, `num_beams`,`depth`: Speculative decoding hyper-parameters for dynamic engine.
`growmap_path`: Speculative decoding tree for static engine.
`template`: Template for prompts. Currently we support `llama3-code` and `meta-llama3`.
Other parameters are generation arguments.

## Reference
```bibtex
@article{chen2024sequoia,
  title={Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding},
  author={Chen, Zhuoming and May, Avner and Svirschevski, Ruslan and Huang, Yuhsun and Ryabinin, Max and Jia, Zhihao and Chen, Beidi},
  journal={arXiv preprint arXiv:2402.12374},
  year={2024}
}
@article{svirschevski2024specexec,
  title={SpecExec: Massively Parallel Speculative Decoding for Interactive LLM Inference on Consumer Devices},
  author={Svirschevski, Ruslan and May, Avner and Chen, Zhuoming and Chen, Beidi and Jia, Zhihao and Ryabinin, Max},
  journal={arXiv preprint arXiv:2406.02532},
  year={2024}
}
```