<div style="text-align: center;">
# UMbreLLa

The best practice for deploying LLMs tailored to single-user scenarios. Using UMbreLLa, 70B-level models can achieve performance comparable to human reading speed on an RTX 4070Ti, delivering exceptional efficiency and responsiveness.

<img src="assets/umbrella.jpeg" width="200" align="top"/>
<b>Offloading+ Speculative Decoding + Quantization</b>
</div>

## Models and Benchmarks

The throughput is measured with a batch size of 1 to directly mirror the user experience.

### MT Bench
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




### Code Completion
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


## Deploying your LLMs with UMbreLLa

### CLI Chatbot

### API Server/Client

### Gradio Chatbot


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