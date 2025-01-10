# Umbrella

Best practice for deploying LLMs for single users.

## Models and Benchmarks

### MT Bench

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
      <!-- 将 GPU 这一格设置为跨两行 -->
      <td rowspan="3">RTX 4090</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <!-- 这里留空或者填入第二行需要的内容 -->
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <!-- 这里留空或者填入第二行需要的内容 -->
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <!-- 将 GPU 这一格设置为跨两行 -->
      <td rowspan="2">RTX 4080 SUPER</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <!-- 这里留空或者填入第二行需要的内容 -->
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <!-- 将 GPU 这一格设置为跨两行 -->
      <td rowspan="2">RTX 4070 Ti</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <!-- 这里留空或者填入第二行需要的内容 -->
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <!-- 将 GPU 这一格设置为跨两行 -->
      <td rowspan="2">L40</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <!-- 这里留空或者填入第二行需要的内容 -->
      <td></td>
      <td></td>
      <td></td>
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
      <!-- 将 GPU 这一格设置为跨两行 -->
      <td rowspan="3">RTX 4090</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <!-- 这里留空或者填入第二行需要的内容 -->
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <!-- 这里留空或者填入第二行需要的内容 -->
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <!-- 将 GPU 这一格设置为跨两行 -->
      <td rowspan="2">RTX 4080 SUPER</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <!-- 这里留空或者填入第二行需要的内容 -->
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <!-- 将 GPU 这一格设置为跨两行 -->
      <td rowspan="2">RTX 4070 Ti</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <!-- 这里留空或者填入第二行需要的内容 -->
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <!-- 将 GPU 这一格设置为跨两行 -->
      <td rowspan="2">L40</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <!-- 这里留空或者填入第二行需要的内容 -->
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>


## Deploying your LLMs with Umbrella

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