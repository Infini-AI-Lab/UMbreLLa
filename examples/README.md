# Examples of UMbreLLa

### 1 Benchmark the decoding/verification speed

```bash
    python bench.py --model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 --offload --D 1 --T 20 
```

<h4>Key Configuration Options</h4>
<ul>
    <li><strong>model</strong>: Specifies the target LLM to serve, e.g., <code>"hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"</code>.</li>
    <li><strong>offload</strong>: Enables offloading of the target model to host memory.</li>
    <li><strong>cuda_graph</strong>: Toggles CUDA graph optimization for the draft model (currently unsupported for AWQ models).</li>
    <li><strong>M</strong>: The maximum token length for input and output combined.</li>
    <li><strong>D</strong>: The number of tokens for one decoding steps (for testing verification).</li>
    <li><strong>T</strong>: Repeated times in benchmarking.</li>
</ul>

### 2 Benchmarking auto-regressive generation

```bash
    python generate.py --model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 --offload
```

<h4>Key Configuration Options</h4>
<ul>
    <li><strong>model</strong>: Specifies the target LLM to serve, e.g., <code>"hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"</code>.</li>
    <li><strong>offload</strong>: Enables offloading of the target model to host memory.</li>
    <li><strong>cuda_graph</strong>: Toggles CUDA graph optimization for the draft model (currently unsupported for AWQ models).</li>
    <li><strong>G</strong>: The maximum generated tokens (smaller than 2000).</li>
    <li><strong>template</strong>: Defines the structure for input prompts. Supported values include:
            <ul>
                <li><code>"llama3-code"</code>: Optimized for code-related tasks.</li>
                <li><code>"meta-llama3"</code>: General-purpose instruction-following template.</li>
            </ul>
        </li>
</ul>

### 3 Speculative Decoding Example

```bash
    python spec_generate.py --model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 --offload
```

<h4>Key Configuration Options</h4>
<ul>
    <li><strong>model</strong>: Specifies the target LLM to serve, e.g., <code>"hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"</code>.</li>
    <li><strong>draft_model</strong>: Lightweight draft model, e.g., <code>"meta-llama/Llama-3.2-1B-Instruct"</code>.</li>
    <li><strong>offload</strong>: Enables offloading of the target model to host memory.</li>
    <li><strong>cuda_graph</strong>: Toggles CUDA graph optimization for the draft model (currently unsupported for AWQ models).</li>
    <li><strong>G</strong>: The maximum generated tokens (smaller than 2000).</li>
    <li><strong>template</strong>: Defines the structure for input prompts. Supported values include:
            <ul>
                <li><code>"llama3-code"</code>: Optimized for code-related tasks.</li>
                <li><code>"meta-llama3"</code>: General-purpose instruction-following template.</li>
            </ul>
        </li>
</ul>

### 4 Benchmarking Speculative Decoding

```bash
python spec_bench.py --configuration ../configs/chat_config_24gb.json #MT Bench
python spec_bench_python.py --configuration ../configs/chat_config_24gb.json #Code Completion
```

### 5 Generate Sequoia Tree

```bash
python construct_sequoia.py --w 5 --d 6
```

<h4>Key Configuration Options</h4>
<ul>
    <li><strong>model</strong>: Specifies the target LLM to serve, e.g., <code>"hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"</code>.</li>
    <li><strong>draft_model</strong>: Lightweight draft model, e.g., <code>"meta-llama/Llama-3.2-1B-Instruct"</code>.</li>
    <li><strong>offload</strong>: Enables offloading of the target model to host memory.</li>
    <li><strong>cuda_graph</strong>: Toggles CUDA graph optimization for the draft model (currently unsupported for AWQ models).</li>
    <li><strong>w</strong>: The width of the Sequoia trees.</li>
    <li><strong>d</strong>: The depth of the Sequoia trees.</li>
    <li><strong>dst</strong>: The json file which saves Sequoia tree, and can be specified as a growmap_path in static speculation engine.</li>
</ul>
