## pecca-rs

Pecca is starting as a Rust port of the excellent [@karpathy](https://github.com/karpathy) [llama2.c](https://github.com/karpathy/llama2.c), itself a minimalistic adaptation of [llama.cpp](https://github.com/ggerganov/llama.cpp).

Compared to other Rust ports, Pecca leverages [ndarray](https://github.com/rust-ndarray/ndarray), which has several advantages:

* Type Safety: all matrices have proper dimensions (instead of giant flat arrays) and most operations will check dimensions compatibility.
* Speed: out of the box and single-threaded, Pecca is already slightly faster than the C version.
* Readability: matrix operations can be written succinctly. This first version of pecca-rs is only 425 lines, including comments.

Going forward, Pecca will leverage Rust and its ecosystem whenever it makes sense, rather than attempting to avoid dependencies above all (like llama.cpp).

## Usage

```bash
git clone https://github.com/rahoua/pecca-rs.git
cd pecca-rs
wget -P ./models/stories/  https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
cargo run --release generate ./models/stories/stories15M.bin
```

Pecca can be run similarly with larger tiny stories models (like the 110M one) or the llama2 models (only 7B recommended so far). For a full list of command line options run:

```bash
pecca-rs --help
```

To get the llama2 models, follow the [instructions](https://github.com/karpathy/llama2.c#metas-llama-2-models) for llama2.c. Pecca supports the same model format. As Pecca does not use memmap, loading and quantizing the model on the fly can take some time. To speed things up, the models can also be saved quantized using the `--write-model` command line switch.

## Performance

At the moment there's no formal benchmark, we just provide rough estimates to give a ballpark of overall performance.

Llama2 model on a Macbook Pro M2 Max:
* llama2.c: 4 tok/s
* llama.cpp, Q4KM quantization: 24 tok/s
* pecca, i8 quantization: 11 tok/s

## Future Directions

A list of possible future developments for the project:

* Evaluation mode that runs both quantized and f32 inference at the same time to quantify quantization error.
* Improved tokenizer.
* Special mode for llama2 code (slight differences with other llama2s).
* Inference performance and general memory footprint during inference.
* Experiment with [SmoothQuant](https://arxiv.org/abs/2211.10438)
* Explore extending ndarray `dot` operation to support cublas or Metal.
* Additional parallelization of independent operations.
* Various refactoring.
* Support for additional models.
