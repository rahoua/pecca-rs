## pecca-rs

Pecca is starting as a Rust port of the excellent [@karpathy](https://github.com/karpathy) [llama2.c](https://github.com/karpathy/llama2.), itself a minimalistic adaptation of [llama.cpp](https://github.com/ggerganov/llama.cpp).

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
cargo run --release ./models/stories/stories15M.bin
```

## Direction

A list of possible future developments for the project:

* Bucket quantization to improve accuracy.
* Make quantization optional with a feature flag.
* Improved tokenizer.
* Speed up loading of large models, likely mostly by saving them quantized.
* Cleaner command-line parsing ([clap](https://docs.rs/clap/latest/clap/))
* Inference performance and general memory footprint during inference.
* Explore extending ndarray `dot` operation to support cublas or Metal.
* Additional parallelization of independent operations.
* Various refactoring.
* Support for additional models.
