
mod llama2;

// Quantization papers:
//   * SmoothQuant - https://arxiv.org/pdf/2211.10438.pdf
//   * AWQ - https://arxiv.org/pdf/2306.00978.pdf
//   * The case for 4 bit precision - https://arxiv.org/pdf/2212.09720.pdf

fn main() {
  llama2::gen();
}
