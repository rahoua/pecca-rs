
mod llama2;

use clap::Parser;

// Quantization papers:
//   * SmoothQuant - https://arxiv.org/pdf/2211.10438.pdf
//   * AWQ - https://arxiv.org/pdf/2306.00978.pdf
//   * The case for 4 bit precision - https://arxiv.org/pdf/2212.09720.pdf

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    model_file: String,

    #[arg(short, long, default_value = "0.9")]
    temperature: f32,
    #[arg(short, long, default_value = "256")]
    steps: usize,
    #[arg(short, long)]
    prompt: Option<String>,
}

fn main() {
    let args = Args::parse();
    llama2::gen(args.model_file, args.temperature, args.steps, args.prompt);
}
