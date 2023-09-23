
// extern crate blas_src;

mod llama2;

use std::fs::File;
use std::io::{self, Write, BufReader, BufWriter};
use std::time::Instant;

use clap::{Parser, ValueEnum};

use llama2::*;

// Quantization papers:
//   * SmoothQuant - https://arxiv.org/pdf/2211.10438.pdf
//   * AWQ - https://arxiv.org/pdf/2306.00978.pdf
//   * The case for 4 bit precision - https://arxiv.org/pdf/2212.09720.pdf


#[macro_use]
extern crate num_derive;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(value_enum)]
    mode: Mode,

    model_file: String,

    #[arg(short, long, default_value = "0.9")]
    temperature: f32,
    #[arg(short, long, default_value = "256")]
    steps: usize,
    #[arg(short, long)]
    prompt: Option<String>,
    #[arg(short, long)]
    system_prompt: Option<String>,

    /// Write the quantized model to a file, much faster to load on subsequent runs
    #[arg(short, long)]
    write_model: Option<String>,
    /// Forces quantization on-the-fly for f32 models
    #[arg(short, long, default_value = "false")]
    force_quantize: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Mode {
    Generate,
    Chat,
}

fn main() {
    let args = Args::parse();

    // Initialize config from file
    let mut file = File::open(&args.model_file)
        .expect("Unable to open the checkpoint file");
    let mut config = Config::read(&mut file)
        .expect("Failed to read the config");

    // Finish reading the checkpoint file by loading weights, at the moment we
    // quantize on the fly
    let start = Instant::now();
    println!("Reading model weights, takes a little while...");
    let mut reader = BufReader::with_capacity(1000*1000, file);
    let weights = Weights::read(&config, args.force_quantize, &mut reader);
    if args.force_quantize {
        config.q_type = QuantizationType::LinearI8;
        config.q_stride = DEFAULT_STRIDE;
    }
    println!("Read model weights in {:.2}s.", start.elapsed().as_secs_f64());

    if args.write_model.is_some() {
        println!("Writing model to {}", args.write_model.as_ref().unwrap());
        let mut writer = File::create(args.write_model.unwrap())
            .expect("Unable to create the output file");
        let mut buf_writer = BufWriter::new(&mut writer);
        config.write(&mut buf_writer).expect("Failed to write the config");
        weights.write(&mut buf_writer).expect("Failed to write the weights");
        println!("Done");
    }

    let steps = args.steps.max(1).min(config.seq_len);

    let tok = Tokenizer::read("./models/tokenizer.bin", config.vocab_size)
        .expect("Failed to load tokenizer");

    let mut trans = Transformer::new(config, weights);
    let sampler = Sampler::new(args.temperature);

    match args.mode {
        Mode::Generate =>
            generate(&mut trans, &tok, &sampler, steps, args.prompt),
        Mode::Chat =>
            chat(&mut trans, &tok, &sampler, steps, args.prompt, args.system_prompt),
    }
}

fn chat(
    transformer: &mut Transformer,
    tokenizer: &Tokenizer,
    sampler: &Sampler,
    steps: usize,
    cli_user_prompt: Option<String>,
    cli_system_prompt: Option<String>)
{
    let system_prompt = cli_system_prompt.unwrap_or("".to_string());
    let mut user_prompt = cli_user_prompt.unwrap_or("".to_string());
    let mut rendered_prompt;
    let mut prompt_tokens: Vec<usize> = vec![];

    let (mut user_turn, mut user_idx) = (true, 0);
    let (mut next, mut token, mut pos) = (0, 0, 0);

    while pos < steps {
        if user_turn {
            if pos > 0 {
                user_prompt.clear();
            }
            if user_prompt.is_empty() {
                read_stdin("User: ", &mut user_prompt);
            }

            if pos == 0 && !system_prompt.is_empty() {
                rendered_prompt = format!("[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]",
					system_prompt, user_prompt);
            } else {
                rendered_prompt = format!("[INST] {} [/INST]", user_prompt);
            }

            prompt_tokens = tokenizer.encode(&rendered_prompt, true, false);
            user_idx = 0;
            user_turn = false;
            print!("Assistant: ");
        }

        if user_idx < prompt_tokens.len() {
            token = prompt_tokens[user_idx];
            user_idx += 1;
        } else {
            token = next;
        }

        if token == 2 {
            user_turn = true;
        }

        let logits = transformer.forward(token, pos);
        next = sampler.sample(logits).expect("No logits");
        pos += 1;

        if user_idx >= prompt_tokens.len() && next != 2 {
            let piece = tokenizer.decode(token, next);
            print!("{}", piece);
            std::io::stdout().flush().unwrap();
        }

        if next == 2 {
            println!();
        }
    }
    println!();
}

pub fn generate(
    transformer: &mut Transformer,
    tok: &Tokenizer,
    sampler: &Sampler,
    steps: usize,
    prompt: Option<String>)
{
    let prompt = prompt.unwrap_or_else(|| "".to_string());
    let prompt_toks = tok.encode(&prompt, true, false);

    let start = Instant::now();
    let mut token = prompt_toks[0];
    println!("<s>");

    for pos in 0..steps {
        let logits = transformer.forward(token, pos);

        let next = if pos < prompt_toks.len() - 1 {
           prompt_toks[pos + 1]
        } else {
            sampler.sample(logits).expect("No logits")
        };

        print!("{}", tok.decode(token, next));
        io::stdout().flush().unwrap();
        token = next;
    }
    println!("\nachieved tok/s: {}", steps as f64 / start.elapsed().as_secs_f64());
}

fn read_stdin(prompt: &str, buffer: &mut String) {
    print!("{}", prompt);
    std::io::stdout().flush().unwrap();
    std::io::stdin().read_line(buffer).unwrap();
    buffer.truncate(buffer.trim_end().len());
}
