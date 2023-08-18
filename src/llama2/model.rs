
use std::fs::File;
use std::io::{self, Read, BufReader};

use ndarray::prelude::*;
use ndarray::StrideShape;

use byteorder::{ReadBytesExt, LittleEndian};
use num_traits::cast::AsPrimitive;

// Weights type, placeholder for quantization
pub type WTy = f32;

// The original format has 7 serialized fields of 4 bytes each
const CONFIG_SIZE: usize = 7*4;

// Transformer configuration. Doesn't directly affect the model as it's based on llama2
// but allows variations in dimensions, number of layers, etc.
#[derive(Default, Debug)]
pub struct Config {
    pub dim: usize, // transformer dimension
    pub hidden_dim: usize, // for ffn layers
    pub n_layers: usize, // number of layers
    pub n_heads: usize, // number of query heads
    pub n_kv_heads: usize, // number of key/value heads (can be < query heads because of multiquery)
    pub vocab_size: usize, // vocabulary size, usually 256 (byte-level)
    pub seq_len: usize, // max sequence length
    pub shared_weights: bool,
}

impl Config {
    pub fn from_file(file: &mut File) -> io::Result<Self> {
        // the C code uses int, which is 4 bytes, we use usize because it's cleaner
        // down the road but requires a little size arithmetic
        let mut config_bytes = vec![0; CONFIG_SIZE];
        file.read_exact(&mut config_bytes)?;

        let mut config_iter = config_bytes.chunks_exact(4);
        let mut read_usize = || -> usize {
            let bytes = config_iter.next().unwrap();
            u32::from_ne_bytes(bytes.try_into().unwrap()) as usize
        };

        let mut config = Config { 
            dim: read_usize(), 
            hidden_dim: read_usize(),
            n_layers: read_usize(),
            n_heads: read_usize(),
            n_kv_heads: read_usize(),
            vocab_size: read_usize(),
            seq_len: read_usize(),
            shared_weights: true,
        };
        // a little whackiness in the format
        if config.vocab_size >= 2_usize.pow(31) {
            config.vocab_size = (!(config.vocab_size as u32) + 1) as usize;
            config.shared_weights = false;
        }
        Ok(config)
    }

    pub fn head_size(&self) -> usize {
        self.dim / self.n_heads
    }
}

// All weights in the Llama2 model. Read in bulk from a binary file.
pub struct Weights {
    pub tet: Array2<WTy>, // (vocab_size, dim) token embeddings table
    pub rms_att_weight: Array2<WTy>, // (n_layers, dim) rmsnorm weights
    pub rms_ffn_weight: Array2<WTy>, // (n_layers, dim)
    pub wq: Array3<WTy>, // (n_layers, dim, (n_heads * head_size))
    pub wk: Array3<WTy>, // (n_layers, dim, (n_kv_heads * head_size))
    pub wv: Array3<WTy>, // (n_layers, dim, (n_kv_heads * head_size))
    pub wo: Array3<WTy>, // (layer, (n_heads * head_size), dim)
    pub w1: Array3<WTy>, // (n_layers, hidden_dim, dim)
    pub w2: Array3<WTy>, // (n_layers, dim, hidden_dim)
    pub w3: Array3<WTy>, // (n_layers, hidden_dim, dim)
    pub rms_final_weight: Array1<WTy>, // (dim)
    pub wcls: Array2<WTy>, // (dim, vocab_size) optional, classifier weights for logits, on the last layer
}

// Utility function to read f32s from file into an ndarray of the provided dimension
fn read_f32<R, S, D>(buf: &mut BufReader<R>, shape: S) -> Array<WTy, D>
where
D: Dimension, 
    S: Into<StrideShape<D>>,
    R: Read,
{
    let shape = shape.into();
    let mut ws = Vec::with_capacity(shape.size());
    let mut reader = buf.take((shape.size() * 4) as u64);
    while let Ok(val) = reader.read_f32::<LittleEndian>() {
        ws.push(val.as_());
    }
    Array::from_shape_vec(shape, ws).expect("read weights: shape mismatch")
}

impl Weights {

    pub fn read<R: Read>(conf: &Config, buf: &mut BufReader<R>) -> Self {
        let tet = read_f32(buf, (conf.vocab_size, conf.dim));
        Weights {
            tet: tet.clone(),
            rms_att_weight: read_f32(buf, (conf.n_layers, conf.dim)),
            wq: read_f32(buf, (conf.n_layers, conf.dim, conf.n_heads * conf.head_size())),
            wk: read_f32(buf, (conf.n_layers, conf.dim, conf.n_kv_heads * conf.head_size())),
            wv: read_f32(buf, (conf.n_layers, conf.dim, conf.n_kv_heads * conf.head_size())),
            wo: read_f32(buf, (conf.n_layers, conf.n_heads * conf.head_size(), conf.dim)),
            rms_ffn_weight: read_f32(buf, (conf.n_layers, conf.dim)),
            w1: read_f32(buf, (conf.n_layers, conf.hidden_dim, conf.dim)),
            w2: read_f32(buf, (conf.n_layers, conf.dim, conf.hidden_dim)),
            w3: read_f32(buf, (conf.n_layers, conf.hidden_dim, conf.dim)),
            rms_final_weight: read_f32(buf, conf.dim),
            wcls: if conf.shared_weights {
                tet
            } else {
                read_f32(buf, (conf.vocab_size, conf.dim))
            },
        }
    }
}

fn to_io_err(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

pub fn load_tokenizer(config: &Config) -> io::Result<(Vec<String>, Vec<f32>)> {
    let mut vocab: Vec<String> = Vec::with_capacity(config.vocab_size);
    let mut vocab_scores: Vec<f32> = Vec::with_capacity(config.vocab_size);

    let file = File::open("./models/tokenizer.bin")?;
    let mut reader = BufReader::new(file);

    let _ = reader.read_u32::<LittleEndian>()
        .map_err(|_| to_io_err("Failed to read max token length"))?;

    for _ in 0..config.vocab_size {
        let vocab_score = reader.read_f32::<LittleEndian>()
            .map_err(|_| to_io_err("Failed to read vocab score"))?;
        vocab_scores.push(vocab_score);

        let len = reader.read_u32::<LittleEndian>()
            .map_err(|_| to_io_err("Failed to read token length"))? as usize;
        let mut token_buffer = vec![0u8; len];
        reader.read_exact(&mut token_buffer)
            .map_err(|_| to_io_err("Failed to read token"))?;
        let token = String::from_utf8(token_buffer)
            .map_err(|_| to_io_err("Failed to convert bytes to UTF-8 string"))?;

        vocab.push(token);
    }
    Ok((vocab, vocab_scores))
}
