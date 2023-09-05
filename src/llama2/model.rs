
use std::fs::File;
use std::io::{self, Read, BufReader};

use ndarray::prelude::*;
use ndarray::StrideShape;

use byteorder::{ReadBytesExt, LittleEndian};
use num_traits::cast::AsPrimitive;

use crate::llama2::quant::*;

// Weights type, placeholder for quantization
pub type WTy = i8;

pub const STRIDE: usize = 64;

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
// note dim == n_heads * head_size
pub struct Weights {
    pub tet: QintArray2<WTy>, // (vocab_size, dim) token embeddings table
    pub rms_att: QintArray2<WTy>, // (n_layers, dim) rmsnorm weights
    pub rms_ffn: QintArray2<WTy>, // (n_layers, dim)
    pub wq: QintArray3<WTy>, // (n_layers, dim, (n_heads * head_size))
    pub wk: QintArray3<WTy>, // (n_layers, dim, (n_kv_heads * head_size))
    pub wv: QintArray3<WTy>, // (n_layers, dim, (n_kv_heads * head_size))
    pub wo: QintArray3<WTy>, // (layer, (n_heads * head_size), dim)
    pub w1: QintArray3<WTy>, // (n_layers, hidden_dim, dim)
    pub w2: QintArray3<WTy>, // (n_layers, dim, hidden_dim)
    pub w3: QintArray3<WTy>, // (n_layers, hidden_dim, dim)
    pub rms_final: QintArray1<WTy>, // (dim)
    pub wcls: QintArray2<WTy>, // (dim, vocab_size) optional, classifier weights for logits, on the last layer
}

// Utility function to read f32s from file into an ndarray of the provided dimension
fn read_f32<R, S, D>(buf: &mut BufReader<R>, shape: S) -> Array<f32, D>
where
    D: Dimension, 
    S: Into<StrideShape<D>>,
    R: Read,
{
    let shape = shape.into();
    let mut ws = Vec::with_capacity(shape.size());
    let mut reader = buf.take((shape.size() * 4) as u64);
    println!("Reading and quantizing {} weights in a {:?} matrix", shape.size(), shape.raw_dim());
    while let Ok(val) = reader.read_f32::<LittleEndian>() {
        ws.push(val.as_());
    }
    Array::from_shape_vec(shape, ws).expect("read weights: shape mismatch")
}

fn read_f32_2<R, S>(buf: &mut BufReader<R>, shape: S) -> QintArray2<i8>
where
    S: Into<StrideShape<Ix2>>,
    R: Read,
{
    let f32a = read_f32(buf, shape);
    let qa = QintArray2::quantize(STRIDE, f32a.view());
    println!("loss: {:.3}%", qa.view().loss(&f32a.view())*100.0);
    qa
}

fn read_f32_3<R, S>(buf: &mut BufReader<R>, shape: S) -> QintArray3<i8>
where
    S: Into<StrideShape<Ix3>>,
    R: Read,
{
    let f32a = read_f32(buf, shape);
    let qa = QintArray3::quantize(STRIDE, f32a.view());
    println!("loss: {:.3}%", qa.view().loss(&f32a.view())*100.0);
    qa
}

impl Weights {

    pub fn read<R: Read>(conf: &Config, buf: &mut BufReader<R>) -> Self {
        let kv_dim = conf.n_kv_heads * conf.head_size();
        let tet = QintArray2::quantize(STRIDE, read_f32(buf, (conf.vocab_size, conf.dim)).view());
        Weights {
            tet: tet.clone(),
            rms_att: read_f32_2(buf, (conf.n_layers, conf.dim)),
            wq: read_f32_3(buf, (conf.n_layers, conf.dim, conf.dim)),
            wk: read_f32_3(buf, (conf.n_layers, conf.dim, kv_dim)),
            wv: read_f32_3(buf, (conf.n_layers, conf.dim, kv_dim)),
            wo: read_f32_3(buf, (conf.n_layers, conf.dim, conf.dim)),
            rms_ffn: read_f32_2(buf, (conf.n_layers, conf.dim)),
            w1: read_f32_3(buf, (conf.n_layers, conf.hidden_dim, conf.dim)),
            w2: read_f32_3(buf, (conf.n_layers, conf.dim, conf.hidden_dim)),
            w3: read_f32_3(buf, (conf.n_layers, conf.hidden_dim, conf.dim)),
            rms_final: QintArray1::quantize(STRIDE, read_f32(buf, conf.dim).view()),
            wcls: if conf.shared_weights {
                tet
            } else {
                read_f32_2(buf, (conf.vocab_size, conf.dim))
            },
        }
    }
}
