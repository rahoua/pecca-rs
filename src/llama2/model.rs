
use std::fs::File;
use std::io::{self, Read, BufReader, BufRead, Write};

use ndarray::prelude::*;
use ndarray::StrideShape;

use byteorder::{WriteBytesExt, LittleEndian, ByteOrder};
use num_traits::cast::AsPrimitive;
use num_traits::{FromPrimitive, ToPrimitive};

use crate::llama2::quant::*;

// Weights type, placeholder for quantization
pub type WTy = i8;

pub const DEFAULT_STRIDE: usize = 64;

// The original format has 7 serialized fields of 4 bytes each
const CONFIG_SIZE: usize = 7*4;

// Quantization type, currently only linear i8
#[derive(Default, Debug, Clone, PartialEq, FromPrimitive, ToPrimitive)]
pub enum QuantizationType {
    #[default]
    None = 0,
    LinearI8 = 1,
}

#[derive(Default, Debug, Clone, PartialEq, FromPrimitive, ToPrimitive)]
pub enum Version {
    Llama2C = 0,
    #[default]
    V1 = 1,
}

// Transformer configuration. Doesn't directly affect the model as it's based on llama2
// but allows variations in dimensions, number of layers, etc.
#[derive(Default, Debug)]
pub struct Config {
    pub version: Version, // version number, currently 1
    pub dim: usize, // transformer dimension
    pub hidden_dim: usize, // for ffn layers
    pub n_layers: usize, // number of layers
    pub n_heads: usize, // number of query heads
    pub n_kv_heads: usize, // number of key/value heads (can be < query heads because of multiquery)
    pub vocab_size: usize, // vocabulary size, usually 256 (byte-level)
    pub seq_len: usize, // max sequence length
    pub shared_weights: bool,
    pub q_type: QuantizationType, // quantization type, currently only linear i8
    pub q_stride: usize, // quantization stride
}

impl Config {
    pub fn from_file(file: &mut File) -> io::Result<Self> {
        // the C code uses int, which is 4 bytes, we use usize because it's cleaner
        // down the road but requires a little size arithmetic
        let mut config_bytes = vec![0; CONFIG_SIZE];
        file.read_exact(&mut config_bytes)?;

        // The original llama2.c format has no version numbers. We detect its presence
        // here by checking the dim, as practically a model with a dim so low is useless.
        let mut version = Version::Llama2C;
        let version_num = LittleEndian::read_u32(&config_bytes[0..4]);
        if version_num < 50 {
            let mut additional_bytes = vec![0; 4 * 4];
            file.read_exact(&mut additional_bytes)?;
            config_bytes.extend_from_slice(&additional_bytes);
            config_bytes.drain(0..4);
            version = FromPrimitive::from_u32(version_num).expect("invalid version");
        }

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

            // our additional properties
            version: version,
            shared_weights: true,
            q_type: QuantizationType::None,
            q_stride: 0,
        };
        if config.version != Version::Llama2C {
            println!("Version read: {:?}", config.version);
            config.shared_weights = read_usize() != 0;
            config.q_type = FromPrimitive::from_usize(read_usize())
                .expect("invalid quantization type");
            config.q_stride = read_usize();
        }
        // a little whackiness in the format
        if config.vocab_size >= 2_usize.pow(31) {
            config.vocab_size = (!(config.vocab_size as u32) + 1) as usize;
            config.shared_weights = false;
        }
        Ok(config)
    }

    pub fn write<W: Write>(&self, buf: &mut W) -> io::Result<()> {
        buf.write_u32::<LittleEndian>(Version::V1.to_usize().unwrap() as u32)?;
        buf.write_u32::<LittleEndian>(self.dim as u32)?;
        buf.write_u32::<LittleEndian>(self.hidden_dim as u32)?;
        buf.write_u32::<LittleEndian>(self.n_layers as u32)?;
        buf.write_u32::<LittleEndian>(self.n_heads as u32)?;
        buf.write_u32::<LittleEndian>(self.n_kv_heads as u32)?;
        buf.write_u32::<LittleEndian>(self.vocab_size as u32)?;
        buf.write_u32::<LittleEndian>(self.seq_len as u32)?;
        buf.write_u32::<LittleEndian>(self.shared_weights as u32)?;
        buf.write_u32::<LittleEndian>(self.q_type.to_usize().unwrap() as u32)?;
        buf.write_u32::<LittleEndian>(self.q_stride as u32)?;
        Ok(())
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

fn read_i8<R, S, D>(conf: &Config, buf: &mut BufReader<R>, shape: S) -> QintArray<i8, D>
where
    R: Read,
    S: Into<StrideShape<D>>,
    D: Dimension,
{
    let shape = shape.into();
    let ndim = shape.raw_dim().ndim();
    let mut scaling_shape = shape.raw_dim().clone();
    scaling_shape[ndim-1] = scaling_shape[ndim-1] / conf.q_stride;

    let scaling = read_array(buf, scaling_shape);
    println!("Reading {} i8 quantized weights", shape.size());
    let arr = read_array(buf, shape);

    QintArray {
        stride: conf.q_stride,
        scaling,
        arr,
    }
}

fn read_f32_qi8<R, S, D>(conf: &Config, buf: &mut BufReader<R>, shape: S) -> QintArray<i8, D>
where
    R: Read,
    S: Into<StrideShape<D>>,
    D: Dimension,
{
    let f32a = read_array(buf, shape);
    let qa = QintArray::quantize(conf.q_stride, f32a.view());
    // println!("loss: {:.3}%", qa.view().loss(&f32a.view())*100.0);
    qa
}

impl Weights {

    pub fn read<R: Read>(conf: &Config, buf: &mut BufReader<R>) -> Self {
        if conf.version == Version::Llama2C {
            println!("Reading a llama2c model...");
            Self::read_llama2c(conf, buf)
        } else {
            println!("Reading a v1 model...");
            Self::read_v1(conf, buf)
        }
    }
    fn read_llama2c<R: Read>(conf: &Config, buf: &mut BufReader<R>) -> Self {
        let kv_dim = conf.n_kv_heads * conf.head_size();
        let tet = read_f32_qi8(conf, buf, (conf.vocab_size, conf.dim));
        let mut w = Weights {
            tet: tet.clone(),
            rms_att: read_f32_qi8(conf, buf, (conf.n_layers, conf.dim)),
            wq: read_f32_qi8(conf, buf, (conf.n_layers, conf.dim, conf.dim)),
            wk: read_f32_qi8(conf, buf, (conf.n_layers, conf.dim, kv_dim)),
            wv: read_f32_qi8(conf, buf, (conf.n_layers, conf.dim, kv_dim)),
            wo: read_f32_qi8(conf, buf, (conf.n_layers, conf.dim, conf.dim)),
            rms_ffn: read_f32_qi8(conf, buf, (conf.n_layers, conf.dim)),
            w1: read_f32_qi8(conf, buf, (conf.n_layers, conf.hidden_dim, conf.dim)),
            w2: read_f32_qi8(conf, buf, (conf.n_layers, conf.dim, conf.hidden_dim)),
            w3: read_f32_qi8(conf, buf, (conf.n_layers, conf.hidden_dim, conf.dim)),
            rms_final: read_f32_qi8(conf, buf, conf.dim),
            wcls:  tet,
        };
        if !conf.shared_weights {
            // skip the old 2 freq_cis_real and freq_cis imag (used to be for RoPE)
            read_array::<_, f32, _, _>(buf, (conf.seq_len, conf.head_size()/2));
            read_array::<_, f32, _, _>(buf, (conf.seq_len, conf.head_size()/2));

            w.wcls = read_f32_qi8(conf, buf, (conf.vocab_size, conf.dim));
        }
        w
    }

    fn read_v1<R: Read>(conf: &Config, buf: &mut BufReader<R>) -> Self {
        let kv_dim = conf.n_kv_heads * conf.head_size();
        Weights {
            tet: read_i8(conf, buf, (conf.vocab_size, conf.dim)),
            rms_att: read_i8(conf, buf, (conf.n_layers, conf.dim)),
            wq: read_i8(conf, buf, (conf.n_layers, conf.dim, conf.dim)),
            wk: read_i8(conf, buf, (conf.n_layers, conf.dim, kv_dim)),
            wv: read_i8(conf, buf, (conf.n_layers, conf.dim, kv_dim)),
            wo: read_i8(conf, buf, (conf.n_layers, conf.dim, conf.dim)),
            rms_ffn: read_i8(conf, buf, (conf.n_layers, conf.dim)),
            w1: read_i8(conf, buf, (conf.n_layers, conf.hidden_dim, conf.dim)),
            w2: read_i8(conf, buf, (conf.n_layers, conf.dim, conf.hidden_dim)),
            w3: read_i8(conf, buf, (conf.n_layers, conf.hidden_dim, conf.dim)),
            rms_final: read_i8(conf, buf, conf.dim),
            wcls: read_i8(conf, buf, (conf.vocab_size, conf.dim)),
        }
    }

    pub fn write<W: Write>(&self, buf: &mut W) -> io::Result<()> {
        write_i8(&self.tet, buf)?;
        write_i8(&self.rms_att, buf)?;
        write_i8(&self.wq, buf)?;
        write_i8(&self.wk, buf)?;
        write_i8(&self.wv, buf)?;
        write_i8(&self.wo, buf)?;
        write_i8(&self.rms_ffn, buf)?;
        write_i8(&self.w1, buf)?;
        write_i8(&self.w2, buf)?;
        write_i8(&self.w3, buf)?;
        write_i8(&self.rms_final, buf)?;
        write_i8(&self.wcls, buf)?;
        Ok(())
    }
}

fn write_i8<W, D>(a: &QintArray<i8, D>, buf: &mut W) -> io::Result<()>
where
    W: Write,
    D: Dimension,
{
    for val in a.scaling.iter() {
        buf.write_f32::<LittleEndian>(val.as_())?;
    }
    for val in a.arr.iter() {
        buf.write_i8(val.as_())?;
    }
    Ok(())
}

fn read_array<R, T, D, S>(buf: &mut R, shape: S) -> Array<T, D>
where
    R: BufRead,
    D: Dimension,
    S: Into<StrideShape<D>>,
    T: BytesAsPrimitive,
{
    let shape = shape.into();
    let mut arr = Vec::with_capacity(shape.size());
    let sz = T::size();
    let mut bytes_to_read = shape.size() * sz;
    while let Ok(bytes) = buf.fill_buf() {
        if bytes.len() == 0 {
            break;
        }
        let bytes_chunk = bytes_to_read.min(bytes.len());
        let mut n = 0;
        loop {
            let val = T::read(&bytes[n..]);
            arr.push(val);
            n += sz;
            if n >= bytes_chunk {
                break;
            }
        }
        bytes_to_read -= bytes_chunk;
        buf.consume(bytes_chunk);
        if bytes_to_read == 0 {
            break;
        }
    }
    Array::from_shape_vec(shape, arr).unwrap()
}

trait BytesAsPrimitive {
    fn size() -> usize where Self: Sized;
    fn read(bytes: &[u8]) -> Self where Self: Sized;
}

impl BytesAsPrimitive for i8 {
    fn size() -> usize { 1 }
    fn read(bytes: &[u8]) -> i8 {
        bytes[0] as i8
    }
}
impl BytesAsPrimitive for f32 {
    fn size() -> usize { 4 }
    fn read(bytes: &[u8]) -> f32 {
        LittleEndian::read_f32(&bytes[0..4])
    }
}
