
// When single-threaded, blas is fastest but with multithreading in rayon it
// somehow slows things down at least on small models. May be worth investigating.
// extern crate blas_src;

use std::env;
use std::fs::File;
use std::io::{self, Read, Write, BufReader};
use std::time::Instant;

use byteorder::{ReadBytesExt, LittleEndian};
use num_traits::cast::AsPrimitive;
use rand::Rng;
use ndarray::prelude::*;
use ndarray::{StrideShape, par_azip};

// Weights and activation types, placeholder for quantization
type WTy = f32;
type ATy = f32;

const ATY_ZERO: ATy = 0.0;
const ATY_ONE: ATy = 1.0;

fn deq(w: WTy) -> ATy {
    w // .to_f32()
}

// The original format has 7 serialized fields of 4 bytes each
const CONFIG_SIZE: usize = 7*4;

// Transformer configuration. Doesn't directly affect the model as it's based on llama2
// but allows variations in dimensions, number of layers, etc.
#[derive(Default, Debug)]
pub struct Config {
    dim: usize, // transformer dimension
    hidden_dim: usize, // for ffn layers
    n_layers: usize, // number of layers
    n_heads: usize, // number of query heads
    n_kv_heads: usize, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: usize, // vocabulary size, usually 256 (byte-level)
    seq_len: usize, // max sequence length
    shared_weights: bool,
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
    tet: Array2<WTy>, // (vocab_size, dim) token embeddings table
    rms_att_weight: Array2<WTy>, // (n_layers, dim) rmsnorm weights
    rms_ffn_weight: Array2<WTy>, // (n_layers, dim)
    wq: Array3<WTy>, // (n_layers, dim, (n_heads * head_size))
    wk: Array3<WTy>, // (n_layers, dim, (n_kv_heads * head_size))
    wv: Array3<WTy>, // (n_layers, dim, (n_kv_heads * head_size))
    wo: Array3<WTy>, // (layer, (n_heads * head_size), dim)
    w1: Array3<WTy>, // (n_layers, hidden_dim, dim)
    w2: Array3<WTy>, // (n_layers, dim, hidden_dim)
    w3: Array3<WTy>, // (n_layers, hidden_dim, dim)
    rms_final_weight: Array1<WTy>, // (dim)
    wcls: Array2<WTy>, // (dim, vocab_size) optional, classifier weights for logits, on the last layer
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

// Generic vector and matrix functions used for inference

fn rmsnorm(o: ArrayViewMut1<ATy>, x: ArrayView1<ATy>, weight: ArrayView1<WTy>) {
    // calculate sum of squares
    let xlen: ATy = x.len().as_();
    let low_val: ATy = 1e-5.as_();
    let ss = (x.dot(&x) / xlen + low_val).sqrt().recip();

    // normalize and scale
    azip!((o_val in o, &x_val in x, &weight_val in weight.mapv(deq).view()) {
        *o_val = weight_val * ss * x_val;
    });
}

fn softmax(mut x: ArrayViewMut1<ATy>) {
    let max_val = x.iter().fold(ATy::NEG_INFINITY, |a, &b| a.max(b));
    x.map_inplace(|value| *value = (*value - max_val).exp() );
    x /= x.sum();
}

fn matmul(xout: ArrayViewMut1<ATy>, x: ArrayView1<ATy>, w: ArrayView2<WTy>) {
    debug_assert_eq!(w.shape(), &[xout.len(), x.len()]);

    // almost all execution time is spent here, worth finicking with
    par_azip!((out_val in xout, row in w.outer_iter()) {
        *out_val = row.dot(&x);
    });
    // One-liner version, without multithreading
    // xout.assign(&w.dot(&x));
}

// sine and cosine tables for RoPE
fn sin_cos(pos: usize, conf: &Config) -> Vec<(ATy, ATy)> {
    (0..conf.dim / 2).map(|i| {
        let head_dim = ((i * 2) % conf.head_size()) as ATy;
        let pos_freq = (pos as f32) * (1.0 / (10000.0f32).powf(head_dim / conf.head_size() as ATy));
        pos_freq.sin_cos()
    }).collect()
}

// Implements the inference. In particular see the transformer function.
pub struct RunState {
    x: Array1<ATy>, // activation at current time stamp (dim,)
    xb: Array1<ATy>, // same, but inside a residual branch (dim,)
    xb2: Array1<ATy>, // an additional buffer just for convenience (dim,)
    hb: Array1<ATy>, // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Array1<ATy>, // buffer for hidden dimension in the ffn (hidden_dim,)
    q: Array1<ATy>, // query (dim,)
    k: Array1<ATy>, // key (dim,)
    v: Array1<ATy>, // value (dim,)
    att: Array1<ATy>, // buffer for scores/attention values (n_heads, seq_len)
    logits: Array1<ATy>, // output logits
    key_cache: Array3<ATy>,   // (layer, seq_len, dim)
    value_cache: Array3<ATy>, // (layer, seq_len, dim)
}

impl RunState {
    pub fn new(config: &Config) -> RunState {
        let a1_dim = Array1::from_elem(config.dim, ATY_ZERO);
        RunState {
            x: a1_dim.clone(),
            xb: a1_dim.clone(),
            xb2: a1_dim.clone(),
            hb: Array1::from_elem(config.hidden_dim, ATY_ZERO),
            hb2: Array1::from_elem(config.hidden_dim, ATY_ZERO),
            q: a1_dim.clone(),
            k: a1_dim.clone(),
            v: a1_dim,
            att: Array1::from_elem(config.n_heads * config.seq_len, ATY_ZERO),
            logits: Array1::from_elem(config.vocab_size, ATY_ZERO),
            key_cache: Array3::from_elem((config.n_layers, config.seq_len, config.dim), ATY_ZERO),
            value_cache: Array3::from_elem((config.n_layers, config.seq_len, config.dim), ATY_ZERO),
        }
    }

    pub fn transformer(&mut self, token: usize, pos: usize, conf: &Config, w: &Weights) {
        let sin_cos = sin_cos(pos, conf);
        // copy the token embedding into x
        self.x.assign(&w.tet.index_axis(Axis(0), token));

        // forward all the layers
        for l in 0..conf.n_layers {
            // attention rmsnorm
            rmsnorm(self.xb.view_mut(), self.x.view(), w.rms_att_weight.index_axis(Axis(0), l));

            // qkv matmuls for this position
            matmul(self.q.view_mut(), self.xb.view(), w.wq.index_axis(Axis(0), l));
            matmul(self.k.view_mut(), self.xb.view(), w.wk.index_axis(Axis(0), l));
            matmul(self.v.view_mut(), self.xb.view(), w.wv.index_axis(Axis(0), l));

            self.rope_enc(&sin_cos, conf);

            // compute multihead attention
            self.attention(l, pos, conf);

            // final matmul to get the output of the attention
            matmul(self.xb2.view_mut(), self.xb.view(), w.wo.index_axis(Axis(0), l));

            // residual connection back into x
            self.x += &self.xb2.view();

            // ffn rmsnorm
            rmsnorm(self.xb.view_mut(), self.x.view(), w.rms_ffn_weight.index_axis(Axis(0), l));

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            matmul(self.hb.view_mut(), self.xb.view(), w.w1.index_axis(Axis(0), l));
            matmul(self.hb2.view_mut(), self.xb.view(), w.w3.index_axis(Axis(0), l));

            // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
            self.hb.iter_mut().for_each(|value| {
                *value = *value * (ATY_ONE / (ATY_ONE + (-*value).exp()))
            });
            // elementwise multiply with w3(x)
            self.hb *= &self.hb2;

            // final matmul
            matmul(self.xb.view_mut(), self.hb.view(), w.w2.index_axis(Axis(0), l));

            // residual connection back into x
            self.x += &self.xb;
        }

        // Final layer norm
        rmsnorm(self.xb.view_mut(), self.x.view(), w.rms_final_weight.view());

        // Class logits
        matmul(self.logits.view_mut(), self.xb.view(), w.wcls.view());
    }

    // Multihead attention calculation, iterates over all heads
    pub fn attention(&mut self, layer: usize, pos: usize, conf: &Config) {
        let hs = conf.head_size();
        let hs_sqrt = <usize as AsPrimitive<ATy>>::as_(hs).sqrt();

        // save key,value at this time step (pos) to our kv cache
        self.key_cache.index_axis_mut(Axis(0), layer).index_axis_mut(Axis(0), pos).assign(&self.k);
        self.value_cache.index_axis_mut(Axis(0), layer).index_axis_mut(Axis(0), pos).assign(&self.v);

        let layer_key_cache = self.key_cache.index_axis(Axis(0), layer);
        let layer_value_cache = self.value_cache.index_axis(Axis(0), layer);

        azip!((
            index h,
            q in self.q.exact_chunks(hs),
            mut xb in self.xb.exact_chunks_mut(hs),
            mut att in self.att.exact_chunks_mut(conf.seq_len))
            {
                // get the cache vectors for this head at the given timestep
                let with_cache = |t, cache: &ArrayView2<ATy>, f: &mut dyn FnMut(ArrayView1<ATy>)| {
                    let axis = cache.index_axis(Axis(0), t);
                    let slice = axis.slice(s![h * hs..(h + 1) * hs]);
                    f(slice);
                };

                // iterate over all timesteps, including the current one
                for t in 0..=pos {
                    // calculate the attention score as the dot product of q and k
                    with_cache(t, &layer_key_cache, &mut |k| {
                        att[t] = q.dot(&k) / hs_sqrt;
                    });
                }

                // softmax the scores to get attention weights, from 0..pos inclusively
                softmax(att.slice_mut(s![0..pos+1]));
                // weighted sum of the values, store back into xb
                xb.fill(ATY_ZERO);

                for t in 0..=pos {
                    with_cache(t, &layer_value_cache, &mut |v| {
                        azip!((xb in &mut xb, &v_val in &v) *xb += att[t] * v_val);
                    });
                }
            });
    }

    // RoPE relative positional encoding: complex-valued rotate q and k by freq_cis in each head
    fn rope_enc(&mut self, sin_cos: &Vec<(ATy, ATy)>, conf: &Config) {
        let kv_dim = (conf.dim * conf.n_kv_heads) / conf.n_heads;
        let mut vecs = [self.q.view_mut(), self.k.view_mut()];

        for (i, &(fci, fcr)) in sin_cos.iter().enumerate() {
            let idx = i * 2;
            let rotn = if idx < kv_dim { 2 } else { 1 };
            for v in 0..rotn {
                let vec = &mut vecs[v];
                let (v0, v1) = (vec[idx], vec[idx + 1]);
                vec[idx] = v0 * fcr - v1 * fci;
                vec[idx + 1] = v0 * fci + v1 * fcr;
            }
        }
    }
}

pub fn sample(probabilities: ArrayView1<ATy>) -> Option<usize> {
    let r: ATy = rand::thread_rng().gen();
    let mut cdf = ATY_ZERO;
    probabilities.iter().enumerate().find(|&(_, &prob)| {
        cdf += prob;
        r < cdf
    }).map(|(i, _)| i)
}

pub fn argmax(v: ArrayView1<ATy>) -> Option<usize> {
    v.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
}

pub fn gen() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: {} <checkpoint_file> [temperature] [steps]", args[0]);
        std::process::exit(1);
    }
    let temperature: ATy = args.get(2).map_or(0.9, |v| v.parse().expect("Fail parse temperature")).as_();
    let steps: usize = args.get(3).map_or(256, |v| v.parse().expect("Failed to parse steps"));

    // Initialize config from file
    let mut file = File::open(&args[1]).expect("Unable to open the checkpoint file");
    let config = Config::from_file(&mut file).expect("Failed to read the config");
    println!("Model config: {:?}", config);

    // Finish reading the checkpoint file by loading weights
    let mut reader = BufReader::new(file);
    let weights = Weights::read(&config, &mut reader);

    let steps = steps.max(1).min(config.seq_len);

    // Load tokenizer
    let (vocab, _vocab_scores) = load_tokenizer(&config).expect("Failed to load tokenizer");

    let mut state = RunState::new(&config);
    let start = Instant::now();
    let mut token = 1; // BOS token
    println!("<s>");

    for pos in 0..steps {
        state.transformer(token, pos, &config, &weights);

        let next = if temperature == ATY_ZERO {
            argmax(state.logits.view())
        } else {
            for logits in state.logits.iter_mut() {
                *logits /= temperature;
            }
            softmax(state.logits.view_mut());
            sample(state.logits.view())
        };

        print!("{}", vocab[next.unwrap()]);
        io::stdout().flush().unwrap();
        token = next.unwrap();
    }
    println!("\nachieved tok/s: {}", steps as f64 / start.elapsed().as_secs_f64());
}
