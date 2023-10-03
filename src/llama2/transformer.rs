
use num_traits::cast::AsPrimitive;
use ndarray::prelude::*;
use ndarray::par_azip;

use crate::llama2::model::*;
use crate::llama2::quant::*;

// Activations type, placeholder for quantization
type ATy = f32;

const ATY_ZERO: ATy = 0.0;
const ATY_ONE: ATy = 1.0;

// Generic vector and matrix functions used for inference

fn rmsnorm(o: ArrayViewMut1<ATy>, x: ArrayView1<ATy>, weight: TensorView1) {
    // calculate sum of squares
    let ss = (x.dot(&x) / x.len() as ATy + 1e-5).sqrt().recip();

    // normalize and scale
    if weight.is_quantized() {
        let weight = weight.to_f32();
        azip!((o_val in o, &x_val in x, &weight_val in weight.view()) {
            *o_val = weight_val * ss * x_val;
        });
    } else {
        let weight = weight.unwrap_f32();
        azip!((o_val in o, &x_val in x, &weight_val in weight) {
            *o_val = weight_val * ss * x_val;
        });
    }
}

pub fn softmax(x: &mut ArrayViewMut1<ATy>) {
    let max_val = x.iter().fold(ATy::NEG_INFINITY, |a, &b| a.max(b));
    x.map_inplace(|value| *value = (*value - max_val).exp() );
    *x /= x.sum();
}

fn matmul<'a>(mut xout: ArrayViewMut1<ATy>, x: &Array1<ATy>, w: &TensorView2<'a>) {
    debug_assert_eq!(w.shape(), &[xout.len(), x.len()]);

    if w.is_quantized() {
        let xq = Tensor::Qi8(QintArray1::quantize(DEFAULT_STRIDE, x.view()));
        xout.assign(&w.dot(xq.view()));
    } else {
        par_azip!((out_val in xout, row in w.unwrap_f32().outer_iter()) {
            *out_val = row.dot(x);
        });
    }
}

// sine and cosine tables for RoPE
fn sin_cos(pos: usize, dim: usize, head_size: usize) -> Vec<(ATy, ATy)> {
    (0..dim / 2).map(|i| {
        let head_dim = ((i * 2) % head_size) as ATy;
        let pos_freq = (pos as f32) * (1.0 / (10000.0f32).powf(head_dim / head_size as ATy));
        pos_freq.sin_cos()
    }).collect()
}

// Implements the inference. In particular see the transformer function.
pub struct Transformer {
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

    conf: Config,
    w: Weights,
}

impl Transformer {
    pub fn new(config: &Config, weights: Weights) -> Transformer {
        let a1_dim = Array1::from_elem(config.dim, ATY_ZERO);
        Transformer {
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
            conf: config.clone(),
            w: weights,
        }
    }

    pub fn forward(&mut self, token: usize, pos: usize) -> ArrayViewMut1<ATy> {
        let sin_cos = sin_cos(pos, self.conf.dim, self.conf.head_size());
        // copy the token embedding into x
        self.x.assign(&self.w.tet.index_axis(Axis(0), token).to_f32().view());

        // forward all the layers
        for l in 0..self.conf.n_layers {
            // attention rmsnorm
            rmsnorm(self.xb.view_mut(), self.x.view(), self.w.rms_att.index_axis(Axis(0), l));

            // qkv matmuls for this position
            matmul(self.q.view_mut(), &self.xb, &self.w.wq.index_axis(Axis(0), l));
            matmul(self.k.view_mut(), &self.xb, &self.w.wk.index_axis(Axis(0), l));
            matmul(self.v.view_mut(), &self.xb, &self.w.wv.index_axis(Axis(0), l));

            self.rope_enc(&sin_cos);

            // compute multihead attention
            self.attention(l, pos);

            // final matmul to get the output of the attention
            matmul(self.xb2.view_mut(), &self.xb, &self.w.wo.index_axis(Axis(0), l));

            // residual connection back into x
            self.x += &self.xb2.view();

            // ffn rmsnorm
            rmsnorm(self.xb.view_mut(), self.x.view(), self.w.rms_ffn.index_axis(Axis(0), l));

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            matmul(self.hb.view_mut(), &self.xb, &self.w.w1.index_axis(Axis(0), l));
            matmul(self.hb2.view_mut(), &self.xb, &self.w.w3.index_axis(Axis(0), l));

            // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
            self.hb.iter_mut().for_each(|value| {
                *value = *value * (ATY_ONE / (ATY_ONE + (-*value).exp()))
            });
            // elementwise multiply with w3(x)
            self.hb *= &self.hb2;

            // final matmul
            matmul(self.xb.view_mut(), &self.hb, &self.w.w2.index_axis(Axis(0), l));

            // residual connection back into x
            self.x += &self.xb;
        }

        // Final layer norm
        rmsnorm(self.xb.view_mut(), self.x.view(), self.w.rms_final.view());

        // Class logits
        matmul(self.logits.view_mut(), &self.xb, &self.w.wcls.view());

        return self.logits.view_mut();
    }

    // Multihead attention calculation, iterates over all heads
    pub fn attention(&mut self, layer: usize, pos: usize) {
        let hs = self.conf.head_size();
        let hs_sqrt = <usize as AsPrimitive<ATy>>::as_(hs).sqrt();

        // save key,value at this time step (pos) to our kv cache
        self.key_cache
            .index_axis_mut(Axis(0), layer)
            .index_axis_mut(Axis(0), pos)
            .assign(&self.k);
        self.value_cache
            .index_axis_mut(Axis(0), layer)
            .index_axis_mut(Axis(0), pos)
            .assign(&self.v);

        let layer_key_cache = self.key_cache.index_axis(Axis(0), layer);
        let layer_value_cache = self.value_cache.index_axis(Axis(0), layer);

        par_azip!((
            index h,
            q in self.q.exact_chunks(hs),
            mut xb in self.xb.exact_chunks_mut(hs),
            mut att in self.att.exact_chunks_mut(self.conf.seq_len))
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
                softmax(&mut att.slice_mut(s![0..pos+1]));
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
    fn rope_enc(&mut self, sin_cos: &Vec<(ATy, ATy)>) {
        let kv_dim = (self.conf.dim * self.conf.n_kv_heads) / self.conf.n_heads;
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
