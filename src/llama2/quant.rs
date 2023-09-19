
use ndarray::prelude::*;
use ndarray::{par_azip, RemoveAxis};

// A bucket-quantized int array. The orignal f32 array is quantized into buckets
// of size `stride`, and the scaling factor for each bucket is stored in the
// `scaling` array. The resulting quantized ints are stored in a regular ndarray
// Array of ints.
// Currently only quantization to i8 is supported.
#[derive(Debug, Clone)]
pub struct QintArray<T, D> where D: Dimension {
    pub stride: usize,
    pub scaling: Array<f32, D>,
    pub arr: Array<T, D>
}

impl<T, D> QintArray<T, D> where D: Dimension {
    pub fn len(&self) -> usize {
        self.arr.len()
    }

    pub fn view(&self) -> QintArrayView<'_, T, D> {
        QintArrayView {
            stride: self.stride,
            scaling: self.scaling.view(),
            arr: self.arr.view(),
        }
    }
}

// i8 quantization
impl<D> QintArray<i8, D> where D: Dimension {
    pub fn quantize(stride: usize, fpa: ArrayView<f32, D>) -> Self {
        quant_i8(stride, fpa)
    }
}

impl<T, D> QintArray<T, D>
where
    D: RemoveAxis,
{
    pub fn index_axis(&self, axis: Axis, index: usize) -> QintArrayView<'_, T, D::Smaller> {
        QintArrayView {
            stride: self.stride,
            scaling: self.scaling.index_axis(axis, index),
            arr: self.arr.index_axis(axis, index),
        }
    }
}

// A quantized array view, similar to ndarray::ArrayView, but with stride and
// scaling factor information.
#[derive(Debug, Clone)]
pub struct QintArrayView<'a, T, D> where D: Dimension {
    pub stride: usize,
    pub scaling: ArrayView<'a, f32, D>,
    pub arr: ArrayView<'a, T, D>
}

impl<'a, 'b, T, D> QintArrayView<'a, T, D>
where
    D: RemoveAxis,
{
    pub fn index_axis(&'a self, axis: Axis, index: usize) -> QintArrayView<'b, T, D::Smaller> where 'a: 'b {
        QintArrayView {
            stride: self.stride,
            scaling: self.scaling.index_axis(axis, index),
            arr: self.arr.index_axis(axis, index),
        }
    }
}

// A quantized array constructed from an Array1<f32>
pub type QintArray1<T> = QintArray<T, Ix1>;

pub type QintArrayView1<'a, T> = QintArrayView<'a, T, Ix1>;

impl QintArrayView1<'_, i8> {
    fn dot_deq(&self, other: &Self) -> f32 {
        // almost all execution time is spent here, worth finicking over
        debug_assert_eq!(self.stride, other.stride);
        debug_assert_eq!(self.arr.len(), other.arr.len());
        debug_assert_eq!(self.scaling.len(), other.scaling.len());

        let mut sum = 0.0;
        azip!((
            x in self.arr.exact_chunks(self.stride),
            y in other.arr.exact_chunks(other.stride),
            sx in self.scaling,
            sy in other.scaling,
        ) {
            let mut stride_sum = 0;
            azip!((xi in x, yi in y) {
                stride_sum += *xi as i32 * *yi as i32;
            });
            sum += stride_sum as f32 / (sx * sy);
        });
        sum
    }

    pub fn loss(&self, original: &ArrayView1<f32>) -> f32 {
        loss(&self.to_f32().view(), original)
    }

    pub fn to_f32(&self) -> Array1<f32> {
        dequant_i8(self)
    }
}

pub type QintArray2<T> = QintArray<T, Ix2>;

pub type QintArrayView2<'a, T> = QintArrayView<'a, T, Ix2>;

impl<'a, 'b> QintArrayView2<'a, i8> {
    pub fn dot(&'a self, x: QintArrayView1<'b, i8>) -> Array1<f32> where 'a: 'b {
        let mut res = Array1::zeros(self.arr.shape()[0]);
        par_azip!((
            dot in res.view_mut(),
            row in self.arr.outer_iter(),
            scale in self.scaling.outer_iter(),
        ) {
            let qv1 = QintArrayView1 {
                stride: self.stride,
                arr: row,
                scaling: scale,
            };
            *dot = qv1.dot_deq(&x);
        });
        res
    }

    pub fn loss(&'a self, original: &ArrayView2<f32>) -> f32 {
        let mut total_loss = 0.0;
        let sz = self.arr.shape()[0];
        for n in 0..sz {
            let q = self.index_axis(Axis(0), n);
            let o = original.index_axis(Axis(0), n);
            total_loss += q.loss(&o)
        }
        total_loss / sz as f32
    }

}

pub type QintArray3<T> = QintArray<T, Ix3>;

pub type QintArrayView3<'a, T> = QintArrayView<'a, T, Ix3>;

impl<'a, 'b> QintArrayView3<'a, i8> {
    pub fn loss(&'a self, original: &ArrayView3<f32>) -> f32 {
        let mut total_loss = 0.0;
        let sz = self.arr.shape()[0];
        for n in 0..sz {
            let q = self.index_axis(Axis(0), n);
            let o = original.index_axis(Axis(0), n);
            total_loss += q.loss(&o)
        }
        total_loss / sz as f32
    }

}

// Calculate the dimension of the scaling factors for a given original dimension
// and stride.
pub fn scaling_dim<D>(dim: &D, stride: usize) -> D where D: Dimension {
    let mut scaling_dim = dim.clone();
    let last_dim = scaling_dim.ndim() - 1;
    scaling_dim[last_dim] /= stride;
    scaling_dim
}

fn quant_i8<D>(stride: usize, fpa: ArrayView<f32, D>) -> QintArray<i8, D> where D: Dimension {
    assert!(fpa.len() % stride == 0);

    let mut qdata = Vec::with_capacity(fpa.len());
    let mut scaling = Vec::with_capacity(fpa.len() / stride);

    for bucket in fpa.as_slice().unwrap().chunks(stride) {
        let (min, max) = bucket.iter()
            .fold((f32::MAX, f32::MIN), |(min, max), &x| (min.min(x), max.max(x)));
        let scale = i8::MAX as f32 / max.abs().max(min.abs());
        scaling.push(scale);
        for b in bucket {
            qdata.push((b * scale).round() as i8);
        }
    }
    QintArray {
        stride,
        scaling: Array::from_shape_vec(scaling_dim(&fpa.raw_dim(), stride), scaling).unwrap(),
        arr: Array::from_shape_vec(fpa.raw_dim(), qdata).unwrap(),
    }
}

fn dequant_i8(ia: &QintArrayView1<i8>) -> Array1<f32> {
    let mut arr = Array1::zeros(ia.arr.dim());
    azip!((
        qbuck in ia.arr.exact_chunks(ia.stride),
        scale in ia.scaling.view(),
        buck in arr.exact_chunks_mut(ia.stride),
    ) {
        azip!((
            qb in qbuck,
            b in buck,
        ) {
            *b = *qb as f32 / scale;
        });

    });
    arr
}

fn loss(fp1: &ArrayView1<f32>, fp2: &ArrayView1<f32>) -> f32 {
    std::iter::zip(fp1, fp2).map(|(x1, x2)| (x1 - x2).abs()).sum::<f32>() / fp1.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::random;

    #[test]
    fn it_quantizes() {
        let fpa = Array::from_vec(vec![-0.9, -0.35, -0.1, 0.2, 0.6, 0.7]); 
        println!("fpa: {:?}", quant_i8(2, fpa.view()));
        assert_eq!(quant_i8(2, fpa.view()).arr.to_vec(), vec![-127, -49, -64, 127, 109, 127]);
    }

    #[test]
    fn it_has_little_loss() {
        let fpa = QintArray1 {
            stride: 2,
            scaling: Array::from_vec(vec![141.11111, 635.0, 181.42857]),
            arr: Array::from_vec(vec![-127, -49, -64, 127, 109, 127]),
        };
        let dq_arr = dequant_i8(&fpa.view());
        let lo = loss(&dq_arr.view(), &Array::from_vec(vec![-0.9, -0.35, -0.1, 0.2, 0.6, 0.7]).view());
        println!("dq arr: {}, loss: {}", dq_arr, lo);
        if lo > 0.001 {
            panic!("high loss: {}", lo);
        }
    }

    #[test]
    fn it_quantizes_many_with_little_loss() {
        let mut total_loss = 0.0;
        let mut rng = rand::thread_rng();
        for n in 0..1000 {
            let fpa = Array::from_vec((0..2048).map(|_| 4.0*random::<f32>()).collect());
            let q = quant_i8(64, fpa.view());
            let l = loss(&fpa.view(), &dequant_i8(&q.view()).view());
            if l > 0.01 {
                panic!("Loss over 0.1%: {} ({})", l, n);
            }
            total_loss += l;
        }
        println!("Average loss: {}", total_loss / 1000.0);
    }

    #[test]
    fn it_dots() {
        let fpa1 = Array::from_vec(vec![-0.9, -0.35, -0.1, 0.2, 0.6, 0.7]); 
        let fpa2 = Array::from_vec(vec![-0.75, -0.2, 0.05, 0.4, 0.9, 0.95]); 
        let fdot = fpa1.dot(&fpa2);
        let i1 = quant_i8(2, fpa1.view());
        let i2 = quant_i8(2, fpa2.view());
        let idot = i1.view().dot_deq(&i2.view());
        println!("f: {}, i: {}", fdot, idot);
        assert!((fdot - idot).abs() < 0.01);
    }
}

