
use ndarray::prelude::*;
use ndarray::{par_azip, RemoveAxis};

// A quantized array constructed with a generic dimension
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

// A quantized array constructed with a generic dimension
#[derive(Debug, Clone)]
pub struct QintArrayView<'a, T, D> where D: Dimension {
    pub stride: usize,
    pub scaling: ArrayView<'a, f32, D>,
    pub arr: ArrayView<'a, T, D>
}

impl<'a, 'b, T, D> QintArrayView<'a, T, D>
where
    D: RemoveAxis,
    <D as ndarray::Dimension>::Smaller: RemoveAxis
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

impl QintArray1<i8> {
    pub fn quantize(stride: usize, fpa: ArrayView1<f32>) -> Self {
        quant_i8(stride, fpa)
    }
}

impl QintArrayView1<'_, i8> {
    // almost all execution time is spent here, worth finicking over
    fn dot_deq(&self, other: &Self) -> f32 {
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

impl QintArray2<i8> {
    pub fn quantize(stride: usize, fpa: ArrayView2<f32>) -> Self {
        assert!(fpa.dim().1 % stride == 0);

        let mut qarr = Array2::zeros(fpa.dim());
        let mut scaling = Array2::zeros((fpa.dim().0, fpa.dim().1 / stride));
        azip!((
            row in fpa.outer_iter(),
            qrow in qarr.outer_iter_mut(),
            srow in scaling.outer_iter_mut(),
        ) {
            let qarr1 = QintArray1::quantize(stride, row);
            qarr1.arr.move_into(qrow);
            qarr1.scaling.move_into(srow);
        });
        QintArray2 {
            stride: stride,
            arr: qarr,
            scaling: scaling,
        }
    }
}

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

impl QintArray3<i8> {
    pub fn quantize(stride: usize, fpa: ArrayView3<f32>) -> Self {
        assert!(fpa.dim().2 % stride == 0);

        let mut qarr = Array3::zeros(fpa.dim());
        let mut scaling = Array3::zeros((fpa.dim().0, fpa.dim().1, fpa.dim().2 / stride));
        azip!((
            row in fpa.outer_iter(),
            qrow in qarr.outer_iter_mut(),
            srow in scaling.outer_iter_mut(),
        ) {
            let qarr2 = QintArray2::quantize(stride, row);
            qarr2.arr.move_into(qrow);
            qarr2.scaling.move_into(srow);
        });
        QintArray3 {
            stride: stride,
            arr: qarr,
            scaling: scaling,
        }
    }
}

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

fn quant_i8(stride: usize, fpa: ArrayView1<f32>) -> QintArray1<i8> {
    assert!(fpa.dim() % stride == 0);

    let mut qarr = Array1::zeros(fpa.dim());
    let mut scaling = Array1::zeros(fpa.dim() / stride);

    azip!((
        bucket in fpa.exact_chunks(stride),
        qbuck in qarr.exact_chunks_mut(stride),
        scale in scaling.view_mut(),
    ) {
        let (min, max) = bucket.iter()
            .fold((f32::MAX, f32::MIN), |(min, max), &x| (min.min(x), max.max(x)));
        *scale = i8::MAX as f32 / max.abs().max(min.abs());
        azip!((
            b in bucket,
            qb in qbuck,
        ) {
            *qb = (*b * *scale).round() as i8;
        });
    });

    QintArray1 {
        stride,
        scaling,
        arr: qarr,
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

