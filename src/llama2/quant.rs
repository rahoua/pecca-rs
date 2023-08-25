
use ndarray::prelude::*;
use ndarray::{par_azip, RemoveAxis};

// A quantized array constructed with a generic dimension
#[derive(Debug, Clone)]
pub struct QintArray<T, D> where D: Dimension {
    pub scaling: Array<f32, D::Smaller>,
    pub arr: Array<T, D>
}

impl<T, D> QintArray<T, D> where D: Dimension {
    pub fn len(&self) -> usize {
        self.arr.len()
    }

    pub fn view(&self) -> QintArrayView<'_, T, D> {
        QintArrayView {
            scaling: self.scaling.view(),
            arr: self.arr.view(),
        }
    }
}

impl<T, D> QintArray<T, D>
where
    D: RemoveAxis,
    <D as ndarray::Dimension>::Smaller: RemoveAxis
{
    pub fn index_axis(&self, axis: Axis, index: usize) -> QintArrayView<'_, T, D::Smaller> {
        QintArrayView {
            scaling: self.scaling.index_axis(axis, index),
            arr: self.arr.index_axis(axis, index),
        }
    }
}

// A quantized array constructed with a generic dimension
#[derive(Debug, Clone)]
pub struct QintArrayView<'a, T, D> where D: Dimension {
    pub scaling: ArrayView<'a, f32, D::Smaller>,
    pub arr: ArrayView<'a, T, D>
}

// A quantized array constructed from an Array1<f32>
pub type QintArray1<T> = QintArray<T, Ix1>;

pub type QintArrayView1<'a, T> = QintArrayView<'a, T, Ix1>;

impl From<ArrayView1<'_, f32>> for QintArray1<i8> {
    fn from(fpa: ArrayView1<f32>) -> Self {
        quant_i8(fpa)
    }
}

impl QintArrayView1<'_, i8> {
    // almost all execution time is spent here, worth finicking with
	fn dot_deq(&self, other: &Self) -> f32 {
		debug_assert_eq!(self.arr.len(), other.arr.len());
		let len = self.arr.len().min(other.arr.len());
		let xs = &self.arr.as_slice().unwrap();
		let ys = &other.arr.as_slice().unwrap();
		let mut sum = 0;
        for n in 0..len {
            sum += xs[n] as i32 * ys[n] as i32;
        }
		sum as f32 / (self.scaling[[]] * other.scaling[[]])
    }

    pub fn to_f32(self) -> Array1<f32> {
        dequant_i8(self)
    }
}

pub type QintArray2<T> = QintArray<T, Ix2>;

pub type QintArrayView2<'a, T> = QintArrayView<'a, T, Ix2>;

impl From<ArrayView2<'_, f32>> for QintArray2<i8> {
    fn from(fpa: ArrayView2<f32>) -> Self {
        let mut qarr2 = Array2::zeros(fpa.dim());
        let mut scaling = Array1::zeros(fpa.dim().0);
        azip!((
            row in fpa.outer_iter(),
            qrow2 in qarr2.outer_iter_mut(),
            qscale in scaling.outer_iter_mut(),
        ) {
            let qarr1 = QintArray1::from(row);
            qarr1.arr.move_into(qrow2);
            qarr1.scaling.move_into(qscale);
        });
        QintArray2 {
            arr: qarr2,
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
                arr: row,
                scaling: scale,
            };
            *dot = qv1.dot_deq(&x);
        });
        res
    }
}

pub type QintArray3<T> = QintArray<T, Ix3>;

impl From<ArrayView3<'_, f32>> for QintArray3<i8> {
    fn from(fpa: ArrayView3<f32>) -> Self {
        let mut qarr3 = Array3::zeros(fpa.dim());
        let mut scaling = Array2::zeros((fpa.dim().0, fpa.dim().1));
        azip!((
            row in fpa.outer_iter(),
            qrow3 in qarr3.outer_iter_mut(),
            qscale in scaling.outer_iter_mut(),
        ) {
            let qarr2 = QintArray2::from(row);
            qarr2.arr.move_into(qrow3);
            qarr2.scaling.move_into(qscale);
        });
        QintArray3 {
            arr: qarr3,
            scaling: scaling,
        }
    }
}

fn quant_i8(fpa: ArrayView1<f32>) -> QintArray1<i8> {
    let (min, max) = fpa.iter().fold((f32::MAX, f32::MIN), |(min, max), &x| (min.min(x), max.max(x)));
    let midpoint = (max + min) / 2.0;
    // let scaling = i8::MAX as f32 / (max.abs() - midpoint);
    let scaling = i8::MAX as f32 / max.abs().max(min.abs());
    QintArray1 {
        scaling: Array::from_elem((), scaling),
        // midpoint,
        // arr: fpa.map(|x| ((x - midpoint) * scaling).round() as i8),
        arr: fpa.map(|x| (x * scaling).round() as i8),
    }
}

fn dequant_i8(ia: QintArrayView1<i8>) -> Array1<f32> {
    ia.arr.mapv(|n| n as f32 / ia.scaling[[]]) // + ia.midpoint)
}

fn loss(fp1: ArrayView1<f32>, fp2: ArrayView1<f32>) -> f32 {
    std::iter::zip(fp1, fp2).map(|(x1, x2)| x1 - x2).sum::<f32>() / fp1.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::random;

    #[test]
    fn it_quantizes() {
        let fpa = Array::from_vec(vec![-0.9, -0.35, -0.1, 0.2, 0.6]); 
        println!("fpa: {:?}", quant_i8(fpa.view()));
        // assert_eq!(quant_i8(fpa.view()).arr.to_vec(), vec![-127, -34, 8, 59, 127]);
        assert_eq!(quant_i8(fpa.view()).arr.to_vec(), vec![-127, -49, -14, 28, 85]);
    }

    #[test]
    fn it_has_little_loss() {
        let fpa = QintArray1 {
            // scaling: 169.33333,
            scaling: 141.11111,
            // midpoint: -0.15,
            arr: Array::from_vec(vec![-127, -49, -14, 28, 85]),
        };
        let dq_arr = dequant_i8(fpa);
        let lo = loss(dq_arr.view(), Array::from_vec(vec![-0.9, -0.35, -0.1, 0.2, 0.6]).view());
        println!("dq arr: {}, loss: {}", dq_arr, lo);
        if lo > 0.001 {
            panic!("high loss: {}", lo);
        }
    }

    #[test]
    fn it_quantizes_many_with_little_loss() {
        let mut total_loss = 0.0;
        let mut rng = rand::thread_rng();
        for n in 1000..2000 {
            let fpa = Array::from_vec((0..n).map(|_| 4.0*random::<f32>()).collect());
            let q = quant_i8(fpa.view());
            let scaling = q.scaling;
            let l = loss(fpa.view(), dequant_i8(q).view());
            if l > 0.001 {
                println!("Scaling: {}", scaling);
                panic!("Loss over 0.1%: {}", l);
            }
            total_loss += l;
        }
        println!("Average loss: {}", total_loss / 1000.0);
    }

    #[test]
    fn it_dots() {
        let fpa1 = Array::from_vec(vec![-0.9, -0.35, -0.1, 0.2, 0.6]); 
        let fpa2 = Array::from_vec(vec![-0.75, -0.2, 0.05, 0.4, 0.9]); 
        let fdot = fpa1.dot(&fpa2);
        let i1 = quant_i8(fpa1.view());
        let i2 = quant_i8(fpa2.view());
        let idot = i1.dot_deq(&i2);
        assert!((fdot - idot) < 0.001);
    }
}

