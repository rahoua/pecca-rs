
use rand::{self, Rng};

use ndarray::prelude::*;

use super::transformer::softmax;

type ATy = f32;
const ATY_ZERO: ATy = 0.0;

pub struct Sampler {
    temperature: f32,
}

impl Sampler {
    pub fn new(temperature: f32) -> Sampler {
        Sampler {
            temperature,
        }
    }

    pub fn sample(&self, mut logits: ArrayViewMut1<ATy>) -> Option<usize> {
        if self.temperature == ATY_ZERO {
            argmax(logits)
        } else {
            for logits in logits.iter_mut() {
               *logits /= self.temperature;
            }
            softmax(&mut logits);
            sample(logits)
        }
    }
}

pub fn sample(probabilities: ArrayViewMut1<ATy>) -> Option<usize> {
    let r: ATy = rand::thread_rng().gen();
    let mut cdf = ATY_ZERO;
    probabilities.iter().enumerate().find(|&(_, &prob)| {
        cdf += prob;
        r < cdf
    }).map(|(i, _)| i)
}

pub fn argmax(v: ArrayViewMut1<ATy>) -> Option<usize> {
    v.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
}
