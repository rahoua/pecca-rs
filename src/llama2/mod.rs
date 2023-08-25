
// When single-threaded, blas is fastest but with multithreading in rayon it
// somehow slows things down at least on small models. May be worth investigating.
// extern crate blas_src;

mod quant;
mod model;
mod llama2c;

pub use self::llama2c::gen;
