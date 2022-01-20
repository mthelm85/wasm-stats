mod utils;

use nalgebra::{ DMatrix };
use wasm_bindgen::prelude::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

// need to add method to do regression via normal equations and LU decomposition

#[wasm_bindgen]
pub fn lin_reg_qr(data: &JsValue, idxs: &JsValue) -> Vec<f64> {
    let data: Vec<Vec<f64>> = data.into_serde().unwrap();
    let idxs: Vec<usize> = idxs.into_serde().unwrap();

    let data: Vec<Vec<f64>> = data.iter()
        .map(|inner| idxs.iter().map(|&i| inner[i]).collect())
        .collect();

    let y = DMatrix::from_vec(data.len(), 1, data.iter()
        .map(|r| r[0])
        .collect());

    let x = DMatrix::from_row_slice(data.len(), data[0].len(), &data.iter()
        .map(|r| [&[1.0], &r[1..data[0].len()]].concat())
        .flatten()
        .collect::<Vec<f64>>()[..]);

    let qr = x.qr();
    let r_inv = qr.r().try_inverse().unwrap();
    let bs = r_inv * qr.q().transpose() * y;
    bs.data.as_vec().to_vec()
}