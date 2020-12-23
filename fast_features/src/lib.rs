// Contains the code for SC'20 paper "Counterfactual Explanations for Machine
// Learning on Multivariate HPC Time Series Data"

// Authors:
//     Emre Ates (1), Burak Aksar (1), Vitus J. Leung (2), Ayse K. Coskun (1)
// Affiliations:
//     (1) Department of Electrical and Computer Engineering, Boston University
//     (2) Sandia National Laboratories

// This work has been partially funded by Sandia National Laboratories. Sandia
// National Laboratories is a multimission laboratory managed and operated by
// National Technology and Engineering Solutions of Sandia, LLC., a wholly owned
// subsidiary of Honeywell International, Inc., for the U.S. Department of
// Energy’s National Nuclear Security Administration under Contract DENA0003525.

use average::{Estimate, Kurtosis};
// use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
// use numpy::{IntoPyArray, PyArrayDyn};
use numpy::{PyArray, PyArrayDyn};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use quantiles::ckms::CKMS;

const ERROR: f64 = 0.000001;

struct StatHolder {
    kurt: Kurtosis,
    perc: CKMS<f64>,
}

impl StatHolder {
    fn new() -> StatHolder {
        StatHolder {
            kurt: Kurtosis::new(),
            perc: CKMS::new(ERROR),
        }
    }

    fn add(&mut self, a: f64) {
        self.perc.insert(a);
        self.kurt.add(a);
    }
}

#[pyfunction]
/// Generate features for the given numpy array.
fn generate_features(input: &PyArrayDyn<f64>, trim: usize) -> Py<PyArrayDyn<f64>> {
    assert!(input.is_c_contiguous());
    let (n_rows, n_cols) = (input.shape()[0], input.shape()[1]);
    let timeseries = input.as_array();
    let mut holders = Vec::new();
    holders.reserve(n_cols);
    for _ in 0..n_cols {
        holders.push(StatHolder::new());
    }
    let mut result = Vec::new();
    result.reserve(n_cols * 11);
    for row in timeseries.outer_iter().rev().skip(trim).rev().skip(trim) {
        for (idx, &elem) in row.iter().enumerate() {
            holders[idx].add(elem);
        }
    }
    for h in &holders {
        result.push(match h.perc.query(1.0) {
            Some(a) => a.1,
            None => panic!("Perc 1.0 failed"),
        });
    }
    for h in &holders {
        result.push(match h.perc.query(0.0) {
            Some(a) => a.1,
            None => panic!("Perc 0.0 failed"),
        });
    }
    for h in &holders {
        result.push(h.kurt.mean());
    }
    for h in &holders {
        result.push(h.kurt.population_variance());
    }
    for h in &holders {
        result.push(h.kurt.skewness());
    }
    for h in &holders {
        result.push(h.kurt.kurtosis());
    }
    for h in &holders {
        result.push(match h.perc.query(0.05) {
            Some(a) => a.1,
            None => panic!("Perc 0.05 failed"),
        });
    }
    for h in &holders {
        result.push(match h.perc.query(0.25) {
            Some(a) => a.1,
            None => panic!("Perc 0.25 failed"),
        });
    }
    for h in &holders {
        result.push(match h.perc.query(0.5) {
            Some(a) => a.1,
            None => panic!("Perc 0.5 failed"),
        });
    }
    for h in &holders {
        result.push(match h.perc.query(0.75) {
            Some(a) => a.1,
            None => panic!("Perc 0.75 failed"),
        });
    }
    for h in &holders {
        result.push(match h.perc.query(0.95) {
            Some(a) => a.1,
            None => panic!("Perc 0.95 failed"),
        });
    }
    let gil = Python::acquire_gil();
    PyArray::from_vec(gil.python(), result)
        .into_dyn()
        .to_owned()
}

/// This module is a python module implemented in Rust.
/// It is used to extract features from timeseries (passed as numpy arrays)
/// List of features is hard-coded
#[pymodule]
fn fast_features(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(generate_features))?;

    Ok(())
}
