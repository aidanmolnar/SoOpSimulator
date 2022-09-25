mod revisit;
mod specular;
mod vec3;

use std::mem::MaybeUninit;
use std::ops::Add;

use revisit::find_revisits;
use specular::find_specular_points;
use vec3::Vec3;

use crossbeam::channel::unbounded;
use ndarray::ArrayViewMut;
use numpy::ndarray::{s, Array, ArrayViewD, Axis, Dim};
use numpy::{IntoPyArray, PyArrayDyn};
use pyo3::{pyclass, pymodule, types::PyModule, PyResult, Python};
use rayon::prelude::*;

pub const RAD_EARTH: f64 = 6371.; //Approximate radius of earth (km)

#[pymodule]
fn rust_sim_core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "find_specular_points")]
    fn find_specular_points_py<'py>(
        py: Python<'py>,
        receivers: &PyArrayDyn<f64>,
        transmitters: &PyArrayDyn<f64>,
    ) -> &'py PyArrayDyn<f64> {
        let receivers = unsafe { receivers.as_array() };
        let transmitters = unsafe { transmitters.as_array() };
        let out = py.allow_threads(|| find_specular_points(receivers, transmitters));
        out.into_dyn().into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "find_revisits")]
    fn find_revisits_py<'py>(
        py: Python<'py>,
        speculars: &PyArrayDyn<f64>,
        grid_size: f64,
    ) -> (&'py PyArrayDyn<u32>, &'py PyArrayDyn<u32>) {
        let speculars = unsafe { speculars.as_array() };
        let out = py.allow_threads(|| find_revisits(speculars, grid_size));
        (
            out.0.into_dyn().into_pyarray(py),
            out.1.into_dyn().into_pyarray(py),
        )
    }

    Ok(())
}
