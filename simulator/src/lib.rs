use std::ops::{Add, Div, Mul, Sub};

use numpy::ndarray::{s, Array, ArrayD, ArrayViewD, ArrayViewMutD, Axis, Dim};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use rayon::prelude::*;
use roots::find_roots_quartic;

#[pymodule]
fn rust_sim_core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // wrapper of `axpy`
    #[pyfn(m)]
    #[pyo3(name = "axpy")]
    fn axpy_py<'py>(
        py: Python<'py>,
        a: f64,
        x: PyReadonlyArrayDyn<f64>,
        y: PyReadonlyArrayDyn<f64>,
    ) -> &'py PyArrayDyn<f64> {
        let x = x.as_array();
        let y = y.as_array();
        let z = axpy(a, x, y);
        z.into_pyarray(py)
    }

    // wrapper of `mult`
    #[pyfn(m)]
    #[pyo3(name = "mult")]
    fn mult_py(_py: Python<'_>, a: f64, x: &PyArrayDyn<f64>) {
        let x = unsafe { x.as_array_mut() };
        mult(a, x);
    }

    #[pyfn(m)]
    #[pyo3(name = "find_specular_points")]
    fn find_specular_points_py<'py>(
        py: Python<'py>,
        receivers: &PyArrayDyn<f64>,
        transmitters: &PyArrayDyn<f64>,
    ) -> &'py PyArrayDyn<f64> {
        let receivers = unsafe { receivers.as_array() };
        let transmitters = unsafe { transmitters.as_array() };
        find_specular_points(&receivers, &transmitters)
            .into_dyn()
            .into_pyarray(py)
    }

    Ok(())
}

// example using immutable borrows producing a new array
fn axpy(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
    a * &x + &y
}

// example using a mutable borrow to modify an array in-place
fn mult(a: f64, mut x: ArrayViewMutD<'_, f64>) {
    x *= a;
}

fn find_specular_points(
    receivers: &ArrayViewD<f64>,
    transmitters: &ArrayViewD<f64>,
) -> Array<f64, Dim<[usize; 4]>> {
    let output_shape = [
        receivers.shape()[0],
        receivers.shape()[1],
        transmitters.shape()[1],
        3,
    ];

    let mut specular = Array::uninit(output_shape);

    for (i, (recv_slice, tran_slice)) in receivers
        .axis_iter(Axis(0))
        .zip(transmitters.axis_iter(Axis(0)))
        .enumerate()
    {
        let specular_slice = find_specular_points_slice(recv_slice, tran_slice);
        specular_slice
            .slice(s![.., .., ..])
            .assign_to(specular.slice_mut(s![i, .., .., ..]))
    }

    // TODO:
    //   Validate sizes?
    //   Iterate along time axis
    //   Parallel iterate over R and T broadcast zipped with mutable slice into output array
    //   Then convert calc specular point from python, use real quartic solver
    unsafe { specular.assume_init() }
}

fn find_specular_points_slice(
    receivers: ArrayViewD<f64>,
    transmitters: ArrayViewD<f64>,
) -> Array<f64, Dim<[usize; 3]>> {
    let output_shape = [receivers.shape()[0], transmitters.shape()[0], 3];
    let iter_shape = [receivers.shape()[0] * transmitters.shape()[0], 3];

    let mut speculars = Array::uninit(output_shape);

    let recv_iter_array = receivers.insert_axis(Axis(1));
    let recv_iter_array = recv_iter_array.broadcast(output_shape).unwrap();
    let recv_iter_array = recv_iter_array.to_shape(iter_shape).unwrap();

    let trans_iter_array = transmitters.insert_axis(Axis(0));
    let trans_iter_array = trans_iter_array.broadcast(output_shape).unwrap();
    let trans_iter_array = trans_iter_array.to_shape(iter_shape).unwrap();

    let mut spec_iter_array = speculars
        .slice_mut(s![.., .., ..])
        .into_shape(iter_shape)
        .unwrap();

    recv_iter_array
        .axis_iter(Axis(0))
        .into_par_iter()
        .zip(trans_iter_array.axis_iter(Axis(0)).into_par_iter())
        .zip(spec_iter_array.axis_iter_mut(Axis(0)).into_par_iter())
        .for_each(|((rec, trans), mut spec)| {
            // TODO: Calculate specular points
            let calc_spec = find_specular_point_single(
                Vec3 {
                    x: rec[0],
                    y: rec[1],
                    z: rec[2],
                },
                Vec3 {
                    x: trans[0],
                    y: trans[1],
                    z: trans[2],
                },
            );

            spec[0].write(calc_spec.x);
            spec[1].write(calc_spec.y);
            spec[2].write(calc_spec.z);
        });

    unsafe { speculars.assume_init() }
}

#[derive(Clone, Copy)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    fn norm(self) -> f64 {
        f64::sqrt(dot(&self, &self))
    }
}

impl Div<f64> for Vec3 {
    type Output = Vec3;

    fn div(self, rhs: f64) -> Self::Output {
        Vec3 {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl Mul<f64> for &Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: f64) -> Self::Output {
        Vec3 {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Add<Vec3> for Vec3 {
    type Output = Vec3;

    fn add(self, rhs: Vec3) -> Self::Output {
        Vec3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub<Vec3> for Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: Vec3) -> Self::Output {
        Vec3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

pub const RAD_EARTH: f64 = 6371.; //Approximate radius of earth (km)

fn find_specular_point_single(recv: Vec3, trans: Vec3) -> Vec3 {
    let recv = recv / RAD_EARTH;
    let trans = trans / RAD_EARTH;

    let u = dot(&recv, &recv);
    let v = dot(&recv, &trans);
    let w = dot(&trans, &trans);

    let a = 4. * w * (u * w - v * v);
    let b = -4. * (u * w - v * v);
    let c = a + 2. * v + w - 4. * u * w;
    let d = 2. * (u - v);
    let e = u - 1.;

    let roots = find_roots_quartic(a, b, c, d, e);

    let mut q_min = f64::MAX;
    let mut final_spec = None;

    for x in roots.as_ref() {
        let y = (-2. * w * x * x + x + 1.) / (2. * v * w + 1.);

        let mut spec = &recv * (*x) + &trans * y;
        spec = spec / spec.norm();

        // Check that recv dot spec and spec dot trans are positive.
        if dot(&spec, &recv) > 0. && dot(&spec, &trans) > 0. {
            let q = (recv - spec).norm() + (spec - trans).norm();

            // Then check if it has lowest q
            if q < q_min {
                q_min = q;
                final_spec = Some(spec);
            }
        }
    }

    if let Some(final_spec) = final_spec {
        &final_spec * RAD_EARTH
    } else {
        Vec3 {
            x: f64::NAN,
            y: f64::NAN,
            z: f64::NAN,
        }
    }
}

fn dot(a: &Vec3, b: &Vec3) -> f64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

fn find_revisits(spec_0: ArrayViewD<'_, f64>, spec_1: ArrayViewD<'_, f64>) {
    // TODO:
    //   Iterate along every combination of R and T in spec_0 and spec_1 (flat slices (no t dimension))
    //   Perform plotline / count for each one
    //   Closure that allows more complicated counting scheme?
    //   Will probably need to use a stream or something
    //   Or maybe unsafe?
}
