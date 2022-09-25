use std::ops::{Add, Div, Mul, Sub};

use num::complex::Complex;
use numpy::ndarray::{s, Array, ArrayViewD, Axis, Dim};
use numpy::{IntoPyArray, PyArrayDyn};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use rayon::prelude::*;

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
        let out = py.allow_threads(|| find_specular_points(&receivers, &transmitters));
        out.into_dyn().into_pyarray(py)
    }

    Ok(())
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

#[derive(Clone, Copy, Debug)]
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
    let c = u + 2. * v + w - 4. * u * w;
    let d = 2. * (u - v);
    let e = u - 1.;

    let roots = quartic_solver(a, b, c, d, e);

    let mut q_min = f64::MAX;
    let mut final_spec = Vec3 {
        x: f64::NAN,
        y: f64::NAN,
        z: f64::NAN,
    };

    for y in roots.as_ref() {
        let x = (-2. * w * y * y + y + 1.) / (2. * v * y + 1.);

        let mut spec = &recv * x + &trans * (*y);
        spec = spec / spec.norm();

        let q = (recv - spec).norm() + (spec - trans).norm();

        // Then check if it has lowest q
        if q < q_min {
            q_min = q;
            final_spec = spec;
        }
    }

    // Incidence angles
    //let recv_incidence_angle = compute_angle(recv - final_spec, final_spec);
    //let trans_incidence_angle = compute_angle(trans - final_spec, final_spec);

    // Check that incidence angle is less than 90 degrees
    if dot(&(recv - final_spec), &final_spec) > 0. && dot(&(trans - final_spec), &final_spec) > 0. {
        &final_spec * RAD_EARTH
    } else {
        Vec3 {
            x: f64::NAN,
            y: f64::NAN,
            z: f64::NAN,
        }
    }

    // Check that recv dot spec and spec dot trans are positive.
    // if dot(&final_spec, &recv) > 0. && dot(&final_spec, &trans) > 0. {
    //
    // } else {
    //     Vec3 {
    //         x: f64::NAN,
    //         y: f64::NAN,
    //         z: f64::NAN,
    //     }
    // }
}

fn approx_equal(a: f64, b: f64, dp: u8) -> bool {
    let p = 10f64.powi(-(dp as i32));
    (a - b).abs() < p
}

fn compute_angle(a: Vec3, b: Vec3) -> f64 {
    let cos = dot(&a, &b) / (a.norm() * b.norm());
    cos.clamp(-1., 1.).acos()
}

fn dot(a: &Vec3, b: &Vec3) -> f64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

fn quartic_solver(a: f64, b: f64, c: f64, d: f64, e: f64) -> [f64; 4] {
    let b_2 = b * b;
    let a_2 = a * a;
    let alpha = (-3. / 8.) * b_2 / a_2 + c / a;
    let beta = (1. / 8.) * b_2 * b / (a_2 * a) - b * c / (2. * a_2) + d / a;
    let gam = (-3. / 256.) * b_2 * b_2 / (a_2 * a_2) + c * b_2 / (16. * a_2 * a)
        - b * d / (4. * a_2)
        + e / a;

    if beta == 0.0 {
        let mut roots = [0.0; 4];

        let inner = Complex {
            re: alpha * alpha - 4. * gam,
            im: 0.0,
        }
        .sqrt();

        roots[0] = (-b / (4. * a) + ((-alpha + inner) / 2.).sqrt()).re;
        roots[1] = (-b / (4. * a) + ((-alpha - inner) / 2.).sqrt()).re;
        roots[2] = (-b / (4. * a) - ((-alpha + inner) / 2.).sqrt()).re;
        roots[3] = (-b / (4. * a) - ((-alpha - inner) / 2.).sqrt()).re;

        return roots;
    }

    let p = -(1. / 12.) * alpha * alpha - gam;
    let q = -(1. / 108.) * alpha * alpha * alpha + alpha * gam / 3. - beta * beta / 8.;
    let r = -q / 2.
        + Complex {
            re: q * q / 4. + p * p * p / 27.,
            im: 0.0,
        }
        .sqrt();

    let u = r.cbrt();

    let y = if u == (Complex { re: 0.0, im: 0.0 }) {
        Complex {
            re: -(5. / 6.) * alpha - q.cbrt(),
            im: 0.,
        }
    } else {
        -(5. / 6.) * alpha + u - p / (3. * u)
    };

    let w = (alpha + 2. * y).sqrt();

    let mut roots = [0.0; 4];

    roots[0] = (-b / (4. * a) + (-w - (-(3. * alpha + 2. * y - 2. * beta / w)).sqrt()) / 2.).re;
    roots[1] = (-b / (4. * a) + (w - (-(3. * alpha + 2. * y + 2. * beta / w)).sqrt()) / 2.).re;
    roots[2] = (-b / (4. * a) + (-w + (-(3. * alpha + 2. * y - 2. * beta / w)).sqrt()) / 2.).re;
    roots[3] = (-b / (4. * a) + (w + (-(3. * alpha + 2. * y + 2. * beta / w)).sqrt()) / 2.).re;

    roots
}

/* fn find_revisits(spec_0: ArrayViewD<'_, f64>, spec_1: ArrayViewD<'_, f64>) {
    // TODO:
    //   Iterate along every combination of R and T in spec_0 and spec_1 (flat slices (no t dimension))
    //   Perform plotline / count for each one
    //   Closure that allows more complicated counting scheme?
    //   Will probably need to use a stream or something
    //   Or maybe unsafe?
}
 */
