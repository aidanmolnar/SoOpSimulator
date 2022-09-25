use std::mem::MaybeUninit;

use super::vec3::Vec3;
use super::RAD_EARTH;
use num::complex::Complex;
use numpy::ndarray::{s, Array, ArrayViewD, ArrayViewMut, Axis, Dim};
use rayon::prelude::*;

// Inputs:
//   Receivers array shape [num_times, num_receivers, 3]
//   Transmitters array shape [num_times, num_transmitters, 3]
// Outputs:
//   Speculars array shape [num_times, num_receivers, num_transmitters, 3]
// Calculates specular reflection points between all pairs of receivers and transmitters at each time step
pub fn find_specular_points(
    receivers: ArrayViewD<f64>,
    transmitters: ArrayViewD<f64>,
) -> Array<f64, Dim<[usize; 4]>> {
    let output_shape = [
        receivers.shape()[0],
        receivers.shape()[1],
        transmitters.shape()[1],
        3,
    ];

    let mut specular = Array::uninit(output_shape);

    // Iterate over each time step
    for (i, (recv_slice, tran_slice)) in receivers
        .axis_iter(Axis(0))
        .zip(transmitters.axis_iter(Axis(0)))
        .enumerate()
    {
        // Calculate and assign all specular points at time step
        find_specular_points_slice(
            recv_slice,
            tran_slice,
            specular.slice_mut(s![i, .., .., ..]),
        );
    }

    unsafe { specular.assume_init() }
}

// Inputs:
//   Receivers array shape [num_receivers, 3]
//   Transmitters array shape [num_transmitters, 3]
//   Speculars slice to write to [num_receivers, num_transmitters, 3]
// Calculates specular reflection points between all pairs of receivers and transmitters
// Does so in parallel using rayon
fn find_specular_points_slice(
    receivers: ArrayViewD<f64>,
    transmitters: ArrayViewD<f64>,
    speculars: ArrayViewMut<MaybeUninit<f64>, Dim<[usize; 3]>>,
) {
    // TODO: validate input sizes

    let output_shape = [receivers.shape()[0], transmitters.shape()[0], 3];

    // Array shape during iteration (iter all pairs of receivers and transmitters)
    let iter_shape = [receivers.shape()[0] * transmitters.shape()[0], 3];

    // Reshape receivers for iterating over all combinations
    let recv_iter_array = receivers.insert_axis(Axis(1)); // Insert transmitter axis
    let recv_iter_array = recv_iter_array.broadcast(output_shape).unwrap(); // Broadcast
    let recv_iter_array = recv_iter_array.to_shape(iter_shape).unwrap(); // Unravel

    // Reshape transmitters for iterating over all combinations
    let trans_iter_array = transmitters.insert_axis(Axis(0)); // Insert receiver axis
    let trans_iter_array = trans_iter_array.broadcast(output_shape).unwrap(); // Broadcast
    let trans_iter_array = trans_iter_array.to_shape(iter_shape).unwrap(); // Unravel

    // Reshape speculars for iterating over all combinations
    let mut spec_iter_array = speculars.into_shape(iter_shape).unwrap();

    // Iterate over every pair of receiver and transmitter
    // Store in specular point array
    recv_iter_array
        .axis_iter(Axis(0))
        .into_par_iter()
        .zip(trans_iter_array.axis_iter(Axis(0)).into_par_iter())
        .zip(spec_iter_array.axis_iter_mut(Axis(0)).into_par_iter())
        .for_each(|((rec, trans), mut spec)| {
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
}

// Analytically calculates the specular reflection point on a spherical earth betwen recv and trans
// Based on: https://www.geometrictools.com/Documentation/SphereReflections.pdf
fn find_specular_point_single(recv: Vec3, trans: Vec3) -> Vec3 {
    let recv = recv / RAD_EARTH;
    let trans = trans / RAD_EARTH;

    let u = recv.dot(&recv);
    let v = recv.dot(&trans);
    let w = trans.dot(&trans);

    let a = 4. * w * (u * w - v * v);
    let b = -4. * (u * w - v * v);
    let c = u + 2. * v + w - 4. * u * w;
    let d = 2. * (u - v);
    let e = u - 1.;

    let roots = quartic_solver(a, b, c, d, e);

    let mut q_min = f64::MAX;
    let mut final_spec = Vec3::nan();

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
    if (recv - final_spec).dot(&final_spec) > 0. && (trans - final_spec).dot(&final_spec) > 0. {
        &final_spec * RAD_EARTH
    } else {
        Vec3::nan()
    }
}

// Returns the real parts of roots of quartic with coefficients a,b,c,d,e
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

// Helper functions, may be useful later
// fn approx_equal(a: f64, b: f64, dp: u8) -> bool {
//     let p = 10f64.powi(-(dp as i32));
//     (a - b).abs() < p
// }

// fn compute_angle(a: Vec3, b: Vec3) -> f64 {
//     let cos = dot(&a, &b) / (a.norm() * b.norm());
//     cos.clamp(-1., 1.).acos()
// }
