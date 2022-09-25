mod specular;
mod vec3;

use std::mem::MaybeUninit;
use std::ops::Add;

use ndarray::ArrayViewMut;
use specular::find_specular_points;
use vec3::Vec3;

use numpy::ndarray::{s, Array, ArrayViewD, Axis, Dim};
use numpy::{IntoPyArray, PyArrayDyn};
use pyo3::{pyclass, pymodule, types::PyModule, PyResult, Python};
use rayon::prelude::*;

const SQRT_TAU: f64 = 2.5066282746310002;
use std::f64::consts::FRAC_2_SQRT_PI;

pub const RAD_EARTH: f64 = 6371.; //Approximate radius of earth (km)
const PROJECTION_LENGTH: f64 = RAD_EARTH * SQRT_TAU;

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
    ) -> (&'py PyArrayDyn<u32>, &'py PyArrayDyn<u32>) {
        let speculars = unsafe { speculars.as_array() };
        let out = py.allow_threads(|| find_revisits(speculars));
        (
            out.0.into_dyn().into_pyarray(py),
            out.1.into_dyn().into_pyarray(py),
        )
    }

    Ok(())
}

// TODO: we need array shape or it needs to be passed
// Inputs:
//   Speculars array shape [num_times, num_receivers, num_transmitters, 3]
// Outputs:
//   Number of revisits at each pixel (north, south)
fn find_revisits(
    speculars: ArrayViewD<'_, f64>,
) -> (Array<u32, Dim<[usize; 2]>>, Array<u32, Dim<[usize; 2]>>) {
    let output_shape = [1000, 1000];

    // Initialize counts
    let mut count_s = Array::zeros(output_shape);
    let mut count_n = Array::zeros(output_shape);

    // Iterate over every step of spec_0 and spec_1

    for (spec_0, spec_1) in speculars
        .axis_iter(Axis(0))
        .zip(speculars.axis_iter(Axis(0)).skip(1))
    {
        find_revisits_slice(spec_0, spec_1, &mut count_s, &mut count_n)
    }

    (count_s, count_n)
}

// Stores them in count_n and count_s
// Inputs:
//   Speculars_0 array shape [num_receivers, num_transmitters, 3]
//   Speculars_1 array shape [num_receivers, num_transmitters, 3]
fn find_revisits_slice(
    speculars_0: ArrayViewD<'_, f64>,
    speculars_1: ArrayViewD<'_, f64>,
    count_s: &mut Array<u32, Dim<[usize; 2]>>,
    count_n: &mut Array<u32, Dim<[usize; 2]>>,
) {
    let pixels_per_side = count_s.shape()[0];
    let iter_shape = [speculars_0.shape()[0] * speculars_0.shape()[1], 3];

    let spec_0_iter_array = speculars_0.into_shape(iter_shape).unwrap();
    let spec_1_iter_array = speculars_1.into_shape(iter_shape).unwrap();

    let mut plot_south = |x: usize, y: usize| count_s[[x, y]] += 1;
    let mut plot_north = |x: usize, y: usize| count_n[[x, y]] += 1;

    // Iterate over each segment
    // TODO: make parallel
    spec_0_iter_array
        .axis_iter(Axis(0))
        .zip(spec_1_iter_array.axis_iter(Axis(0)))
        .for_each(|(spec_0, spec_1)| {
            plot_spec_trail(
                Vec3 {
                    x: spec_0[0],
                    y: spec_0[1],
                    z: spec_0[2],
                },
                Vec3 {
                    x: spec_1[0],
                    y: spec_1[1],
                    z: spec_1[2],
                },
                &mut plot_south,
                &mut plot_north,
                pixels_per_side,
            );
        });

    // TODO:
    //   Iterate along every combination of R and T in spec_0 and spec_1 (flat slices (no t dimension))
    //   Perform plotline / count for each one
    //   Closure that allows more complicated counting scheme?
    //   Will probably need to use a stream or something
    //   Or maybe unsafe?
}

fn sphere_to_projection(point: Vec3) -> (f64, f64) {
    let mult = if point.z > 0.0 {
        (2.0 * RAD_EARTH * (RAD_EARTH - point.z)).sqrt()
    } else {
        (2.0 * RAD_EARTH * (RAD_EARTH + point.z)).sqrt()
    };

    if point.y.abs() <= point.x.abs() {
        let a = mult * point.x.signum() / FRAC_2_SQRT_PI;
        let b = mult * point.x.signum() * (FRAC_2_SQRT_PI) * (point.y / point.x).atan();
        (a, b)
    } else {
        let a = mult * point.y.signum() * (FRAC_2_SQRT_PI) * (point.x / point.y).atan();
        let b = mult * point.y.signum() / FRAC_2_SQRT_PI;
        (a, b)
    }
}

fn plot_spec_trail<N, S>(
    spec_0: Vec3,
    spec_1: Vec3,
    plot_south: N,
    plot_north: S,
    pixels_per_side: usize,
) where
    N: FnMut(usize, usize),
    S: FnMut(usize, usize),
{
    // Don't render the line if one of the coordinates is not valid
    if spec_0.x.is_nan() || spec_1.x.is_nan() {
        return;
    }

    //# Find the fraction of the way through the line when it crosses hemispheres
    let t = if spec_0.z == spec_1.z {
        0.5
    } else {
        //From linear interpolation;
        // z = z0 + t*(z1-z0) = 0
        spec_0.z / (spec_0.z - spec_1.z)
    };

    // Projection coordinates
    let (a0, b0) = sphere_to_projection(spec_0);
    let (a1, b1) = sphere_to_projection(spec_1);

    // Check if line crosses hemisphers
    if 0.0 < t && t < 1.0 {
        // Interpolated point where line crosses hemispheres
        let m = spec_0 + &(spec_1 - spec_0) * t;

        // Projection of cross point
        let (am, bm) = sphere_to_projection(m);

        if spec_0.z < 0.0 {
            plot_line(a0, b0, am, bm, pixels_per_side, plot_south);
            plot_line(am, bm, a1, b1, pixels_per_side, plot_north);
        } else {
            plot_line(a0, b0, am, bm, pixels_per_side, plot_north);
            plot_line(am, bm, a1, b1, pixels_per_side, plot_south);
        }
    } else {
        // Line is exclusively in one hemisphere
        if spec_0.z < 0.0 {
            plot_line(a0, b0, a1, b1, pixels_per_side, plot_south);
        } else {
            plot_line(a0, b0, a1, b1, pixels_per_side, plot_north);
        }
    }
}

// Converts a projected point to a pixel location
//   Changing origin from middle to bottom left
//   Scaling by PROJECTION_LENGTH / pixels_per_side
fn map_to_grid(coord: f64, pixels_per_side: usize) -> i32 {
    ((coord / PROJECTION_LENGTH + 0.5) * pixels_per_side as f64) as i32
}

fn plot_line<F>(x0: f64, y0: f64, x1: f64, y1: f64, pixels_per_side: usize, mut plot: F)
where
    F: FnMut(usize, usize),
{
    // Rescale coordinates to grid used for line algorithm
    let x0 = (x0 / PROJECTION_LENGTH + 0.5) * pixels_per_side as f64;
    let y0 = (y0 / PROJECTION_LENGTH + 0.5) * pixels_per_side as f64;
    let x1 = (x1 / PROJECTION_LENGTH + 0.5) * pixels_per_side as f64;
    let y1 = (y1 / PROJECTION_LENGTH + 0.5) * pixels_per_side as f64;

    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();

    let mut x = x0.floor() as isize;
    let mut y = y0.floor() as isize;

    let mut n = 1;
    let mut error = 0.0;

    let x_inc: isize = if dx == 0.0 {
        error = f64::INFINITY;
        0
    } else if x1 > x0 {
        n += x1.floor() as isize - x;
        error += (x0.floor() + 1.0 - x0) * dy;
        1
    } else {
        n += x - x1.floor() as isize;
        error += (x0 - x0.floor()) * dy;
        -1
    };

    let y_inc: isize = if dy == 0.0 {
        error = f64::NEG_INFINITY;
        0
    } else if y1 > y0 {
        n += y1.floor() as isize - y;
        error -= (y0.floor() + 1.0 - y0) * dx;
        1
    } else {
        n += y - y1.floor() as isize;
        error -= (y0 - y0.floor()) * dx;
        -1
    };

    // Skips end point so interpolated segments don't double count
    while n > 1 {
        if 0 <= x && x < pixels_per_side as isize && 0 <= y && y < pixels_per_side as isize {
            plot(x as usize, y as usize);
        }

        if error > 0.0 {
            y += y_inc;
            error -= dx;
        } else {
            x += x_inc;
            error += dy;
        }

        n -= 1;
    }
}

// Bresenham's Line Drawing Algorithm
//   Based on: https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
fn plot_line_wiki<F>(mut x0: i32, mut y0: i32, x1: i32, y1: i32, mut plot: F)
where
    F: FnMut(usize, usize),
{
    let dx = x0.abs_diff(x0) as i32;
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y0.abs_diff(y0) as i32);
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut er = dx + dy;
    let mut first = true;

    loop {
        // Skip plotting the first pixel, so segments don't double count it
        if !first {
            plot(x0 as usize, y0 as usize);
        } else {
            first = false;
        }

        if x0 == x1 && y0 == y1 {
            break;
        }
        let er2 = 2 * er;
        if er2 >= dy {
            if x0 == x1 {
                break;
            } else {
                er += dy;
                x0 += sx;
            }
        }
        if er2 <= dx {
            if y0 == y1 {
                break;
            } else {
                er += dx;
                y0 += sy;
            }
        }
    }
}
