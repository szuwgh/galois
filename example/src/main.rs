use galois::{DTensor, Shape};
use ndarray::prelude::*;
fn main() {
    let image = image::open("/opt/rsproject/gptgrep/galois/example/grace_hopper.jpg")
        .unwrap()
        .to_rgb8();
    let resized =
        image::imageops::resize(&image, 224, 224, ::image::imageops::FilterType::Triangle);
    let m1: DTensor<f32> = DTensor::with_shape_fn(Shape::from_array([1, 3, 224, 224]), |s| {
        let (i, c, y, x) = s.dims4();
        let mean = [0.485, 0.456, 0.406][c];
        let std = [0.229, 0.224, 0.225][c];
        (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
    });

    let image: Array4<f32> = Array4::from_shape_fn((1, 3, 224, 224), |(i, c, y, x)| {
        let mean = [0.485, 0.456, 0.406][c];
        let std = [0.229, 0.224, 0.225][c];
        (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
    })
    .into();
    println!("{:?}", image.as_slice().unwrap() == m1.as_slice());
}
