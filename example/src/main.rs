use galois::{arr, cube, mat, Shape, Tensor};

fn main() {
    let m1 = mat(&[
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
    ]);
    let m2 = arr(&[1.0, 2.0, 3.0]);
    let m3 = &m1 + &m2;

    let c = cube(&[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]);
    let a = arr(&[1.0, 2.0]);

    let c1 = &c + &a;
    let c2 = &c * &a;

    let m1 = Tensor::<f64>::with_shape(
        vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        Shape::from_slice([2, 3]),
    );
    let m2 = Tensor::<f64>::with_shape(vec![1.0, 2.0, 3.0], Shape::from_slice([3]));
    let m3 = &m1 + &m2;
    let m4 = &m1 * &m2;
}
