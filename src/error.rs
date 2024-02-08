use crate::Dim;
use crate::Shape;
use std::io;
use std::io::Error as IOError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ShapeErrorKind {
    #[error("operands could not be broadcast together")]
    IncompatibleShape = 1,
}

pub type GResult<T> = Result<T, GError>;

#[derive(Error, Debug)]
pub enum GError {
    #[error("shape error:{0}")]
    ShapeError(ShapeErrorKind),
    #[error("Unexpected io: {0}, {1}")]
    UnexpectIO(String, io::Error),
    #[error("Unexpected: {0}")]
    Unexpected(String),
    // === Dimension Index Errors ===
    #[error("{op}: dimension index {dim} out of range for shape {shape:?}")]
    DimOutOfRange {
        shape: Shape,
        dim: usize,
        op: &'static str,
    },
    // === Shape Errors ===
    #[error("unexpected rank, expected: {expected}, got: {got} ({shape:?})")]
    UnexpectedNumberOfDims {
        expected: usize,
        got: usize,
        shape: Shape,
    },

    #[error("shape mismatch in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    ShapeMismatchBinaryOp {
        lhs: Shape,
        rhs: Shape,
        op: &'static str,
    },

    #[error("shape mismatch in lhs: {lhs_l:?}, rhs: {rhs_l:?}")]
    MatMulUnexpectedStriding {
        lhs_l: Dim,
        rhs_l: Dim,
        bmnk: (usize, usize, usize, usize),
        msg: &'static str,
    },

    #[error("narrow invalid args {msg}: {shape:?}, dim: {dim}, start: {start}, len:{len}")]
    NarrowInvalidArgs {
        shape: Shape,
        dim: usize,
        start: usize,
        len: usize,
        msg: &'static str,
    },
}

impl From<&str> for GError {
    fn from(e: &str) -> Self {
        GError::Unexpected(e.to_string())
    }
}

impl From<(&str, io::Error)> for GError {
    fn from(e: (&str, io::Error)) -> Self {
        GError::UnexpectIO(e.0.to_string(), e.1)
    }
}

impl From<String> for GError {
    fn from(e: String) -> Self {
        GError::Unexpected(e)
    }
}

impl From<IOError> for GError {
    fn from(e: IOError) -> Self {
        GError::Unexpected(e.to_string())
    }
}

impl From<GError> for String {
    fn from(e: GError) -> Self {
        format!("{}", e)
    }
}
