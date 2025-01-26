#[macro_export]
macro_rules! multiply_tuple {
    ($scalar:expr, ($($elem:expr),*)) => {
        (
            $(
                $scalar * $elem
            ),*
        )
    };
}
