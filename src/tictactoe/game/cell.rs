#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    Empty,
    O,
    X,
}
impl Cell {
    pub const VARIANTS: [Self; 3] = [Empty, O, X];
    pub const fn as_char(&self) -> char {
        match self {
            Empty => ' ',
            O => 'O',
            X => 'X',
        }
    }
}
use Cell::*;
