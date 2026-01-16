#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    Empty,
    O,
    X,
}
impl Cell {
    pub const VARIANTS: [Self; 3] = [Empty, O, X];
    pub const VARIANT_COUNT: usize = Self::VARIANTS.len();
    pub const fn as_char(&self) -> char {
        match self {
            Empty => ' ',
            O => 'O',
            X => 'X',
        }
    }
}
use Cell::*;
use burn::serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum Player {
    X,
    O,
}
impl core::cmp::PartialEq<Cell> for Player {
    fn eq(&self, other: &Cell) -> bool {
        matches!((self, other), (Player::O, Cell::O) | (Player::X, Cell::X))
    }
}
impl core::cmp::PartialEq<Player> for Cell {
    fn eq(&self, other: &Player) -> bool {
        matches!((other, self), (Player::O, Cell::O) | (Player::X, Cell::X))
    }
}
impl From<Player> for Cell {
    fn from(value: Player) -> Self {
        match value {
            Player::O => O,
            Player::X => X,
        }
    }
}
