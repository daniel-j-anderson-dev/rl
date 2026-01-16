pub mod cell;
pub mod grid;
pub mod index;
pub mod model;

pub use self::{
    cell::Cell::{self, *},
    grid::Grid,
    index::Index::{self, *},
};

pub const ROW_COUNT: usize = 3;
pub const COLUMN_COUNT: usize = 3;
pub const PLAYER_COUNT: usize = 2;
