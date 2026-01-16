use constcat::concat_slices;
type X = [Index; 3];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Index {
    I00 = 0,
    I01 = 1,
    I02 = 2,
    I10 = 3,
    I11 = 4,
    I12 = 5,
    I20 = 6,
    I21 = 7,
    I22 = 8,
}
impl Index {
    pub const fn from_usize(x: usize) -> Option<Self> {
        Some(match x {
            0 => Self::I00,
            1 => Self::I01,
            2 => Self::I02,
            3 => Self::I10,
            4 => Self::I11,
            5 => Self::I12,
            6 => Self::I20,
            7 => Self::I21,
            8 => Self::I22,
            _ => return None,
        })
    }
    pub const fn from_row_column(row: usize, column: usize) -> Option<Self> {
        Some(match (row, column) {
            (0, 0) => I00,
            (0, 1) => I01,
            (0, 2) => I02,
            (1, 0) => I10,
            (1, 1) => I11,
            (1, 2) => I12,
            (2, 0) => I20,
            (2, 1) => I21,
            (2, 2) => I22,
            _ => return None,
        })
    }
    pub const fn row_column(&self) -> (usize, usize) {
        match self {
            I00 => (0, 0),
            I01 => (0, 1),
            I02 => (0, 2),
            I10 => (1, 0),
            I11 => (1, 1),
            I12 => (1, 2),
            I20 => (2, 0),
            I21 => (2, 1),
            I22 => (2, 2),
        }
    }
    pub const ALL: &[Self] = &[I00, I01, I02, I10, I11, I12, I20, I21, I22];
    pub const ROWS: &[[Self; 3]; 3] = &[[I00, I01, I02], [I10, I11, I12], [I20, I21, I22]];
    pub const COLUMNS: &[[Self; 3]; 3] = &[[I00, I10, I20], [I01, I11, I21], [I02, I12, I22]];
    pub const DIAGONALS: &[[Self; 3]; 2] = &[[I00, I11, I22], [I20, I11, I02]];
    pub const GROUPS: &[[Self; 3]] =
        concat_slices!([[Index; 3]]: Index::ROWS, Index::COLUMNS, Index::DIAGONALS);
}
use Index::*;
#[test]
fn generate_index_variants() {
    use crate::tictactoe::game::grid::Grid;

    let mut k = 0;
    for i in 0..Grid::AREA {
        for j in 0..Grid::AREA {
            println!("I{i}{j} = {k},");
            k += 1;
        }
    }
}

pub fn serialize_index(row: usize, column: usize, width: usize) -> usize {
    row * width + column
}

pub fn deserialize_index(index: usize, width: usize) -> (usize, usize) {
    (index / width, index % width)
}
