use itertools::Itertools;

use crate::tictactoe::{
    cell::Cell::{self, *},
    index::*,
};

pub fn all_valid_grids() -> impl Iterator<Item = Grid> {
    core::iter::repeat_n(Cell::VARIANTS, 9)
        .multi_cartesian_product()
        .filter_map(Grid::from_cells)
        .filter(Grid::is_valid)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    pub(crate) cells: [[Cell; Grid::SIZE]; Grid::SIZE],
}
impl Grid {
    pub const AREA: usize = 9;
    pub const VOLUME: usize = 2 * 3 * 3;
    pub const SIZE: usize = 3;
    pub const RANK: usize = 3;
    pub const SHAPE: [usize; Self::RANK] = [2, Self::SIZE, Self::SIZE];

    pub fn from_cells<A: AsRef<[Cell]>>(cells: A) -> Option<Self> {
        let cells = cells.as_ref();
        Some(Self {
            cells: core::array::from_fn(|i| {
                core::array::from_fn(|j| cells[serialize_index(i, j, Grid::SIZE)])
            }),
        })
    }

    pub const fn cells(&self) -> &[[Cell; Grid::SIZE]; Grid::SIZE] {
        &self.cells
    }

    pub fn is_valid(&self) -> bool {
        let (x_count, o_count, _empty_count) =
            self.cells
                .as_flattened()
                .iter()
                .fold((0usize, 0usize, 0usize), |(x, o, e), cell| match cell {
                    Empty => (x, o, e + 1),
                    O => (x, o + 1, e),
                    X => (x + 1, o, e),
                });

        if x_count != o_count && x_count != o_count + 1 {
            return false;
        }

        match self.winner() {
            Empty => true,
            X => x_count == o_count + 1,
            O => x_count == o_count,
        }
    }

    pub fn winner(&self) -> Cell {
        Index::GROUPS
            .iter()
            .find_map(|&[a, b, c]| {
                let cell = self[a];
                if cell != Empty && self[b] == cell && self[c] == cell {
                    Some(cell)
                } else {
                    None
                }
            })
            .unwrap_or(Empty)
    }
}
impl core::ops::Index<Index> for Grid {
    type Output = Cell;
    fn index(&self, index: Index) -> &Self::Output {
        let (i, j) = index.row_column();
        &self.cells[i][j]
    }
}
impl core::ops::IndexMut<Index> for Grid {
    fn index_mut(&mut self, index: Index) -> &mut Self::Output {
        let (i, j) = index.row_column();
        &mut self.cells[i][j]
    }
}
impl core::fmt::Display for Grid {
    fn fmt(&self, w: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(w, "+-+-+-+")?;
        for i in 0..3 {
            write!(w, "|")?;
            for j in 0..3 {
                let index = Index::from_row_column(i, j).expect("0..3 is inbounds");
                write!(w, "{}|", self[index].as_char(),)?;
            }
            writeln!(w, "\n+-+-+-+")?;
        }
        writeln!(w)?;
        Ok(())
    }
}
