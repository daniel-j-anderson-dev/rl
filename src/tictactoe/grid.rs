use itertools::Itertools;

use crate::tictactoe::{
    COLUMN_COUNT, ROW_COUNT,
    cell::Cell::{self, *},
    index::*,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    pub(crate) cells: [[Cell; ROW_COUNT]; COLUMN_COUNT],
}
impl Grid {
    pub const AREA: usize = 9;
    pub const SIZE: usize = 3;

    pub fn all() -> impl Iterator<Item = Grid> {
        core::iter::repeat_n(Cell::VARIANTS, Index::TOTAL_COUNT)
            .multi_cartesian_product()
            .filter_map(Self::from_cells)
    }

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

    pub fn count_cells(&self) -> (usize, usize, usize) {
        self.cells.as_flattened().iter().fold(
            (0usize, 0usize, 0usize),
            |(x_count, o_count, empty_count), cell| match cell {
                Empty => (x_count, o_count, empty_count + 1),
                O => (x_count, o_count + 1, empty_count),
                X => (x_count + 1, o_count, empty_count),
            },
        )
    }

    pub fn is_valid(&self) -> bool {
        let (x_count, o_count, _empty_count) = self.count_cells();

        if x_count != o_count && x_count != o_count + 1 {
            return false;
        }

        match self.game_over() {
            Some(GameOver::X) => x_count == o_count + 1,
            Some(GameOver::O) => x_count == o_count,
            _ => true,
        }
    }

    pub fn is_x_turn(&self) -> bool {
        let (x_count, o_count, _empty_count) = self.count_cells();
        x_count != o_count && x_count != o_count + 1
    }

    pub fn current_turn(&self) -> Cell {
        if self.is_x_turn() { X } else { O }
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
impl Grid {
    pub fn game_over(&self) -> Option<GameOver> {
        let winner = Index::GROUPS.iter().find_map(|&[a, b, c]| {
            let cell = self[a];
            if cell != Empty && self[b] == cell && self[c] == cell {
                Some(cell)
            } else {
                None
            }
        });

        match winner {
            Some(X) => Some(GameOver::X),
            Some(O) => Some(GameOver::O),
            None if self.cells.as_flattened().iter().all(|&c| c != Empty) => Some(GameOver::Draw),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameOver {
    X,
    O,
    Draw,
}
