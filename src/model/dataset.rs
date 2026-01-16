use std::i32;

use burn::{Tensor, data::dataloader::batcher::Batcher, prelude::Backend, tensor::Bool};

use crate::tictactoe::{grid::GameOver, *};

impl Grid {

    /// Returns a single board where 'perspective' cells are 1.0,
    /// opponent cells are -1.0, and empty are 0.0.
    pub fn perspective_view(&self, perspective: Cell) -> [[f32; 3]; 3] {
        let opponent = if perspective == X { O } else { X };
        core::array::from_fn(|row_index| {
            core::array::from_fn(|column_index| {
                if self.cells[row_index][column_index] == perspective {
                    1.0
                } else if self.cells[row_index][column_index] == opponent {
                    -1.0
                } else {
                    0.0
                }
            })
        })
    }
}
impl Index {
    pub fn one_hot_encode(self) -> [f32; Index::TOTAL_COUNT] {
        const T: f32 = 1.0;
        const F: f32 = 0.0;
        match self {
            I00 => [T, F, F, F, F, F, F, F, F],
            I01 => [F, T, F, F, F, F, F, F, F],
            I02 => [F, F, T, F, F, F, F, F, F],
            I10 => [F, F, F, T, F, F, F, F, F],
            I11 => [F, F, F, F, T, F, F, F, F],
            I12 => [F, F, F, F, F, T, F, F, F],
            I20 => [F, F, F, F, F, F, T, F, F],
            I21 => [F, F, F, F, F, F, F, T, F],
            I22 => [F, F, F, F, F, F, F, F, T],
        }
    }
}

pub struct TicTacToeBatch<B: Backend> {
    /// Shape: [batch_size, row_count, column_count]
    pub inputs: Tensor<B, 3>,
    /// Shape: [batch_size, row_count * player_count]
    pub targets: Tensor<B, 2>,
}

pub struct TicTacToeBatcher {}

impl<B: Backend> Batcher<B, Grid, TicTacToeBatch<B>> for TicTacToeBatcher {
    fn batch(&self, items: Vec<Grid>, device: &<B as Backend>::Device) -> TicTacToeBatch<B> {
        let inputs = items
            .iter()
            .map(|grid| {
                let perspective = if grid.is_x_turn() { X } else { O };
                Tensor::<B, 2, _>::from_data(grid.perspective_view(perspective), device)
            })
            .collect::<Vec<_>>();
        let inputs = Tensor::stack(inputs, 0);

        let targets = items
            .iter()
            .map(|grid| {
                [X, O]
                    .map(|player| grid.find_best_move(player))
                    .map(Index::one_hot_encode)
            })
            .map(|data| Tensor::<_, 2, _>::from_data(data, device))
            .collect::<Vec<_>>();
        let targets = Tensor::stack(targets, 0);

        TicTacToeBatch { inputs, targets }
    }
}
