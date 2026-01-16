use burn::{
    Tensor,
    data::dataloader::batcher::Batcher,
    prelude::Backend,
    tensor::{Int, TensorData},
};

use crate::tictactoe::*;

impl Grid {
    /// shape = [2, 3, 3]
    /// meaning: [player_index, row_index, column_index]
    /// player_index == 0 is `Cell::X`
    /// player_index == 1 is `Cell::O`
    pub fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 3> {
        const SHAPE: [usize; 3] = [PLAYER_COUNT, ROW_COUNT, COLUMN_COUNT];
        let serialized_data = (0..2)
            .flat_map(move |player_index| {
                (0..3).flat_map(move |row_index| {
                    (0..3).map(move |column_index| {
                        match (player_index, self.cells[row_index][column_index]) {
                            (0, X) => 1u8,
                            (1, O) => 1,
                            _ => 0,
                        }
                    })
                })
            })
            .collect();
        Tensor::from_data(TensorData::new(serialized_data, SHAPE), device)
    }
}

pub struct TicTacToeBatch<B: Backend> {
    /// Shape: [batch_size, player_count, row_count, column_count]
    pub grids: Tensor<B, 4>,
    /// Shape: [batch_size, player_count]
    /// targets[b][0] == the index where `X` should be placed
    /// targets[b][1] == the index where `O` should be placed
    pub targets: Tensor<B, 3, Int>,
}

pub struct TicTacToeBatcher {}

impl<B: Backend> Batcher<B, Grid, TicTacToeBatch<B>> for TicTacToeBatcher {
    fn batch(&self, items: Vec<Grid>, device: &<B as Backend>::Device) -> TicTacToeBatch<B> {
        todo!()
    }
}
