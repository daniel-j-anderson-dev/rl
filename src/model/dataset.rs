use burn::{Tensor, prelude::Backend, tensor::TensorData};

use crate::tictactoe::*;

impl Grid {
    /// shape = [1, 2, 3, 3]
    /// meaning: [batch_index, player_index, row_index, column_index]
    /// player_index == 0 is `Cell::X`
    /// player_index == 1 is `Cell::O`
    pub fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, { Self::RANK }> {
        let batch_size = 1;
        let player_count = 2;
        let row_count = 3;
        let column_count = 3;
        let shape = [batch_size, player_count, row_count, column_count];
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
        Tensor::from_data(TensorData::new(serialized_data, shape), device)
    }
}
