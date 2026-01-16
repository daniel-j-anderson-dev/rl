use burn::{
    Tensor,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    prelude::Backend,
    tensor::TensorData,
};

use crate::tictactoe::*;

impl Grid {
    /// shape = [2, 3, 3]
    /// meaning: [player_index, row_index, column_index]
    /// player_index == 0 is `Cell::X`
    /// player_index == 1 is `Cell::O`
    pub fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, { Self::RANK }> {
        Tensor::from_data(
            TensorData::new(
                (0..2)
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
                    .collect(),
                Grid::SHAPE,
            ),
            device,
        )
    }
}

#[derive(Debug, Module)]
pub struct TicTacToeNetwork<B: Backend> {
    input: Linear<B>,
    hidden: Linear<B>,
    output: Linear<B>,
    activation: Relu,
}
impl<B: Backend> TicTacToeNetwork<B> {
    pub fn init(device: &B::Device) -> Self {
        Self {
            input: LinearConfig::new(Grid::VOLUME, 64).init(device),
            hidden: LinearConfig::new(64, 64).init(device),
            output: LinearConfig::new(64, Index::ALL.len()).init(device),
            activation: Relu,
        }
    }

    /// - input:
    ///     - shape: [player_index, row_index, column_index]
    ///     - the value in each cell represents
    /// - output:
    ///     - shape: [player_index, row_index, column_index]
    ///     - the value in each cell represents a confidence of the best piece placement for a player at a grid index
    pub fn forward(&self, x: Tensor<B, { Grid::RANK }>) -> Tensor<B, { Grid::RANK }> {
        let x = self.input.forward(x);
        let x = self.activation.forward(x);
        let x = self.hidden.forward(x);
        let x = self.activation.forward(x);
        self.output.forward(x)
    }
}
