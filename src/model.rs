pub mod dataset;

use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    prelude::Backend,
};

use crate::tictactoe::*;

#[derive(Debug, Config)]
pub struct TicTacToeNetworkConfig {
    batch_size: usize,
}
impl TicTacToeNetworkConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> TicTacToeNetwork<B> {
        let Self { batch_size } = self;
        TicTacToeNetwork::init(device, batch_size)
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
    pub fn init(device: &B::Device, batch_size: usize) -> Self {
        Self {
            input: LinearConfig::new(batch_size * PLAYER_COUNT * ROW_COUNT * COLUMN_COUNT, 64)
                .init(device),
            hidden: LinearConfig::new(64, 64).init(device),
            output: LinearConfig::new(64, Index::TOTAL_COUNT * PLAYER_COUNT).init(device),
            activation: Relu,
        }
    }

    /// - input:
    ///     - shape: [batch_size, player_index, row_index, column_index] == [batch_size, 2, 3, 3]
    ///     - the value in each cell represents
    /// - output:
    ///     - shape: [batch_size, player_index, serialized_grid_index]
    ///     - the value in each cell represents a confidence of the best piece placement for a player at a grid index
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch_size, ..] = x.dims();
        // flatten player_index, row_index, column_index
        let x = x.reshape([batch_size, PLAYER_COUNT * ROW_COUNT * COLUMN_COUNT]);
        let x = self.input.forward(x);
        let x = self.activation.forward(x);
        let x = self.hidden.forward(x);
        let x = self.activation.forward(x);
        let x = self.output.forward(x);

        // separate `X`s from `O`s and flatten the last row and col indexes
        x.reshape([batch_size, PLAYER_COUNT, ROW_COUNT * COLUMN_COUNT])
    }
}
// impl<B:Backend> TrainStep<> for TicTacToeNetwork<B> {

// }
