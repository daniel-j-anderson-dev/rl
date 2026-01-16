pub mod dataset;
pub mod train;

use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    prelude::Backend,
};

use crate::tictactoe::*;

#[derive(Debug, Config)]
pub struct TicTacToeNetworkConfig {}
impl TicTacToeNetworkConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> TicTacToeNetwork<B> {
        TicTacToeNetwork::init(device)
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
        // -1 opponent, 1 self, 0empty
        let channel_count = 1;
        Self {
            input: LinearConfig::new(channel_count * ROW_COUNT * COLUMN_COUNT, 64).init(device),
            hidden: LinearConfig::new(64, 64).init(device),
            output: LinearConfig::new(64, ROW_COUNT * COLUMN_COUNT).init(device),
            activation: Relu,
        }
    }

    /// - input:
    ///     - shape: [batch_size, row_index, column_index] == [batch_size, 3, 3]
    ///     - the value in each cell represents
    /// - output:
    ///     - shape: [batch_size, serialized_grid_index] == [batch_size, 9]
    ///     - the value in each cell represents a confidence of the best piece placement for a player at a grid index
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, ..] = x.dims();
        let x = x.reshape([batch_size, ROW_COUNT * COLUMN_COUNT]);
        let x = self.input.forward(x);
        let x = self.activation.forward(x);
        let x = self.hidden.forward(x);
        let x = self.activation.forward(x);
        let x = self.output.forward(x);
        x
    }
}
