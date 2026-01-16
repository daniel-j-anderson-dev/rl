use burn::{
    Tensor,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    prelude::Backend,
};

use crate::tictactoe::{grid::Grid, index::Index};

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
        let x = self.output.forward(x);
        x
    }
}
