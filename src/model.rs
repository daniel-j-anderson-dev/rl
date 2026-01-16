use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    prelude::Backend,
};

use crate::tictactoe::{grid::Grid, index::Index};

// input: tictactoe grid
// output: where to place piece (for both x and o)
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
            input: LinearConfig::new(Grid::SHAPE.iter().product(), 64).init(device),
            hidden: LinearConfig::new(64, 64).init(device),
            output: LinearConfig::new(64, Index::ALL.len()).init(device),
            activation: Relu,
        }
    }
}
