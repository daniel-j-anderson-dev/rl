pub mod cell;
pub mod grid;
pub mod index;
pub mod model;

pub use self::{
    cell::{
        Cell::{self, *},
        Player,
    },
    grid::Grid,
    index::Index::{self, *},
    model::*,
};

#[test]
fn train_tic_tac_toe_model() {
    use burn::{
        backend::{Autodiff, Wgpu},
        optim::AdamConfig,
    };

    train::<Autodiff<Wgpu>>(
        "./artifacts",
        TrainingConfig::new(TicTacToeNetworkConfig::new(), AdamConfig::new(), Player::X),
        Default::default(),
    )
    .unwrap();
}

pub const ROW_COUNT: usize = 3;
pub const COLUMN_COUNT: usize = 3;
pub const PLAYER_COUNT: usize = 2;
