pub mod cell;
pub mod grid;
pub mod index;

use burn::{
    prelude::*,
    tensor::{Shape, Tensor, TensorData, backend::Backend},
};

use crate::tictactoe::{
    cell::Cell::{self, *},
    grid::Grid,
    index::Index,
};

/// shape = [2, 3, 3]
/// meaning: [player_index, row_index, column_index]
/// player_index == 0 is `Cell::X`
/// player_index == 1 is `Cell::O`
pub fn encode_grid<B: Backend>(device: &B::Device, grid: &Grid) -> Tensor<B, 3> {
    Tensor::from_data(
        TensorData::new(
            (0..2)
                .flat_map(move |player_index| {
                    (0..3).flat_map(move |row_index| {
                        (0..3).map(move |column_index| {
                            match Index::from_row_column(row_index, column_index) {
                                Some(cell_index) => match (player_index, grid[cell_index]) {
                                    (0, X) => 1u8,
                                    (1, O) => 1,
                                    _ => 0,
                                },
                                None => unreachable!("(0..3, 0..3) are all valid indexes"),
                            }
                        })
                    })
                })
                .collect(),
            [2, 3, 3],
        ),
        device,
    )
}

#[test]
fn f() {
    use burn::{backend::Wgpu, prelude::Backend};
    let device = <Wgpu as Backend>::Device::default();
    let t = encode_grid::<Wgpu>(&device, &grid::all_valid_grids().next().unwrap());
}
