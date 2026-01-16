use rl::tictactoe::*;

use std::io::Write;

use itertools::{Itertools, repeat_n};

fn main() {
    let mut f = std::fs::File::options()
        .create(true)
        .write(true)
        .truncate(true)
        .open("output.txt")
        .unwrap();

    let valid = repeat_n(Cell::VARIANTS, 9)
        .multi_cartesian_product()
        .filter_map(Grid::from_cells)
        .filter(Grid::is_valid)
        .collect::<Vec<_>>();

    for grid in valid {
        writeln!(f, "{}", grid).unwrap();
    }
}
