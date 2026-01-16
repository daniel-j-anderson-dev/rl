use burn::{Tensor, data::dataloader::batcher::Batcher, prelude::Backend};

use crate::tictactoe::{grid::GameOver, *};

impl Grid {
    /// minimax: https://en.wikipedia.org/wiki/Minimax#Combinatorial_game_theory
    pub fn find_best_move(&self, player: Cell) -> Index {
        assert!(player != Empty);

        let mut best_score = if player == X { i32::MIN } else { i32::MAX };
        let mut best_move = I00;
        for &i in Index::ALL.iter().filter(|&&i| self[i] == Empty) {
            let mut next_grid = *self;
            next_grid[i] = player;

            let score = next_grid.minimax(player == O);

            if player == X {
                if score > best_score {
                    best_score = score;
                    best_move = i;
                }
            } else if score < best_score {
                best_score = score;
                best_move = i;
            }
        }
        best_move
    }

    fn minimax(&self, is_x_turn: bool) -> i32 {
        match self.game_over() {
            Some(GameOver::X) => 1,
            Some(GameOver::O) => -1,
            Some(GameOver::Draw) => 0,
            None => {
                let possible_moves = Index::ALL.iter().filter(|&&i| self[i] == Empty);
                if is_x_turn {
                    possible_moves
                        .map(|&i| {
                            let mut next = *self;
                            next[i] = X;
                            next.minimax(false)
                        })
                        .max()
                } else {
                    possible_moves
                        .map(|&i| {
                            let mut next = *self;
                            next[i] = O;
                            next.minimax(true)
                        })
                        .min()
                }
                .expect("non-terminal position must have legal moves")
            }
        }
    }

    /// Returns a single board where 'perspective' cells are 1.0,
    /// opponent cells are -1.0, and empty are 0.0.
    pub fn perspective_view(&self, perspective: Cell) -> [[f32; 3]; 3] {
        let opponent = if perspective == X { O } else { X };
        core::array::from_fn(|row_index| {
            core::array::from_fn(|column_index| {
                if self.cells[row_index][column_index] == perspective {
                    1.0
                } else if self.cells[row_index][column_index] == opponent {
                    -1.0
                } else {
                    0.0
                }
            })
        })
    }
}
impl Index {
    pub fn one_hot_encode(self) -> [f32; Index::TOTAL_COUNT] {
        const T: f32 = 1.0;
        const F: f32 = 0.0;
        match self {
            I00 => [T, F, F, F, F, F, F, F, F],
            I01 => [F, T, F, F, F, F, F, F, F],
            I02 => [F, F, T, F, F, F, F, F, F],
            I10 => [F, F, F, T, F, F, F, F, F],
            I11 => [F, F, F, F, T, F, F, F, F],
            I12 => [F, F, F, F, F, T, F, F, F],
            I20 => [F, F, F, F, F, F, T, F, F],
            I21 => [F, F, F, F, F, F, F, T, F],
            I22 => [F, F, F, F, F, F, F, F, T],
        }
    }
}

pub struct TicTacToeBatch<B: Backend> {
    /// Shape: [batch_size, row_count, column_count]
    pub inputs: Tensor<B, 3>,
    /// Shape: [batch_size, row_count * player_count]
    pub targets: Tensor<B, 2>,
}

pub struct TicTacToeBatcher {}

impl<B: Backend> Batcher<B, Grid, TicTacToeBatch<B>> for TicTacToeBatcher {
    fn batch(&self, items: Vec<Grid>, device: &<B as Backend>::Device) -> TicTacToeBatch<B> {
        let (inputs, targets) = items.iter().fold(
            (Vec::new(), Vec::new()),
            |(mut items, mut targets), grid| {
                items.push(Tensor::<B, 2, _>::from_data(
                    grid.perspective_view(grid.current_turn()),
                    device,
                ));
                targets.push(Tensor::<_, 1, _>::from_data(
                    grid.find_best_move(grid.current_turn()).one_hot_encode(),
                    device,
                ));
                (items, targets)
            },
        );
        let inputs = Tensor::stack(inputs, 0);
        let targets = Tensor::stack(targets, 0);

        TicTacToeBatch { inputs, targets }
    }
}
