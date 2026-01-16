use burn::{
    Tensor,
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::InMemDataset},
    module::Module,
    nn::loss::CrossEntropyLossConfig,
    optim::AdamConfig,
    prelude::Backend,
    record::CompactRecorder,
    tensor::{Int, backend::AutodiffBackend},
    train::{
        ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
        metric::{AccuracyMetric, LossMetric},
    },
};

use crate::tictactoe::*;

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: TicTacToeNetworkConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
    pub player: Player,
}

pub fn train<B: AutodiffBackend>(
    artifact_path: &str,
    config: TrainingConfig,
    device: B::Device,
) -> Result<(), Box<dyn core::error::Error>> {
    config.save(format!("{}/config.json", artifact_path))?;

    B::seed(&device, config.seed);

    let dataset = Grid::all()
        .filter_map(|grid| {
            grid.is_valid()
                .then(|| (grid, grid.find_best_move(config.player)))
        })
        .collect::<Vec<_>>();

    let batcher = TicTacToeBatcher;

    let data_loader_train = DataLoaderBuilder::<B, _, _>::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(42)
        .num_workers(config.num_workers)
        .build(InMemDataset::new(dataset.clone()));

    let data_loader_validate = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .build(InMemDataset::new(dataset));

    let training = SupervisedTraining::new(artifact_path, data_loader_train, data_loader_validate)
        .metrics((AccuracyMetric::new(), LossMetric::new()))
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary();

    let model = TicTacToeNetworkConfig::new().init(&device);
    let learner = Learner::new(model, config.optimizer.init(), config.learning_rate);
    let result = training.launch(learner);

    result
        .model
        .save_file(format!("{}/model", artifact_path), &CompactRecorder::new())?;

    Ok(())
}

impl<B: Backend> TicTacToeNetwork<B> {
    pub fn forward_track_cross_entropy_loss(
        &self,
        input: Tensor<B, 3>,
        target: Tensor<B, 2>,
    ) -> (Tensor<B, 1>, Tensor<B, 2>, Tensor<B, 1, Int>) {
        // feed the batch through the network
        let output = self.forward(input);

        // convert targets from one-hot-encoding
        let targets = target.argmax(1).reshape([output.dims()[0]]);

        // calculate cross entropy loss
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        (loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep for TicTacToeNetwork<B> {
    type Input = TicTacToeBatch<B>;
    type Output = ClassificationOutput<B>;
    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let (loss, output, targets) =
            self.forward_track_cross_entropy_loss(batch.inputs, batch.targets);
        let output = ClassificationOutput::new(loss.clone(), output, targets);
        TrainOutput::new(self, loss.backward(), output)
    }
}

impl<B: Backend> InferenceStep for TicTacToeNetwork<B> {
    type Input = TicTacToeBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: Self::Input) -> Self::Output {
        let (loss, output, targets) =
            self.forward_track_cross_entropy_loss(batch.inputs, batch.targets);
        ClassificationOutput::new(loss, output, targets)
    }
}
