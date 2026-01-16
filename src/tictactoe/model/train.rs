use burn::{
    Tensor,
    nn::loss::CrossEntropyLossConfig,
    prelude::Backend,
    tensor::{Int, backend::AutodiffBackend},
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};

use crate::model::{TicTacToeNetwork, dataset::TicTacToeBatch};

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
