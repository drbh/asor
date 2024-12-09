use crate::tensor::{Tensor, TensorError};
use crate::{LinearLayer, Softmax};

pub struct MnistClassifier {
    linear: LinearLayer,
}

impl MnistClassifier {
    pub fn new() -> Self {
        Self {
            linear: LinearLayer::new(784, 10),
        }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, TensorError> {
        let logits = self.linear.forward(input)?;
        Softmax::forward(&logits)
    }

    pub fn predict(&self, input: &Tensor) -> Result<Vec<usize>, TensorError> {
        let probs = self.forward(input)?;

        Ok(probs
            .data
            .chunks(10)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap()
            })
            .collect())
    }
}

impl Default for MnistClassifier {
    fn default() -> Self {
        Self::new()
    }
}
