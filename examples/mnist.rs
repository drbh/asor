use asor::mnist_classifier::MnistClassifier;
use asor::tensor::{Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    println!("Testing MNIST Classifier...");

    let test_images = Tensor::zeros(vec![2, 784]);
    let classifier = MnistClassifier::new();
    let predictions = classifier.predict(&test_images)?;
    println!("MNIST predictions: {:?}", predictions);

    Ok(())
}
