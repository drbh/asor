use std::vec;

use asor::tensor::{Index, Tensor, TensorError};
use asor::vit::VisionTransformer;
use asor::Softmax;

fn main() -> Result<(), TensorError> {
    println!("\nTesting Vision Transformer...");

    let vit = VisionTransformer::new(
        32, // image_size
        4,  // patch_size
        3,  // in_channels
        10, // num_classes
        96, // hidden_size
        6,  // num_layers
        8,  // num_heads
    );

    let test_images = Tensor::ones(vec![1, 3, 32, 32]);

    println!("Input image shape: {:?}", test_images.shape);
    let output = vit.forward(&test_images)?;
    let logits = output.get(&[Index::Full, Index::Single(0), Index::Full])?;
    println!("ViT logits shape: {:?}", logits.shape);
    let predictions = Softmax::forward(&logits, Some(1))?;
    println!("Predictions");
    predictions.print();
    Ok(())
}
