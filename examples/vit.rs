use asor::tensor::{Tensor, TensorError};
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

    let test_images = Tensor::zeros(vec![1, 3, 32, 32]);

    // Print shapes for debugging
    println!("Input image shape: {:?}", test_images.shape);
    let logits = vit.forward(&test_images)?;
    let predictions = Softmax::forward(&logits)?;

    println!("ViT predictions shape: {:?}", predictions.shape);
    println!("Class probabilities:");
    for i in 0..predictions.shape[0] {
        print!("Image {}: [", i);
        for j in 0..predictions.shape[1] {
            print!("{:.3} ", predictions.data[i * predictions.shape[1] + j]);
        }
        println!("]");
    }

    Ok(())
}
