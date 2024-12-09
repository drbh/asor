use crate::tensor::{Tensor, TensorError};
use crate::{LayerNorm, LinearLayer, TransformerBlock};

#[derive(Clone)]
pub struct VisionTransformer {
    patch_embed: LinearLayer,
    pos_embed: Tensor,
    transformer_blocks: Vec<TransformerBlock>,
    norm: LayerNorm,
    head: LinearLayer,
    patch_size: usize,
    image_size: usize,
    in_channels: usize,
}

impl VisionTransformer {
    pub fn new(
        image_size: usize,
        patch_size: usize,
        in_channels: usize,
        num_classes: usize,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
    ) -> Self {
        let num_patches = (image_size / patch_size) * (image_size / patch_size);
        let patch_dim = in_channels * patch_size * patch_size;

        let pos_embed = Tensor::zeros(vec![1, num_patches + 1, hidden_size]);
        let transformer_blocks = (0..num_layers)
            .map(|_| TransformerBlock::new(hidden_size, num_heads))
            .collect::<Vec<_>>();

        Self {
            patch_embed: LinearLayer::new(patch_dim, hidden_size),
            pos_embed,
            transformer_blocks,
            norm: LayerNorm::new(hidden_size),
            head: LinearLayer::new(hidden_size, num_classes),
            patch_size,
            image_size,
            in_channels,
        }
    }

    fn extract_patches(&self, x: &Tensor) -> Result<Tensor, TensorError> {
        if x.shape.len() != 4 {
            return Err(TensorError::InvalidDimensions(
                "Input must be a 4D tensor [batch_size, channels, height, width]".into(),
            ));
        }

        let batch_size = x.shape[0];
        let n_h = self.image_size / self.patch_size;
        let n_w = self.image_size / self.patch_size;
        let n_patches = n_h * n_w;
        let patch_dim = self.in_channels * self.patch_size * self.patch_size;

        let mut patches = vec![0.0; batch_size * n_patches * patch_dim];
        for b in 0..batch_size {
            for h in 0..n_h {
                for w in 0..n_w {
                    let patch_idx = h * n_w + w;
                    for c in 0..self.in_channels {
                        for ph in 0..self.patch_size {
                            for pw in 0..self.patch_size {
                                let img_h = h * self.patch_size + ph;
                                let img_w = w * self.patch_size + pw;
                                let img_idx = ((b * self.in_channels + c) * self.image_size
                                    + img_h)
                                    * self.image_size
                                    + img_w;
                                let patch_offset = (b * n_patches + patch_idx) * patch_dim
                                    + (c * self.patch_size * self.patch_size
                                        + ph * self.patch_size
                                        + pw);
                                patches[patch_offset] = x.data[img_idx];
                            }
                        }
                    }
                }
            }
        }
        Tensor::new(patches, vec![batch_size, n_patches, patch_dim])
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, TensorError> {
        // extract and embed patches
        let patches = self.extract_patches(x)?;
        let batch_size = patches.shape[0];
        let n_patches = patches.shape[1];

        let mut embedded_patches =
            Vec::with_capacity(batch_size * n_patches * self.pos_embed.shape[2]);
        for i in 0..n_patches {
            let start = i * patches.shape[2];
            let end = (i + 1) * patches.shape[2];
            let patch_tensor =
                Tensor::new(patches.data[start..end].to_vec(), vec![1, patches.shape[2]])?;

            let patch_embed = self.patch_embed.forward(&patch_tensor)?;
            let _embedded_patches = patch_embed.add(&self.pos_embed)?;
            embedded_patches.extend_from_slice(&_embedded_patches.data);
        }

        // reshape to [batch_size, n_patches, hidden_dim]
        let hidden_dim = self.pos_embed.shape[2];

        let embedded = Tensor::new(
            vec![1.0; hidden_dim]
                .into_iter()
                .chain(embedded_patches.into_iter())
                .collect(),
            vec![batch_size, n_patches + 1, hidden_dim],
        )?;

        // position embeddings
        let tokens = embedded.add(&self.pos_embed)?;

        let mut x = tokens;
        for block in &self.transformer_blocks {
            x = block.forward(&x)?;
        }

        let x = self.norm.forward(&x)?;

        let batch_features: Vec<f32> = (0..batch_size)
            .flat_map(|b| {
                let start = b * n_patches * hidden_dim;
                let mut avg = vec![0.0; hidden_dim];
                for p in 0..n_patches {
                    for (h, item) in avg.iter_mut().enumerate().take(hidden_dim) {
                        *item += x.data[start + p * hidden_dim + h] / (n_patches as f32);
                    }
                }
                avg
            })
            .collect();

        // final classification
        self.head
            .forward(&Tensor::new(batch_features, vec![batch_size, hidden_dim])?)
    }
}
