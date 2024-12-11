use crate::tensor::{Tensor, TensorError};
use crate::{LayerNorm, LinearLayer, TransformerBlock};

#[derive(Clone)]
pub struct VisionTransformer {
    patch_embeds: Vec<LinearLayer>,
    pos_embed: Tensor,
    transformer_blocks: Vec<TransformerBlock>,
    norm: LayerNorm,
    head: LinearLayer,
    patch_size: usize,
    image_size: usize,
    in_channels: usize,
    cls_token: Tensor,
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

        let pos_embed = Tensor::ones(vec![1, num_patches + 1, hidden_size]);
        let transformer_blocks = (0..num_layers)
            .map(|_| TransformerBlock::new(hidden_size, num_heads))
            .collect::<Vec<_>>();

        let cls_token = Tensor::ones(vec![1, 1, hidden_size]);

        Self {
            patch_embeds: vec![LinearLayer::new(patch_dim, hidden_size); num_patches],
            pos_embed,
            transformer_blocks,
            norm: LayerNorm::new(hidden_size),
            head: LinearLayer::new(hidden_size, num_classes),
            patch_size,
            image_size,
            in_channels,
            cls_token,
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

        let mut patches: Vec<Vec<f32>> = vec![];
        for b in 0..batch_size {
            for i in 0..n_h {
                for j in 0..n_w {
                    let patch = x.data[b * n_h * n_w * patch_dim
                        + i * n_w * patch_dim
                        + j * patch_dim
                        ..b * n_h * n_w * patch_dim + i * n_w * patch_dim + (j + 1) * patch_dim]
                        .to_vec();

                    let patch = patch.clone();
                    patches.push(patch);
                }
            }
        }
        let patches = Tensor::new(
            patches.into_iter().flatten().collect(),
            vec![batch_size, n_patches, patch_dim],
        )?;
        Ok(patches)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, TensorError> {
        let patches = self.extract_patches(x)?;
        let _batch_size = patches.shape[0];
        let n_patches = patches.shape[1];
        let mut embedded_patches = Vec::<Tensor>::with_capacity(n_patches);
        for i in 0..n_patches {
            let start = i * patches.shape[2];
            let end = (i + 1) * patches.shape[2];
            let patch_tensor =
                Tensor::new(patches.data[start..end].to_vec(), vec![1, patches.shape[2]])?;
            let _embedded_patches = patch_tensor.matmul(&self.patch_embeds[i].weights)?;
            let _embedded_patches = _embedded_patches.add(&self.patch_embeds[i].bias)?;
            embedded_patches.push(_embedded_patches);
        }

        let embedded =
            Tensor::concat(&embedded_patches.iter().collect::<Vec<_>>(), 0)?.unsqueeze(0)?;

        let embedded = Tensor::concat(&[&self.cls_token, &embedded], 1)?;

        let _hidden_dim = self.pos_embed.shape[2];
        let tokens = embedded.add(&self.pos_embed)?;

        let mut x = tokens;
        for block in &self.transformer_blocks {
            x = block.forward(&x)?;
        }

        let x = self.norm.forward(&x)?;
        let x = self.head.forward(&x)?;
        Ok(x)
    }
}
