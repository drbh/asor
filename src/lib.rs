// #![no_std]

pub mod mnist_classifier;
pub mod tensor;
pub mod vit;

use tensor::{Tensor, TensorError};

#[derive(Clone)]
pub struct LinearLayer {
    weights: Tensor,
    bias: Tensor,
}

impl LinearLayer {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            weights: Tensor::ones(vec![in_features, out_features]),
            bias: Tensor::ones(vec![1, out_features]),
        }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, TensorError> {
        if input.shape.len() != 2 {
            return Err(TensorError::InvalidDimensions(
                "Linear layer requires 2D input [batch_size, features]".into(),
            ));
        }

        let output = input.matmul(&self.weights)?;
        let mut result = output.data.clone();

        for i in 0..output.shape[0] {
            for j in 0..output.shape[1] {
                result[i * output.shape[1] + j] += self.bias.data[j];
            }
        }

        Tensor::new(result, output.shape.clone())
    }
}

#[derive(Clone)]
pub struct LayerNorm {
    gamma: Tensor,
    beta: Tensor,
    eps: f32,
}

impl LayerNorm {
    pub fn new(size: usize) -> Self {
        Self {
            gamma: Tensor::new(vec![1.0; size], vec![1, size]).unwrap(),
            beta: Tensor::zeros(vec![1, size]),
            eps: 1e-5,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, TensorError> {
        let batch_size = input.shape[0];
        let hidden_size = input.shape[1];
        let mut result = vec![0.0; input.data.len()];

        for i in 0..batch_size {
            let start = i * hidden_size;
            let end = start + hidden_size;
            let row = &input.data[start..end];

            let mean: f32 = row.iter().sum::<f32>() / hidden_size as f32;
            let var: f32 =
                row.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / hidden_size as f32;

            for j in 0..hidden_size {
                result[start + j] = (row[j] - mean) / (var + self.eps).sqrt();
                result[start + j] = result[start + j] * self.gamma.data[j] + self.beta.data[j];
            }
        }

        Tensor::new(result, input.shape.clone())
    }
}

#[derive(Clone)]
pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    q_proj: LinearLayer,
    k_proj: LinearLayer,
    v_proj: LinearLayer,
    out_proj: LinearLayer,
}

impl MultiHeadAttention {
    pub fn new(hidden_size: usize, num_heads: usize) -> Self {
        let head_dim = hidden_size / num_heads;
        Self {
            num_heads,
            head_dim,
            q_proj: LinearLayer::new(hidden_size, hidden_size),
            k_proj: LinearLayer::new(hidden_size, hidden_size),
            v_proj: LinearLayer::new(hidden_size, hidden_size),
            out_proj: LinearLayer::new(hidden_size, hidden_size),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, TensorError> {
        let batch_size = x.shape[0];
        let seq_len = x.shape[1];
        let hidden_size = x.shape[2];

        let _q = x.reshape(vec![batch_size * seq_len, hidden_size])?;

        // project to Q, K, V
        let q = self.q_proj.forward(&_q)?.reshape(vec![
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ])?;
        let k = self
            .k_proj
            .forward(&x.reshape(vec![batch_size * seq_len, hidden_size])?)?
            .reshape(vec![batch_size, seq_len, self.num_heads, self.head_dim])?;
        let v = self
            .v_proj
            .forward(&x.reshape(vec![batch_size * seq_len, hidden_size])?)?
            .reshape(vec![batch_size, seq_len, self.num_heads, self.head_dim])?;

        let q_flat = q.reshape(vec![batch_size * self.num_heads, seq_len, self.head_dim])?;
        let k_flat = k.reshape(vec![batch_size * self.num_heads, seq_len, self.head_dim])?;
        let v_flat = v.reshape(vec![batch_size * self.num_heads, seq_len, self.head_dim])?;

        let scale = (self.head_dim as f32).sqrt();
        let t_k_flat = k_flat.transpose()?;

        // TODO! fix here
        let attn = q_flat.matmul(&t_k_flat)?.scale(1.0 / scale)?;
        let attn =
            Softmax::forward(&attn.reshape(vec![batch_size * self.num_heads * seq_len, seq_len])?)?
                .reshape(vec![batch_size * self.num_heads, seq_len, seq_len])?;

        let out = attn
            .matmul(&v_flat)?
            .reshape(vec![batch_size, seq_len, hidden_size])?;

        self.out_proj
            .forward(&out.reshape(vec![batch_size * seq_len, hidden_size])?)?
            .reshape(vec![batch_size, seq_len, hidden_size])
    }
}

#[derive(Clone)]
pub struct TransformerBlock {
    attention: MultiHeadAttention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    mlp: LinearLayer,
}

impl TransformerBlock {
    pub fn new(hidden_size: usize, num_heads: usize) -> Self {
        Self {
            attention: MultiHeadAttention::new(hidden_size, num_heads),
            norm1: LayerNorm::new(hidden_size),
            norm2: LayerNorm::new(hidden_size),
            mlp: LinearLayer::new(hidden_size, hidden_size),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, TensorError> {
        let x = self.norm1.forward(x)?;
        let x = &self.attention.forward(&x)?;
        let x = self.norm2.forward(&x)?;
        // TODO: Add mlp
        Ok(x)
    }
}

pub struct Softmax;

impl Softmax {
    pub fn forward(input: &Tensor) -> Result<Tensor, TensorError> {
        if input.shape.len() != 2 {
            return Err(TensorError::InvalidDimensions(
                "Softmax requires 2D input".into(),
            ));
        }

        let mut result = Vec::with_capacity(input.data.len());
        let (rows, cols) = (input.shape[0], input.shape[1]);

        for i in 0..rows {
            let row_start = i * cols;
            let row = &input.data[row_start..row_start + cols];

            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            let exp_vals: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();

            let sum: f32 = exp_vals.iter().sum();
            result.extend(exp_vals.iter().map(|&x| x / sum));
        }

        Tensor::new(result, input.shape.clone())
    }
}

// TODO: Implement a static vector
// struct StaticVec<T, const N: usize> {
//     data: [T; N],
//     len: usize,
// }

// impl<T, const N: usize> StaticVec<T, N> {
//     const fn new() -> Self {
//         // This requires `MaybeUninit` for uninitialized arrays
//         StaticVec {
//             data: unsafe { core::mem::MaybeUninit::uninit().assume_init() },
//             len: 0,
//         }
//     }

//     fn push(&mut self, value: T) {
//         if self.len < N {
//             self.data[self.len] = value;
//             self.len += 1;
//         } else {
//             panic!("StaticVec overflow");
//         }
//     }

//     fn as_slice(&self) -> &[T] {
//         &self.data[..self.len]
//     }
// }
