// #![no_std]

pub mod mnist_classifier;
pub mod tensor;
pub mod vit;

mod tests;
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
        let x = input.matmul(&self.weights)?;
        let x = x.add(&self.bias)?;
        Ok(x)
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
            gamma: Tensor::new(vec![1.0; size], vec![1, size, 1]).unwrap(),
            beta: Tensor::ones(vec![1, size]),
            eps: 1e-6,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, TensorError> {
        let mean = input.mean(2, true)?;
        let variance = input.variance(2, true)?;
        let norm = input.sub(&mean)?;
        let denom = &variance.add_scalar(self.eps)?;
        let norm = norm.div(denom)?;
        let norm = norm.matmul(&self.gamma)?;
        let norm = norm.add(&self.beta)?;
        Ok(norm)
    }
}

#[derive(Clone)]
pub struct MultiHeadAttention {
    _num_heads: usize,
    _head_dim: usize,
    q_proj: LinearLayer,
    k_proj: LinearLayer,
    v_proj: LinearLayer,
    out_proj: LinearLayer,
}

impl MultiHeadAttention {
    pub fn new(hidden_size: usize, num_heads: usize) -> Self {
        let head_dim = hidden_size / num_heads;
        Self {
            _num_heads: num_heads,
            _head_dim: head_dim,
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

        let _q = &self.q_proj.weights;
        let _k = &self.k_proj.weights;
        let _v = &self.v_proj.weights;
        let qkv = Tensor::concat(&[_q, _k, _v], 1)?;
        let qkv_bias = Tensor::concat(
            &[&self.q_proj.bias, &self.k_proj.bias, &self.v_proj.bias],
            0,
        )?
        .reshape(vec![1, 3 * hidden_size])?;

        // linear
        let qkv_x = x.matmul(&qkv)?;
        let qkv_x = qkv_x.add(&qkv_bias)?;
        let qkv_x = qkv_x.reshape(vec![batch_size, seq_len, 3, 8, hidden_size / 8])?;
        let qkv_x = qkv_x.permute(&[2, 0, 3, 1, 4])?;
        let qkv_x = Tensor::split(&qkv_x, 0, &[1, 1, 1])?;

        let q = qkv_x[0].clone().squeeze(Some(0))?;
        let k = qkv_x[1].clone().squeeze(Some(0))?;
        let v = qkv_x[2].clone().squeeze(Some(0))?;
        let k = k.transpose_at(&[2, 3])?;

        let attn = q.matmul(&k)?;
        let attn = attn.softmax(tensor::Index::Single(2))?;

        let x = attn.matmul(&v)?;
        let x = x.transpose_at(&[1, 2])?;
        let x = x.reshape(vec![batch_size, seq_len, hidden_size])?;

        let proj = Tensor::ones(vec![hidden_size, hidden_size]);

        let x = x.matmul(&proj)?;
        let x = x.add(&self.out_proj.bias)?;

        Ok(x)
    }
}

#[derive(Clone)]
pub struct Mlp {
    fc1: LinearLayer,
    fc2: LinearLayer,
}

impl Mlp {
    pub fn new(hidden_size: usize) -> Self {
        Self {
            fc1: LinearLayer::new(hidden_size, 4 * hidden_size),
            fc2: LinearLayer::new(4 * hidden_size, hidden_size),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, TensorError> {
        let x = self.fc1.forward(x)?;
        let x = x.gelu()?;
        let x = self.fc2.forward(&x)?;
        Ok(x)
    }
}

#[derive(Clone)]
pub struct TransformerBlock {
    attention: MultiHeadAttention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    mlp: Mlp,
}

impl TransformerBlock {
    pub fn new(hidden_size: usize, num_heads: usize) -> Self {
        Self {
            attention: MultiHeadAttention::new(hidden_size, num_heads),
            norm1: LayerNorm::new(hidden_size),
            norm2: LayerNorm::new(hidden_size),
            mlp: Mlp::new(hidden_size),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, TensorError> {
        let normed = self.norm1.forward(x)?;
        let attn_out = &self.attention.forward(&normed)?;
        let normed = self.norm2.forward(x)?;
        let mlp_out = &self.mlp.forward(&normed)?;
        let x = x.add(mlp_out)?.add(attn_out)?;
        Ok(x.clone())
    }
}

pub struct Softmax;

impl Softmax {
    pub fn forward(input: &Tensor, dim: Option<usize>) -> Result<Tensor, TensorError> {
        let dim = dim.unwrap_or(input.shape.len() - 1);

        if dim >= input.shape.len() {
            return Err(TensorError::InvalidDimensions(format!(
                "Dimension {} is out of bounds for tensor with {} dimensions",
                dim,
                input.shape.len()
            )));
        }

        let strides = input.compute_strides(&input.shape);
        let dim_size = input.shape[dim];
        let mut result = vec![0.0; input.data.len()];

        let outer_size = input.data.len() / (dim_size * strides[dim]);
        let inner_size = strides[dim];

        for outer in 0..outer_size {
            for i in 0..inner_size {
                let base_idx = outer * (dim_size * strides[dim]) + i;

                let mut max_val = f32::NEG_INFINITY;
                for j in 0..dim_size {
                    let idx = base_idx + j * strides[dim];
                    max_val = max_val.max(input.data[idx]);
                }

                let mut sum = 0.0;
                let mut exp_vals = vec![0.0; dim_size];
                for (j, exp_val) in exp_vals.iter_mut().enumerate() {
                    let idx = base_idx + j * strides[dim];
                    *exp_val = (input.data[idx] - max_val).exp();
                    sum += *exp_val;
                }

                for (j, &exp_val) in exp_vals.iter().enumerate() {
                    let idx = base_idx + j * strides[dim];
                    result[idx] = exp_val / sum;
                }
            }
        }

        Tensor::new(result, input.shape.clone())
    }

    pub fn forward_last_dim(input: &Tensor) -> Result<Tensor, TensorError> {
        Self::forward(input, None)
    }
}

impl Softmax {
    pub fn forward_with_max(
        input: &Tensor,
        dim: Option<usize>,
    ) -> Result<(Tensor, Vec<f32>), TensorError> {
        let dim = dim.unwrap_or(input.shape.len() - 1);
        if dim >= input.shape.len() {
            return Err(TensorError::InvalidDimensions(format!(
                "Dimension {} is out of bounds for tensor with {} dimensions",
                dim,
                input.shape.len()
            )));
        }

        let strides = input.compute_strides(&input.shape);
        let dim_size = input.shape[dim];
        let mut result = vec![0.0; input.data.len()];
        let mut max_values = Vec::new();

        let outer_size = input.data.len() / (dim_size * strides[dim]);
        let inner_size = strides[dim];

        for outer in 0..outer_size {
            for i in 0..inner_size {
                let base_idx = outer * (dim_size * strides[dim]) + i;

                let mut max_val = f32::NEG_INFINITY;
                for j in 0..dim_size {
                    let idx = base_idx + j * strides[dim];
                    max_val = max_val.max(input.data[idx]);
                }
                max_values.push(max_val);

                let mut sum = 0.0;
                let mut exp_vals = vec![0.0; dim_size];
                for (j, exp_val) in exp_vals.iter_mut().enumerate() {
                    let idx = base_idx + j * strides[dim];
                    *exp_val = (input.data[idx] - max_val).exp();
                    sum += *exp_val;
                }

                for (j, &exp_val) in exp_vals.iter().enumerate() {
                    let idx = base_idx + j * strides[dim];
                    result[idx] = exp_val / sum;
                }
            }
        }

        Ok((Tensor::new(result, input.shape.clone())?, max_values))
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
