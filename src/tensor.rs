use std::fmt;

#[derive(Debug)]
pub enum TensorError {
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    InvalidDimensions(String),
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.shape.len() != 2 {
            return write!(f, "Cannot display non-2D tensor");
        }

        let (rows, cols) = (self.shape[0], self.shape[1]);
        for i in 0..rows {
            writeln!(f, "[")?;
            for j in 0..cols {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:7.3}", self.data[i * cols + j])?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, TensorError> {
        let expected_size = shape.iter().product();
        if data.len() != expected_size {
            return Err(TensorError::ShapeMismatch {
                expected: vec![expected_size],
                got: vec![data.len()],
            });
        }
        Ok(Self { data, shape })
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self {
            data: vec![0.0; size],
            shape,
        }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self {
            data: vec![1.0; size],
            shape,
        }
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        match (self.shape.len(), other.shape.len()) {
            // Regular 2D matrix multiplication
            (2, 2) => {
                if self.shape[1] != other.shape[0] {
                    return Err(TensorError::ShapeMismatch {
                        expected: vec![self.shape[1]],
                        got: vec![other.shape[0]],
                    });
                }

                let (m, k) = (self.shape[0], self.shape[1]);
                let n = other.shape[1];
                let mut result = vec![0.0; m * n];

                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0;
                        for k in 0..k {
                            sum += self.data[i * self.shape[1] + k]
                                * other.data[k * other.shape[1] + j];
                        }
                        result[i * n + j] = sum;
                    }
                }

                Tensor::new(result, vec![m, n])
            }

            // Batch matrix multiplication (3D x 3D)
            (3, 3) => {
                if self.shape[0] != other.shape[0] || self.shape[2] != other.shape[1] {
                    return Err(TensorError::ShapeMismatch {
                        expected: vec![self.shape[2]],
                        got: vec![other.shape[1]],
                    });
                }

                let (batch, m, k) = (self.shape[0], self.shape[1], self.shape[2]);
                let n = other.shape[2];
                let mut result = vec![0.0; batch * m * n];
                let self_stride = m * k;
                let other_stride = k * n;
                let result_stride = m * n;

                for b in 0..batch {
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = 0.0;
                            for kk in 0..k {
                                let self_idx = b * self_stride + i * k + kk;
                                let other_idx = b * other_stride + kk * n + j;
                                sum += self.data[self_idx] * other.data[other_idx];
                            }
                            result[b * result_stride + i * n + j] = sum;
                        }
                    }
                }

                Tensor::new(result, vec![batch, m, n])
            }

            (d1, d2) => Err(TensorError::InvalidDimensions(
                format!("Matmul requires both tensors to be 2D or 3D, got {d1}D and {d2}D").into(),
            )),
        }
    }

    pub fn transpose(&self) -> Result<Tensor, TensorError> {
        match self.shape.len() {
            2 => {
                let (rows, cols) = (self.shape[0], self.shape[1]);
                let mut result = vec![0.0; self.data.len()];

                for i in 0..rows {
                    for j in 0..cols {
                        result[j * rows + i] = self.data[i * cols + j];
                    }
                }

                Tensor::new(result, vec![cols, rows])
            }
            3 => {
                let (d1, d2, d3) = (self.shape[0], self.shape[1], self.shape[2]);
                let mut result = vec![0.0; self.data.len()];

                for i in 0..d1 {
                    for j in 0..d2 {
                        for k in 0..d3 {
                            let from_idx = i * (d2 * d3) + j * d3 + k;
                            let to_idx = k * (d1 * d2) + j * d1 + i;
                            result[to_idx] = self.data[from_idx];
                        }
                    }
                }

                Tensor::new(result, vec![d3, d2, d1])
            }
            dims => Err(TensorError::InvalidDimensions(
                format!("Transpose requires 2D or 3D tensor, got {dims}D").into(),
            )),
        }
    }

    pub fn scale(&self, factor: f32) -> Result<Tensor, TensorError> {
        let result = self.data.iter().map(|&x| x * factor).collect();
        Tensor::new(result, self.shape.clone())
    }

    pub fn select_token(&self, idx: usize) -> Result<Tensor, TensorError> {
        if idx >= self.shape[1] {
            return Err(TensorError::InvalidDimensions(
                "Token index out of bounds".into(),
            ));
        }
        let batch_size = self.shape[0];
        let hidden_size = self.shape[2];
        let mut result = vec![0.0; batch_size * hidden_size];

        for i in 0..batch_size {
            for j in 0..hidden_size {
                result[i * hidden_size + j] =
                    self.data[i * self.shape[1] * hidden_size + idx * hidden_size + j];
            }
        }

        Tensor::new(result, vec![batch_size, hidden_size])
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Tensor, TensorError> {
        let new_size = new_shape.iter().product();
        if new_size != self.data.len() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![new_size],
                got: vec![self.data.len()],
            });
        }
        Tensor::new(self.data.clone(), new_shape)
    }

    pub fn add(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        // Handle broadcasting for batch dimension
        if self.shape[0] != other.shape[0] {
            if other.shape[0] == 1 {
                // Broadcast other tensor across batch dimension
                let mut result = Vec::with_capacity(self.data.len());

                for _batch in 0..self.shape[0] {
                    result.extend_from_slice(&other.data);
                }

                let broadcasted =
                    Tensor::new(result, vec![self.shape[0], other.shape[1], other.shape[2]])?;

                return self.add(&broadcasted);
            } else {
                return Err(TensorError::ShapeMismatch {
                    expected: self.shape.clone(),
                    got: other.shape.clone(),
                });
            }
        }

        let result = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        Tensor::new(result, self.shape.clone())
    }
}
