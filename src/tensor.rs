use std::fmt;

#[derive(Debug, Clone, Copy)]
pub enum Index {
    Single(usize),
    Full,
}

impl From<usize> for Index {
    fn from(idx: usize) -> Self {
        Index::Single(idx)
    }
}

#[derive(Debug)]
pub enum TensorError {
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    InvalidDimensions(String),
    InvalidInput(String),
    IndexOutOfBounds(String),
    InvalidIndex(String),
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

    pub fn relu(&self) -> Result<Self, TensorError> {
        let result = self.data.iter().map(|&x| x.max(0.0)).collect();
        Ok(Self {
            data: result,
            shape: self.shape.clone(),
        })
    }

    pub fn gelu(&self) -> Result<Self, TensorError> {
        let result = self
            .data
            .iter()
            .map(|&x| x * 0.5 * (1.0 + (2.0 * x).tanh()))
            .collect();
        Ok(Self {
            data: result,
            shape: self.shape.clone(),
        })
    }

    pub fn mean(&self, axis: usize, keep_dims: bool) -> Result<Self, TensorError> {
        if axis >= self.shape.len() {
            return Err(TensorError::InvalidDimensions(format!(
                "Axis {} is out of bounds for tensor with {} dimensions",
                axis,
                self.shape.len()
            )));
        }

        let mut new_shape = self.shape.clone();
        if keep_dims {
            new_shape[axis] = 1;
        } else {
            new_shape.remove(axis);
        }

        let axis_size = self.shape[axis];
        let mut result = Vec::with_capacity(self.data.len() / axis_size);
        let strides = self.compute_strides(&self.shape);

        let chunk_size = self.shape[axis];
        let outer_size = self.data.len() / (chunk_size * strides[axis]);
        let inner_size = strides[axis];

        for outer in 0..outer_size {
            for i in 0..inner_size {
                let base_idx = outer * (chunk_size * strides[axis]) + i;
                let mut sum = 0.0;

                for j in 0..chunk_size {
                    let idx = base_idx + j * strides[axis];
                    sum += self.data[idx];
                }

                result.push(sum / chunk_size as f32);
            }
        }

        Ok(Self {
            data: result,
            shape: new_shape,
        })
    }

    pub fn variance(&self, axis: usize, keep_dims: bool) -> Result<Self, TensorError> {
        if axis >= self.shape.len() {
            return Err(TensorError::InvalidDimensions(format!(
                "Axis {} is out of bounds for tensor with {} dimensions",
                axis,
                self.shape.len()
            )));
        }

        let mean = self.mean(axis, true)?;
        let mut new_shape = self.shape.clone();
        if keep_dims {
            new_shape[axis] = 1;
        } else {
            new_shape.remove(axis);
        }

        let axis_size = self.shape[axis];
        let mut result = Vec::with_capacity(self.data.len() / axis_size);
        let strides = self.compute_strides(&self.shape);

        let chunk_size = self.shape[axis];
        let outer_size = self.data.len() / (chunk_size * strides[axis]);
        let inner_size = strides[axis];

        for outer in 0..outer_size {
            for i in 0..inner_size {
                let base_idx = outer * (chunk_size * strides[axis]) + i;
                let mut sum_sq_diff = 0.0;
                let mean_val = mean.data[result.len()];

                for j in 0..chunk_size {
                    let idx = base_idx + j * strides[axis];
                    let diff = self.data[idx] - mean_val;
                    sum_sq_diff += diff * diff;
                }

                result.push(sum_sq_diff / chunk_size as f32);
            }
        }

        Ok(Self {
            data: result,
            shape: new_shape,
        })
    }

    pub fn std(&self, axis: usize, keep_dims: bool) -> Result<Self, TensorError> {
        let var = self.variance(axis, keep_dims)?;
        Ok(Self {
            data: var.data.into_iter().map(|x| x.sqrt()).collect(),
            shape: var.shape,
        })
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

    pub fn get(&self, indexes: &[Index]) -> Result<Self, TensorError> {
        if indexes.len() != self.shape.len() {
            return Err(TensorError::InvalidDimensions(format!(
                "Expected {} indexes, got {}",
                self.shape.len(),
                indexes.len()
            )));
        }

        let strides = self.compute_strides(&self.shape);

        let new_shape: Vec<usize> = indexes
            .iter()
            .zip(&self.shape)
            .filter_map(|(idx, &dim)| match idx {
                Index::Full => Some(dim),
                Index::Single(_) => None,
            })
            .collect();

        if new_shape.is_empty() {
            let index = self.compute_single_index(indexes, &self.shape, &strides)?;
            return Ok(Self {
                data: vec![self.data[index]],
                shape: vec![1],
            });
        }

        let mut new_data = Vec::new();
        self.collect_elements(indexes, &self.shape, &strides, &mut new_data, 0, 0, &[])?;

        Ok(Self {
            data: new_data,
            shape: new_shape,
        })
    }

    fn compute_single_index(
        &self,
        indexes: &[Index],
        shape: &[usize],
        strides: &[usize],
    ) -> Result<usize, TensorError> {
        let mut index = 0;
        for ((&idx, &dim), &stride) in indexes.iter().zip(shape.iter()).zip(strides.iter()) {
            match idx {
                Index::Single(i) => {
                    if i >= dim {
                        return Err(TensorError::IndexOutOfBounds(format!(
                            "Index {} out of bounds for dimension {}",
                            i, dim
                        )));
                    }
                    index += i * stride;
                }
                Index::Full => {
                    return Err(TensorError::InvalidIndex(
                        "Cannot compute single index with full dimension slice".into(),
                    ));
                }
            }
        }
        Ok(index)
    }

    #[allow(clippy::too_many_arguments)]
    fn collect_elements(
        &self,
        indexes: &[Index],
        shape: &[usize],
        strides: &[usize],
        result: &mut Vec<f32>,
        dim: usize,
        current_index: usize,
        current_coords: &[usize],
    ) -> Result<(), TensorError> {
        if dim == shape.len() {
            result.push(self.data[current_index]);
            return Ok(());
        }

        let mut coords = current_coords.to_owned();
        match indexes[dim] {
            Index::Single(idx) => {
                if idx >= shape[dim] {
                    return Err(TensorError::IndexOutOfBounds(format!(
                        "Index {} out of bounds for dimension {}",
                        idx, dim
                    )));
                }
                coords.push(idx);
                self.collect_elements(
                    indexes,
                    shape,
                    strides,
                    result,
                    dim + 1,
                    current_index + idx * strides[dim],
                    &coords,
                )?;
            }
            Index::Full => {
                for i in 0..shape[dim] {
                    coords.push(i);
                    self.collect_elements(
                        indexes,
                        shape,
                        strides,
                        result,
                        dim + 1,
                        current_index + i * strides[dim],
                        &coords,
                    )?;
                    coords.pop();
                }
            }
        }
        Ok(())
    }

    pub fn compute_index(&self, coords: &[usize], shape: &[usize], strides: &[usize]) -> usize {
        let mut index = 0;
        for ((&c, &s), &dim) in coords.iter().zip(strides.iter()).zip(shape.iter()) {
            index += (c % dim) * s;
        }
        index
    }

    pub fn compute_strides(&self, shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    pub fn div(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        let broadcast_shape = Self::compute_broadcast_shapes(&self.shape, &other.shape)
            .ok_or_else(|| TensorError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            })?;

        let self_strides = self.compute_strides(&self.shape);
        let other_strides = other.compute_strides(&other.shape);

        let total_size: usize = broadcast_shape.iter().product();
        let mut result = Vec::with_capacity(total_size);

        let to_coords = |mut idx: usize, shape: &[usize]| -> Vec<usize> {
            let mut coords = vec![0; shape.len()];
            for i in (0..shape.len()).rev() {
                coords[i] = idx % shape[i];
                idx /= shape[i];
            }
            coords
        };

        let self_offset = broadcast_shape.len() - self.shape.len();
        let other_offset = broadcast_shape.len() - other.shape.len();

        for i in 0..total_size {
            let coords = to_coords(i, &broadcast_shape);

            let self_coords: Vec<usize> = coords.iter().skip(self_offset).copied().collect();
            let other_coords: Vec<usize> = coords.iter().skip(other_offset).copied().collect();

            let self_idx = self.compute_index(&self_coords, &self.shape, &self_strides);
            let other_idx = other.compute_index(&other_coords, &other.shape, &other_strides);

            result.push(self.data[self_idx] / other.data[other_idx]);
        }

        Tensor::new(result, broadcast_shape)
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        match (self.shape.len(), other.shape.len()) {
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
            (3, 2) => {
                if self.shape[2] != other.shape[0] {
                    return Err(TensorError::ShapeMismatch {
                        expected: vec![self.shape[2]],
                        got: vec![other.shape[0]],
                    });
                }

                let (batch, m, k) = (self.shape[0], self.shape[1], self.shape[2]);
                let n = other.shape[1];
                let mut result = vec![0.0; batch * m * n];
                let self_stride = m * k;
                let result_stride = m * n;

                for b in 0..batch {
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = 0.0;
                            for kk in 0..k {
                                let self_idx = b * self_stride + i * k + kk;
                                let other_idx = kk * n + j;
                                sum += self.data[self_idx] * other.data[other_idx];
                            }
                            result[b * result_stride + i * n + j] = sum;
                        }
                    }
                }

                Tensor::new(result, vec![batch, m, n])
            }

            (2, 3) => {
                if self.shape[1] != other.shape[1] {
                    return Err(TensorError::ShapeMismatch {
                        expected: vec![self.shape[1]],
                        got: vec![other.shape[1]],
                    });
                }

                let (m, k) = (self.shape[0], self.shape[1]);
                let (batch, _, n) = (other.shape[0], other.shape[1], other.shape[2]);
                let mut result = vec![0.0; batch * m * n];
                let other_stride = k * n;
                let result_stride = m * n;

                for b in 0..batch {
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = 0.0;
                            for kk in 0..k {
                                let self_idx = i * k + kk;
                                let other_idx = b * other_stride + kk * n + j;
                                sum += self.data[self_idx] * other.data[other_idx];
                            }
                            result[b * result_stride + i * n + j] = sum;
                        }
                    }
                }

                Tensor::new(result, vec![batch, m, n])
            }

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

            (4, 4) => {
                if self.shape[0] != other.shape[0] || self.shape[1] != other.shape[1] {
                    return Err(TensorError::ShapeMismatch {
                        expected: vec![self.shape[0], self.shape[1]],
                        got: vec![other.shape[0], other.shape[1]],
                    });
                }
                if self.shape[3] != other.shape[2] {
                    return Err(TensorError::ShapeMismatch {
                        expected: vec![self.shape[3]],
                        got: vec![other.shape[2]],
                    });
                }

                let (batch1, batch2, m, k) =
                    (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
                let n = other.shape[3];

                let mut result = vec![0.0; batch1 * batch2 * m * n];

                let self_b1_stride = self.shape[1] * self.shape[2] * self.shape[3];
                let self_b2_stride = self.shape[2] * self.shape[3];
                let self_m_stride = self.shape[3];

                let other_b1_stride = other.shape[1] * other.shape[2] * other.shape[3];
                let other_b2_stride = other.shape[2] * other.shape[3];
                let other_k_stride = other.shape[3];

                let result_b1_stride = batch2 * m * n;
                let result_b2_stride = m * n;
                let result_m_stride = n;

                for b1 in 0..batch1 {
                    for b2 in 0..batch2 {
                        for i in 0..m {
                            for j in 0..n {
                                let mut sum = 0.0;
                                for kk in 0..k {
                                    let self_idx = b1 * self_b1_stride
                                        + b2 * self_b2_stride
                                        + i * self_m_stride
                                        + kk;
                                    let other_idx = b1 * other_b1_stride
                                        + b2 * other_b2_stride
                                        + kk * other_k_stride
                                        + j;
                                    sum += self.data[self_idx] * other.data[other_idx];
                                }
                                let result_idx = b1 * result_b1_stride
                                    + b2 * result_b2_stride
                                    + i * result_m_stride
                                    + j;
                                result[result_idx] = sum;
                            }
                        }
                    }
                }

                Tensor::new(result, vec![batch1, batch2, m, n])
            }

            (d1, d2) => Err(TensorError::InvalidDimensions(format!(
                "Unsupported tensor dimensions for matmul: {d1}D and {d2}D"
            ))),
        }
    }

    //
    pub fn transpose_at(&self, indices: &[usize]) -> Result<Tensor, TensorError> {
        if indices.len() != 2 {
            return Err(TensorError::InvalidDimensions(format!(
                "Expected 2 indices, got {}",
                indices.len()
            )));
        }

        let (i, j) = (indices[0], indices[1]);
        if i >= self.shape.len() || j >= self.shape.len() {
            return Err(TensorError::InvalidDimensions(format!(
                "Indices {} and {} are out of bounds for tensor with {} dimensions",
                i,
                j,
                self.shape.len()
            )));
        }

        let mut new_shape = self.shape.clone();
        new_shape.swap(i, j);

        let mut new_data = vec![0.0; self.data.len()];
        let old_strides = self.compute_strides(&self.shape);
        let new_strides = self.compute_strides(&new_shape);

        for idx in 0..self.data.len() {
            let mut coords = vec![0; self.shape.len()];
            let mut remaining = idx;
            for (dim, &stride) in old_strides.iter().enumerate() {
                coords[dim] = remaining / stride;
                remaining %= stride;
            }

            coords.swap(i, j);

            let new_idx: usize = coords
                .iter()
                .enumerate()
                .map(|(dim, &coord)| coord * new_strides[dim])
                .sum();

            new_data[new_idx] = self.data[idx];
        }

        Tensor::new(new_data, new_shape)
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
            dims => Err(TensorError::InvalidDimensions(format!(
                "Transpose requires 2D or 3D tensor, got {dims}D"
            ))),
        }
    }

    pub fn scale(&self, factor: f32) -> Result<Tensor, TensorError> {
        let result = self.data.iter().map(|&x| x * factor).collect();
        Tensor::new(result, self.shape.clone())
    }

    pub fn softmax(&self, idx: Index) -> Result<Tensor, TensorError> {
        let mut result = Vec::with_capacity(self.data.len());
        let strides = self.compute_strides(&self.shape);

        match idx {
            Index::Single(axis) => {
                if axis >= self.shape.len() {
                    return Err(TensorError::InvalidDimensions(format!(
                        "Axis {} is out of bounds for tensor with {} dimensions",
                        axis,
                        self.shape.len()
                    )));
                }

                let mut coords = vec![0; self.shape.len()];
                for i in 0..self.data.len() {
                    coords[axis] = i / strides[axis];
                    let start = coords[axis] * strides[axis];
                    let end = start + strides[axis];
                    let slice = &self.data[start..end];
                    let max_val = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let sum: f32 = slice.iter().map(|x| (x - max_val).exp()).sum();
                    result.push((self.data[i] - max_val).exp() / sum);
                }
            }
            Index::Full => {
                let max_val = self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = self.data.iter().map(|x| (x - max_val).exp()).sum();
                result = self
                    .data
                    .iter()
                    .map(|x| (x - max_val).exp() / sum)
                    .collect();
            }
        }

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

    fn compute_broadcast_shapes(shape1: &[usize], shape2: &[usize]) -> Option<Vec<usize>> {
        let n1 = shape1.len();
        let n2 = shape2.len();
        let n = n1.max(n2);

        let padded1: Vec<usize> = std::iter::repeat(1)
            .take(n - n1)
            .chain(shape1.iter().copied())
            .collect();
        let padded2: Vec<usize> = std::iter::repeat(1)
            .take(n - n2)
            .chain(shape2.iter().copied())
            .collect();

        let mut result = Vec::with_capacity(n);
        for (d1, d2) in padded1.iter().zip(padded2.iter()) {
            if d1 == d2 {
                result.push(*d1);
            } else if *d1 == 1 {
                result.push(*d2);
            } else if *d2 == 1 {
                result.push(*d1);
            } else {
                return None;
            }
        }
        Some(result)
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        let broadcast_shape = Self::compute_broadcast_shapes(&self.shape, &other.shape)
            .ok_or_else(|| TensorError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            })?;

        let self_strides = self.compute_strides(&self.shape);
        let other_strides = other.compute_strides(&other.shape);

        let total_size: usize = broadcast_shape.iter().product();
        let mut result = Vec::with_capacity(total_size);

        let to_coords = |mut idx: usize, shape: &[usize]| -> Vec<usize> {
            let mut coords = vec![0; shape.len()];
            for i in (0..shape.len()).rev() {
                coords[i] = idx % shape[i];
                idx /= shape[i];
            }
            coords
        };

        let self_offset = broadcast_shape.len() - self.shape.len();
        let other_offset = broadcast_shape.len() - other.shape.len();

        for i in 0..total_size {
            let coords = to_coords(i, &broadcast_shape);

            let self_coords: Vec<usize> = coords.iter().skip(self_offset).copied().collect();
            let other_coords: Vec<usize> = coords.iter().skip(other_offset).copied().collect();

            let self_idx = self.compute_index(&self_coords, &self.shape, &self_strides);
            let other_idx = other.compute_index(&other_coords, &other.shape, &other_strides);

            result.push(self.data[self_idx] - other.data[other_idx]);
        }

        Tensor::new(result, broadcast_shape)
    }

    pub fn add_scalar(&self, scalar: f32) -> Result<Tensor, TensorError> {
        let result = self.data.iter().map(|&x| x + scalar).collect();
        Tensor::new(result, self.shape.clone())
    }

    pub fn add(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        let broadcast_shape = Self::compute_broadcast_shapes(&self.shape, &other.shape)
            .ok_or_else(|| TensorError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            })?;

        let self_strides = self.compute_strides(&self.shape);
        let other_strides = other.compute_strides(&other.shape);

        let total_size: usize = broadcast_shape.iter().product();
        let mut result = Vec::with_capacity(total_size);

        let to_coords = |mut idx: usize, shape: &[usize]| -> Vec<usize> {
            let mut coords = vec![0; shape.len()];
            for i in (0..shape.len()).rev() {
                coords[i] = idx % shape[i];
                idx /= shape[i];
            }
            coords
        };

        let self_offset = broadcast_shape.len() - self.shape.len();
        let other_offset = broadcast_shape.len() - other.shape.len();

        for i in 0..total_size {
            let coords = to_coords(i, &broadcast_shape);

            let self_coords: Vec<usize> = coords.iter().skip(self_offset).copied().collect();
            let other_coords: Vec<usize> = coords.iter().skip(other_offset).copied().collect();

            let self_idx = self.compute_index(&self_coords, &self.shape, &self_strides);
            let other_idx = self.compute_index(&other_coords, &other.shape, &other_strides);

            result.push(self.data[self_idx] + other.data[other_idx]);
        }

        Tensor::new(result, broadcast_shape)
    }

    pub fn print(&self) {
        const MAX_ITEMS_PER_ROW: usize = 8;
        const MAX_ROWS: usize = 6;

        println!("Tensor({:?})", self.shape);
        println!("[");

        match self.shape.len() {
            1 => {
                let total = self.shape[0];
                if total <= MAX_ITEMS_PER_ROW {
                    print!("    ");
                    for i in 0..total {
                        print!("{:7.3} ", self.data[i]);
                    }
                    println!();
                } else {
                    print!("    ");
                    for i in 0..MAX_ITEMS_PER_ROW / 2 {
                        print!("{:7.3} ", self.data[i]);
                    }
                    print!("... ");
                    for i in (total - MAX_ITEMS_PER_ROW / 2)..total {
                        print!("{:7.3} ", self.data[i]);
                    }
                    println!();
                }
            }
            2 => {
                let rows = self.shape[0];
                let cols = self.shape[1];

                let (head_rows, tail_rows) = if rows > MAX_ROWS {
                    (MAX_ROWS / 2, MAX_ROWS / 2)
                } else {
                    (rows, 0)
                };

                for i in 0..rows {
                    if i == head_rows && tail_rows > 0 && rows > MAX_ROWS {
                        println!("    ...");
                        continue;
                    }
                    if rows > MAX_ROWS && i >= head_rows && i < rows - tail_rows {
                        continue;
                    }

                    print!("    ");
                    if cols > MAX_ITEMS_PER_ROW {
                        for j in 0..MAX_ITEMS_PER_ROW / 2 {
                            print!("{:7.3} ", self.data[i * cols + j]);
                        }
                        print!("... ");
                        for j in cols - MAX_ITEMS_PER_ROW / 2..cols {
                            print!("{:7.3} ", self.data[i * cols + j]);
                        }
                    } else {
                        for j in 0..cols {
                            print!("{:7.3} ", self.data[i * cols + j]);
                        }
                    }
                    println!();
                }
            }
            _ => {
                println!("    Cannot display tensors with more than 2 dimensions");
            }
        }

        println!("]");
        println!("Shape <{:?}>", self.shape);
    }

    pub fn concat(tensors: &[&Tensor], axis: usize) -> Result<Tensor, TensorError> {
        if tensors.is_empty() {
            return Err(TensorError::InvalidInput("Empty tensor list".into()));
        }

        let reference_shape = &tensors[0].shape;
        if axis >= reference_shape.len() {
            return Err(TensorError::InvalidInput(format!(
                "Axis {} is out of bounds for tensor with {} dimensions",
                axis,
                reference_shape.len()
            )));
        }

        for (_i, tensor) in tensors.iter().enumerate().skip(1) {
            if tensor.shape.len() != reference_shape.len() {
                return Err(TensorError::ShapeMismatch {
                    expected: reference_shape.clone(),
                    got: tensor.shape.clone(),
                });
            }
            for (d, (&s1, &s2)) in reference_shape.iter().zip(tensor.shape.iter()).enumerate() {
                if d != axis && s1 != s2 {
                    return Err(TensorError::ShapeMismatch {
                        expected: reference_shape.clone(),
                        got: tensor.shape.clone(),
                    });
                }
            }
        }

        let mut result_shape = reference_shape.clone();
        result_shape[axis] = tensors.iter().map(|t| t.shape[axis]).sum();
        let mut result = vec![0.0; result_shape.iter().product()];

        let mut strides = vec![1; result_shape.len()];
        for i in (0..strides.len() - 1).rev() {
            strides[i] = strides[i + 1] * result_shape[i + 1];
        }

        let outer_dims: usize = result_shape[..axis].iter().product();
        let inner_dims: usize = result_shape[axis + 1..].iter().product();

        let mut axis_offset = 0;

        for tensor in tensors {
            let axis_size = tensor.shape[axis];

            let mut input_strides = vec![1; tensor.shape.len()];
            for i in (0..input_strides.len() - 1).rev() {
                input_strides[i] = input_strides[i + 1] * tensor.shape[i + 1];
            }

            for outer in 0..outer_dims {
                for axis_pos in 0..axis_size {
                    for inner in 0..inner_dims {
                        let src_axis_idx =
                            outer * input_strides[0] + axis_pos * input_strides[axis] + inner;

                        let dst_axis_idx =
                            outer * strides[0] + (axis_offset + axis_pos) * strides[axis] + inner;

                        result[dst_axis_idx] = tensor.data[src_axis_idx];
                    }
                }
            }

            axis_offset += axis_size;
        }

        Tensor::new(result, result_shape)
    }

    pub fn split(
        tensor: &Tensor,
        axis: usize,
        split_sizes: &[usize],
    ) -> Result<Vec<Tensor>, TensorError> {
        if axis >= tensor.shape.len() {
            return Err(TensorError::InvalidDimensions(format!(
                "Axis {} is out of bounds for tensor with {} dimensions",
                axis,
                tensor.shape.len()
            )));
        }

        let total_size: usize = split_sizes.iter().sum();
        if total_size != tensor.shape[axis] {
            return Err(TensorError::InvalidDimensions(format!(
                "Total split size {} does not match tensor size {}",
                total_size, tensor.shape[axis]
            )));
        }

        let mut result = Vec::with_capacity(split_sizes.len());
        let mut start_idx = 0;

        let slice_size: usize = tensor.shape[axis + 1..].iter().product();
        let outer_size: usize = tensor.shape[..axis].iter().product();

        for &size in split_sizes {
            let mut shape = tensor.shape.clone();
            shape[axis] = size;
            let chunk_size = size * slice_size;
            let mut data = Vec::with_capacity(outer_size * chunk_size);

            for outer in 0..outer_size {
                let outer_offset = outer * (tensor.shape[axis] * slice_size);

                for i in 0..size {
                    let src_start = outer_offset + (start_idx + i) * slice_size;
                    let src_end = src_start + slice_size;
                    data.extend_from_slice(&tensor.data[src_start..src_end]);
                }
            }

            result.push(Tensor::new(data, shape)?);
            start_idx += size;
        }

        Ok(result)
    }

    pub fn squeeze(&self, axis: Option<usize>) -> Result<Tensor, TensorError> {
        match axis {
            Some(axis) => {
                if axis >= self.shape.len() {
                    return Err(TensorError::InvalidDimensions(format!(
                        "Axis {} is out of bounds for tensor with {} dimensions",
                        axis,
                        self.shape.len()
                    )));
                }

                if self.shape[axis] != 1 {
                    return Err(TensorError::InvalidDimensions(format!(
                        "Cannot squeeze dimension {} with size {}",
                        axis, self.shape[axis]
                    )));
                }

                let mut new_shape = Vec::with_capacity(self.shape.len() - 1);
                for (i, &dim) in self.shape.iter().enumerate() {
                    if i != axis {
                        new_shape.push(dim);
                    }
                }

                Ok(Tensor::new(self.data.clone(), new_shape)?)
            }
            None => {
                let new_shape: Vec<usize> = self
                    .shape
                    .iter()
                    .filter(|&&dim| dim != 1)
                    .copied()
                    .collect();

                let final_shape = if new_shape.is_empty() {
                    vec![1]
                } else {
                    new_shape
                };

                Ok(Tensor::new(self.data.clone(), final_shape)?)
            }
        }
    }

    pub fn unsqueeze(&self, axis: usize) -> Result<Tensor, TensorError> {
        if axis > self.shape.len() {
            return Err(TensorError::InvalidDimensions(format!(
                "Axis {} is out of bounds for tensor with {} dimensions",
                axis,
                self.shape.len()
            )));
        }

        let mut new_shape = Vec::with_capacity(self.shape.len() + 1);
        new_shape.extend_from_slice(&self.shape[..axis]);
        new_shape.push(1);
        new_shape.extend_from_slice(&self.shape[axis..]);

        Tensor::new(self.data.clone(), new_shape)
    }

    pub fn compute_coords(&self, index: usize, shape: &[usize], strides: &[usize]) -> Vec<usize> {
        let mut coords = vec![0; shape.len()];
        let mut index = index;
        for i in 0..shape.len() {
            coords[i] = index / strides[i];
            index %= strides[i];
        }
        coords
    }

    pub fn permute(&self, permutation: &[usize]) -> Result<Tensor, TensorError> {
        if permutation.len() != self.shape.len() {
            return Err(TensorError::InvalidDimensions(format!(
                "Permutation length {} does not match tensor dimensions {}",
                permutation.len(),
                self.shape.len()
            )));
        }

        let mut new_shape = vec![0; self.shape.len()];
        for (i, &p) in permutation.iter().enumerate() {
            if p >= self.shape.len() {
                return Err(TensorError::InvalidDimensions(format!(
                    "Permutation index {} is out of bounds for tensor with {} dimensions",
                    p,
                    self.shape.len()
                )));
            }
            new_shape[i] = self.shape[p];
        }

        let mut new_data = vec![0.0; self.data.len()];
        let old_strides = self.compute_strides(&self.shape);
        let new_strides = self.compute_strides(&new_shape);

        for i in 0..self.data.len() {
            let old_coords = self.compute_coords(i, &self.shape, &old_strides);

            let mut new_coords = vec![0; self.shape.len()];
            for (new_dim, &old_dim) in permutation.iter().enumerate() {
                new_coords[new_dim] = old_coords[old_dim];
            }

            let new_idx = self.compute_index(&new_coords, &new_shape, &new_strides);
            new_data[new_idx] = self.data[i];
        }

        Tensor::new(new_data, new_shape)
    }
}
