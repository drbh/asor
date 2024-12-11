#![allow(unused_imports)]

use crate::tensor::{Index, Tensor};
use crate::LinearLayer;
use crate::Mlp;

// tests for basic tensor creation and initialization
#[test]
fn test_tensor_new() {
    // test successful creation with valid inputs
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    assert_eq!(tensor.data, vec![1.0, 2.0, 3.0]);
    assert_eq!(tensor.shape, vec![1, 3]);

    // test creation with empty data
    let empty_tensor = Tensor::new(vec![], vec![0]).unwrap();
    assert_eq!(empty_tensor.data, vec![]);
    assert_eq!(empty_tensor.shape, vec![0]);

    // test error case: mismatched dimensions
    assert!(Tensor::new(vec![1.0, 2.0], vec![2, 2]).is_err());
}

// tests for creating tensors filled with ones
#[test]
fn test_tensor_ones() {
    // test 2D tensor
    let tensor = Tensor::ones(vec![2, 3]);
    assert_eq!(tensor.data, vec![1.0; 6]);
    assert_eq!(tensor.shape, vec![2, 3]);

    // test 1D tensor
    let tensor_1d = Tensor::ones(vec![3]);
    assert_eq!(tensor_1d.data, vec![1.0; 3]);
    assert_eq!(tensor_1d.shape, vec![3]);

    // test empty tensor
    let empty_tensor = Tensor::ones(vec![0]);
    assert_eq!(empty_tensor.data, vec![]);
    assert_eq!(empty_tensor.shape, vec![0]);
}

// test adding two tensors of the same shape
#[test]
fn test_add_same_shape() {
    let tensor1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let tensor2 = Tensor::new(vec![4.0, 5.0, 6.0], vec![1, 3]).unwrap();
    let result = tensor1.add(&tensor2).unwrap();
    assert_eq!(result.data, vec![5.0, 7.0, 9.0]);
    assert_eq!(result.shape, vec![1, 3]);
}

// test adding ones tensor to another tensor
#[test]
fn test_add_ones() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let result = tensor.add(&Tensor::ones(vec![1, 3])).unwrap();
    assert_eq!(result.data, vec![2.0, 3.0, 4.0]);
    assert_eq!(result.shape, vec![1, 3]);
}

// test broadcasting addition with different shapes
#[test]
fn test_add_broadcasting() {
    // test broadcasting in last dimension
    let tensor = Tensor::new(vec![1.0, 4.0, 20.0], vec![1, 3]).unwrap();
    let result = tensor.add(&Tensor::ones(vec![1, 1])).unwrap();
    assert_eq!(result.data, vec![2.0, 5.0, 21.0]);
    assert_eq!(result.shape, vec![1, 3]);

    // test broadcasting in first dimension
    let tensor2 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    let result2 = tensor2.add(&Tensor::ones(vec![1, 3])).unwrap();
    assert_eq!(result2.data, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    assert_eq!(result2.shape, vec![2, 3]);
}

#[test]
fn test_add_broadcasting_example() {
    let patch_embed =
        Tensor::new((0..96).map(|i| (i % 8 + 1) as f32).collect(), vec![1, 96]).unwrap();
    let pos_embed: Tensor = Tensor::ones(vec![1, 65, 96]);
    let out = patch_embed.add(&pos_embed).unwrap();

    // make the expected output tensor ie [2., 3., 4.,  ..., 7., 8., 9.]
    let expected: Vec<f32> = (0..96).map(|i| (i % 8 + 2) as f32).collect();
    // expected needs to repeat 65 times
    let expected: Vec<f32> = (0..65).flat_map(|_| expected.clone()).collect();

    assert_eq!(out.data, expected);
    assert_eq!(out.shape, vec![1, 65, 96]);
}

#[test]
fn test_add_unqueeze() {
    let pos_embed: Tensor = Tensor::ones(vec![1, 65, 96]);
    assert_eq!(pos_embed.shape, vec![1, 65, 96]);
    let out = pos_embed.squeeze(Some(0)).unwrap();
    assert_eq!(out.shape, vec![65, 96]);
    let out = out.unsqueeze(0).unwrap();
    assert_eq!(out.shape, vec![1, 65, 96]);
}

#[test]
fn test_attention_forward() {
    let x: Tensor = Tensor::ones(vec![1, 65, 96]);

    let _batch_size = x.shape[0];
    let seq_len = x.shape[1];
    let hidden_size = x.shape[2];

    let qkv = Tensor::ones(vec![hidden_size, 3 * hidden_size]);
    let qkv_bias = Tensor::ones(vec![3 * hidden_size]);

    // multiply qkv with x
    let qkv_x = x.matmul(&qkv).unwrap();

    // assert that everything is equal to 96
    assert_eq!(qkv_x.shape, vec![1, 65, 3 * hidden_size]);
    let random_values = vec![2, 4, 8, 16, 32, 64];
    for i in random_values {
        assert_eq!(qkv_x.data[i], 96.0);
    }

    let qkv_x = qkv_x.add(&qkv_bias).unwrap();
    let qkv_x = Tensor::split(&qkv_x, 2, &[hidden_size, hidden_size, hidden_size]).unwrap();
    let q = qkv_x[0].clone();
    let k = qkv_x[1].clone();
    let _v = qkv_x[2].clone();

    let q = q.reshape(vec![8, seq_len, hidden_size / 8]).unwrap();
    let k = k.reshape(vec![8, hidden_size / 8, seq_len]).unwrap();

    let _qk = q.matmul(&k).unwrap();

    // assert!(false);
}

#[test]
fn test_attention_forward2() {
    let x: Tensor = Tensor::new(
        (1..65 * 96 + 1).map(|i| i as f32).collect(),
        vec![1, 65, 96],
    )
    .unwrap();

    let _batch_size = x.shape[0];
    let _seq_len = x.shape[1];
    let hidden_size = x.shape[2];

    let qkv = Tensor::ones(vec![hidden_size, 3 * hidden_size]);
    let qkv_bias = Tensor::ones(vec![3 * hidden_size]);
    let qkv_x = x.matmul(&qkv).unwrap();
    let _qkv_x = qkv_x.add(&qkv_bias).unwrap();
    // let qkv_x = qkv_x.permute(&[2, 0, 3, 1, 4]).unwrap();
    // let qkv_x = Tensor::split(&qkv_x, 0, &[1, 1, 1]).unwrap();
    // let q = qkv_x[0].clone().squeeze(Some(0)).unwrap();
    // let k = qkv_x[1].clone().squeeze(Some(0)).unwrap();
    // let v = qkv_x[2].clone().squeeze(Some(0)).unwrap();

    // let k = k.transpose_at(&[2, 3]).unwrap();
    // let attn = q.matmul(&k).unwrap();
    // let attn = attn.scale(0.288_675_13).unwrap();

    // let attn = attn.softmax(Index::Single(2)).unwrap();
    // let x = attn.matmul(&v).unwrap();
    // let x = x.transpose_at(&[1, 2]).unwrap();
    // let x = x.reshape(vec![1, 65, 96]).unwrap();
    // let proj = Tensor::ones(vec![96, 96]);
    // let x = x.matmul(&proj).unwrap();
    // let _x = x.scale(0.5).unwrap();

    // assert!(false);
}

#[test]
fn test_linear_layer() {
    let linear = LinearLayer::new(96, 96);
    let x: Tensor = Tensor::ones(vec![1, 65, 96]);
    let _out = linear.forward(&x).unwrap();

    // assert!(false);
}

#[test]
fn test_mlp_forward() {
    let mlp = Mlp::new(96);
    let x: Tensor = Tensor::ones(vec![1, 65, 96]);
    let _out = mlp.forward(&x).unwrap();

    // assert!(false);
}

#[test]
fn test_layer_norm() {
    let x: Tensor = Tensor::ones(vec![1, 65, 96]);
    let layer_norm = crate::LayerNorm::new(96);
    let _out = layer_norm.forward(&x).unwrap();

    // assert!(false);
}
