use mlx_rs::ops::concatenate;
use mlx_rs::Array;
use crate::error::Result;

#[derive(Debug, Default)]
pub struct KVCache {
    pub key_cache: Vec<Option<Array>>,
    pub value_cache: Vec<Option<Array>>,
}

impl KVCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            key_cache: vec![None; num_layers],
            value_cache: vec![None; num_layers],
        }
    }

    pub fn update(&mut self, layer: usize, key: &Array, value: &Array) -> Result<(Array, Array)> {
        let full_key = match &self.key_cache[layer] {
            Some(prev_k) => concatenate(&[prev_k, key], 2)?,
            None => key.clone(),
        };

        let full_val = match &self.value_cache[layer] {
            Some(prev_v) => concatenate(&[prev_v, value], 2)?,
            None => value.clone(),
        };

        self.key_cache[layer] = Some(full_key.clone());
        self.value_cache[layer] = Some(full_val.clone());

        Ok((full_key, full_val))
    }

    pub fn reset(&mut self) {
        for k in &mut self.key_cache {
            *k = None;
        }
        for v in &mut self.value_cache {
            *v = None;
        }
    }
}
