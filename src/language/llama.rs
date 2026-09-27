use mlx_rs::module::Module;
use mlx_rs::nn::Linear;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{softmax_axis, transpose_axes};
use mlx_rs::Array;
use crate::config::VlmConfig;
use crate::error::Result;
use crate::language::cache::KVCache;

pub struct CausalSelfAttention {
    pub num_heads: usize,
    pub head_dim: usize,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
}

impl CausalSelfAttention {
    pub fn new(hidden_size: usize, num_heads: usize) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        Ok(Self {
            num_heads,
            head_dim,
            q_proj: Linear::new(hidden_size as i32, hidden_size as i32)?,
            k_proj: Linear::new(hidden_size as i32, hidden_size as i32)?,
            v_proj: Linear::new(hidden_size as i32, hidden_size as i32)?,
            out_proj: Linear::new(hidden_size as i32, hidden_size as i32)?,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        layer_idx: usize,
        cache: Option<&mut KVCache>,
    ) -> Result<Array> {
        let shape = x.shape();
        let bsz = shape[0] as usize;
        let seq_len = shape[1] as usize;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q_s = q.reshape(&[bsz as i32, seq_len as i32, self.num_heads as i32, self.head_dim as i32])?;
        let q_h = transpose_axes(&q_s, &[0, 2, 1, 3])?;

        let k_s = k.reshape(&[bsz as i32, seq_len as i32, self.num_heads as i32, self.head_dim as i32])?;
        let k_h = transpose_axes(&k_s, &[0, 2, 1, 3])?;

        let v_s = v.reshape(&[bsz as i32, seq_len as i32, self.num_heads as i32, self.head_dim as i32])?;
        let v_h = transpose_axes(&v_s, &[0, 2, 1, 3])?;

        // Update KV cache if provided
        let (full_k, full_v) = if let Some(c) = cache {
            c.update(layer_idx, &k_h, &v_h)?
        } else {
            (k_h, v_h)
        };

        let kv_seq_len = full_k.shape()[2] as usize;
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let scale_arr = Array::from_f32(scale);
        let k_t = transpose_axes(&full_k, &[0, 1, 3, 2])?;
        let mut scores = q_h.matmul(&k_t)?.multiply(&scale_arr)?;

        // Apply causal mask for autoregressive sequence if seq_len > 1
        if seq_len > 1 {
            let mut mask_data = vec![0.0f32; seq_len * kv_seq_len];
            for q_idx in 0..seq_len {
                for k_idx in 0..kv_seq_len {
                    if k_idx > (kv_seq_len - seq_len) + q_idx {
                        mask_data[q_idx * kv_seq_len + k_idx] = -10000.0f32;
                    }
                }
            }
            let mask_arr = Array::from_slice(&mask_data, &[1, 1, seq_len as i32, kv_seq_len as i32]);
            scores = scores.add(&mask_arr)?;
        }

        let weights = softmax_axis(&scores, -1, None)?;
        let context = weights.matmul(&full_v)?;

        let context_t = transpose_axes(&context, &[0, 2, 1, 3])?;
        let full = context_t.reshape(&[bsz as i32, seq_len as i32, (self.num_heads * self.head_dim) as i32])?;

        Ok(self.out_proj.forward(&full)?)
    }
}

pub struct CausalDecoderLayer {
    pub attention: CausalSelfAttention,
    pub mlp1: Linear,
    pub mlp2: Linear,
    pub layer_idx: usize,
}

impl CausalDecoderLayer {
    pub fn new(hidden_size: usize, num_heads: usize, intermediate: usize, layer_idx: usize) -> Result<Self> {
        Ok(Self {
            attention: CausalSelfAttention::new(hidden_size, num_heads)?,
            mlp1: Linear::new(hidden_size as i32, intermediate as i32)?,
            mlp2: Linear::new(intermediate as i32, hidden_size as i32)?,
            layer_idx,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        cache: Option<&mut KVCache>,
    ) -> Result<Array> {
        let attn_out = self.attention.forward(x, self.layer_idx, cache)?;
        let res1 = x.add(&attn_out)?;
        let norm1 = rms_norm(&res1, 1e-6)?;

        let h = self.mlp1.forward(&norm1)?;
        let act = mlx_rs::nn::silu(&h)?;
        let ff_out = self.mlp2.forward(&act)?;
        let res2 = norm1.add(&ff_out)?;
        rms_norm(&res2, 1e-6)
    }
}

pub struct CausalLanguageModel {
    pub config: VlmConfig,
    pub layers: Vec<CausalDecoderLayer>,
    pub lm_head: Linear,
}

impl CausalLanguageModel {
    pub fn new(config: VlmConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.text_num_layers);
        for i in 0..config.text_num_layers {
            layers.push(CausalDecoderLayer::new(
                config.text_hidden_size,
                config.text_num_heads,
                config.text_intermediate_size,
                i,
            )?);
        }

        let lm_head = Linear::new(config.text_hidden_size as i32, config.text_vocab_size as i32)?;

        Ok(Self {
            config,
            layers,
            lm_head,
        })
    }

    pub fn forward(
        &mut self,
        inputs_embeds: &Array,
        mut cache: Option<&mut KVCache>,
    ) -> Result<Array> {
        let mut h = inputs_embeds.clone();

        for layer in &mut self.layers {
            let c = cache.as_deref_mut();
            h = layer.forward(&h, c)?;
        }

        // Project last token to vocabulary logits [1, vocab_size]
        let seq_len = h.shape()[1] as usize;
        let last_hidden = h.index((.., (seq_len - 1) as i32..seq_len as i32, ..));
        Ok(self.lm_head.forward(&last_hidden)?)
    }
}

pub fn rms_norm(x: &Array, eps: f32) -> Result<Array> {
    let x_sq = x.square()?;
    let mean_sq = x_sq.mean_axis(-1, true)?;
    let eps_arr = Array::from_f32(eps);
    let rms = mean_sq.add(&eps_arr)?.sqrt()?;
    Ok(x.divide(&rms)?)
}
