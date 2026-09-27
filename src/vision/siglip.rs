use mlx_rs::module::Module;
use mlx_rs::nn::Linear;
use mlx_rs::ops::{softmax_axis, transpose_axes};
use mlx_rs::Array;
use image::GenericImageView;
use crate::config::VlmConfig;
use crate::error::Result;

pub struct VisionAttention {
    pub num_heads: usize,
    pub head_dim: usize,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
}

impl VisionAttention {
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

    pub fn forward(&mut self, x: &Array) -> Result<Array> {
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

        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let scale_arr = Array::from_f32(scale);
        let k_t = transpose_axes(&k_h, &[0, 1, 3, 2])?;
        let scores = q_h.matmul(&k_t)?.multiply(&scale_arr)?;

        let weights = softmax_axis(&scores, -1, None)?;
        let context = weights.matmul(&v_h)?;

        let context_t = transpose_axes(&context, &[0, 2, 1, 3])?;
        let full = context_t.reshape(&[bsz as i32, seq_len as i32, (self.num_heads * self.head_dim) as i32])?;

        Ok(self.out_proj.forward(&full)?)
    }
}

pub struct VisionBlock {
    pub attention: VisionAttention,
    pub mlp1: Linear,
    pub mlp2: Linear,
}

impl VisionBlock {
    pub fn new(hidden_size: usize, num_heads: usize) -> Result<Self> {
        Ok(Self {
            attention: VisionAttention::new(hidden_size, num_heads)?,
            mlp1: Linear::new(hidden_size as i32, (4 * hidden_size) as i32)?,
            mlp2: Linear::new((4 * hidden_size) as i32, hidden_size as i32)?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let attn_out = self.attention.forward(x)?;
        let res1 = x.add(&attn_out)?;
        let norm1 = layer_norm(&res1, 1e-6)?;

        let h = self.mlp1.forward(&norm1)?;
        let act = mlx_rs::nn::gelu(&h)?;
        let ff_out = self.mlp2.forward(&act)?;
        let res2 = norm1.add(&ff_out)?;
        layer_norm(&res2, 1e-6)
    }
}

pub struct SigLipVisionEncoder {
    pub config: VlmConfig,
    pub patch_proj: Linear,
    pub blocks: Vec<VisionBlock>,
}

impl SigLipVisionEncoder {
    pub fn new(config: VlmConfig) -> Result<Self> {
        let patch_dim = config.vision_patch_size * config.vision_patch_size * 3;
        let patch_proj = Linear::new(patch_dim as i32, config.vision_hidden_size as i32)?;

        let mut blocks = Vec::with_capacity(config.vision_num_layers);
        for _ in 0..config.vision_num_layers {
            blocks.push(VisionBlock::new(config.vision_hidden_size, config.vision_num_heads)?);
        }

        Ok(Self {
            config,
            patch_proj,
            blocks,
        })
    }

    /// Extract patches from an input image
    pub fn extract_patches<I: GenericImageView<Pixel = image::Rgb<u8>>>(
        &self,
        image: &I,
    ) -> Result<Array> {
        let (w, h) = image.dimensions();
        let p = self.config.vision_patch_size;
        let num_patches_x = (w as usize) / p;
        let num_patches_y = (h as usize) / p;
        let total_patches = num_patches_x * num_patches_y;
        let patch_dim = p * p * 3;

        let mut data = Vec::with_capacity(total_patches * patch_dim);

        for py in 0..num_patches_y {
            for px in 0..num_patches_x {
                for y in 0..p {
                    for x in 0..p {
                        let pixel = image.get_pixel((px * p + x) as u32, (py * p + y) as u32);
                        data.push((pixel[0] as f32 / 127.5) - 1.0);
                        data.push((pixel[1] as f32 / 127.5) - 1.0);
                        data.push((pixel[2] as f32 / 127.5) - 1.0);
                    }
                }
            }
        }

        Ok(Array::from_slice(&data, &[1, total_patches as i32, patch_dim as i32]))
    }

    /// Forward pass through vision encoder, accepting either [B, H, W, C] or [B, NumPatches, PatchDim]
    pub fn forward(&mut self, image_tensor: &Array) -> Result<Array> {
        let shape = image_tensor.shape();
        let patches = if shape.len() == 4 {
            let bsz = shape[0];
            let p = self.config.vision_patch_size as i32;
            let num_patches_y = shape[1] / p;
            let num_patches_x = shape[2] / p;
            let total_patches = num_patches_y * num_patches_x;
            let patch_dim = p * p * shape[3];

            let r = image_tensor.reshape(&[bsz, num_patches_y, p, num_patches_x, p, shape[3]])?;
            let t = mlx_rs::ops::transpose_axes(&r, &[0, 1, 3, 2, 4, 5])?;
            t.reshape(&[bsz, total_patches, patch_dim])?
        } else {
            image_tensor.clone()
        };

        let mut h = self.patch_proj.forward(&patches)?;
        for block in &mut self.blocks {
            h = block.forward(&h)?;
        }
        Ok(h)
    }

    /// Encode image into visual tokens [1, num_patches, vision_hidden_size]
    pub fn encode_image<I: GenericImageView<Pixel = image::Rgb<u8>>>(
        &mut self,
        image: &I,
    ) -> Result<Array> {
        let patches = self.extract_patches(image)?;
        let mut h = self.patch_proj.forward(&patches)?;

        for block in &mut self.blocks {
            h = block.forward(&h)?;
        }

        Ok(h)
    }
}

pub fn layer_norm(x: &Array, eps: f32) -> Result<Array> {
    let mean = x.mean_axis(-1, true)?;
    let diff = x.subtract(&mean)?;
    let diff_sq = diff.square()?;
    let var = diff_sq.mean_axis(-1, true)?;
    let eps_arr = Array::from_f32(eps);
    let std = var.add(&eps_arr)?.sqrt()?;
    Ok(diff.divide(&std)?)
}
