use mlx_rs::module::Module;
use mlx_rs::nn::Linear;
use mlx_rs::Array;
use crate::error::Result;

pub struct MultiModalProjector {
    pub linear1: Linear,
    pub linear2: Linear,
}

impl MultiModalProjector {
    pub fn new(vision_dim: usize, text_dim: usize) -> Result<Self> {
        Ok(Self {
            linear1: Linear::new(vision_dim as i32, text_dim as i32)?,
            linear2: Linear::new(text_dim as i32, text_dim as i32)?,
        })
    }

    pub fn forward(&mut self, vision_tokens: &Array) -> Result<Array> {
        let h = self.linear1.forward(vision_tokens)?;
        let act = mlx_rs::nn::gelu(&h)?;
        Ok(self.linear2.forward(&act)?)
    }
}
