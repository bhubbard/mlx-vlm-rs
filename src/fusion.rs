use mlx_rs::module::Module;
use mlx_rs::nn::Embedding;
use mlx_rs::ops::concatenate;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;
use crate::error::Result;

#[derive(Debug)]
pub struct MultiModalFusion {
    pub image_token_id: i32,
    pub text_embedding: Embedding,
}

impl MultiModalFusion {
    pub fn new(vocab_size: usize, hidden_size: usize, image_token_id: i32) -> Result<Self> {
        Ok(Self {
            image_token_id,
            text_embedding: Embedding::new(vocab_size as i32, hidden_size as i32)?,
        })
    }

    /// Embed text token IDs into dense vectors [batch, seq_len, hidden_size]
    pub fn embed_tokens(&mut self, token_ids: &Array) -> Result<Array> {
        Ok(self.text_embedding.forward(token_ids)?)
    }

    /// Fuses vision embeddings with text embeddings.
    /// If token_ids contains image_token_id, vision embeddings replace that placeholder token.
    /// Otherwise, vision embeddings are prepended to text embeddings (prefix fusion).
    pub fn fuse(
        &mut self,
        tokens: &[i32],
        token_embeds: &Array,
        vision_embeds: Option<&Array>,
    ) -> Result<Array> {
        let vision = match vision_embeds {
            Some(v) => v,
            None => return Ok(token_embeds.clone()),
        };

        // Check if image token placeholder exists in prompt
        if let Some(pos) = tokens.iter().position(|&t| t == self.image_token_id) {
            let total_len = tokens.len();

            let mut segments: Vec<Array> = Vec::new();
            if pos > 0 {
                let prefix = token_embeds.index((.., 0..pos as i32, ..));
                segments.push(prefix);
            }

            segments.push(vision.clone());

            if pos + 1 < total_len {
                let suffix = token_embeds.index((.., (pos + 1) as i32..total_len as i32, ..));
                segments.push(suffix);
            }

            let seg_refs: Vec<&Array> = segments.iter().collect();
            Ok(concatenate(&seg_refs, 1)?)
        } else {
            // Prefix fusion: [vision_embeds, text_embeds]
            Ok(concatenate(&[vision, token_embeds], 1)?)
        }
    }
}
