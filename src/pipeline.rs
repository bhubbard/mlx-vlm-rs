use image::imageops::FilterType;
use mlx_rs::Array;
use crate::config::{GenerationConfig, VlmConfig};
use crate::error::Result;
use crate::fusion::MultiModalFusion;
use crate::language::cache::KVCache;
use crate::language::llama::CausalLanguageModel;
use crate::sampler::Sampler;
use crate::vision::{MultiModalProjector, SigLipVisionEncoder};

pub struct VlmPipeline {
    pub config: VlmConfig,
    pub vision_encoder: SigLipVisionEncoder,
    pub projector: MultiModalProjector,
    pub fusion: MultiModalFusion,
    pub language_model: CausalLanguageModel,
    pub kv_cache: KVCache,
}

impl VlmPipeline {
    pub fn new(config: VlmConfig) -> Result<Self> {
        let vision_encoder = SigLipVisionEncoder::new(config.clone())?;
        let projector = MultiModalProjector::new(config.vision_hidden_size, config.text_hidden_size)?;
        let fusion = MultiModalFusion::new(
            config.text_vocab_size,
            config.text_hidden_size,
            config.image_token_id as i32,
        )?;
        let language_model = CausalLanguageModel::new(config.clone())?;
        let kv_cache = KVCache::new(config.text_num_layers);

        Ok(Self {
            config,
            vision_encoder,
            projector,
            fusion,
            language_model,
            kv_cache,
        })
    }

    /// Load and pre-process an image file into a normalized [1, H, W, 3] float Array.
    pub fn load_image(&self, path: &str) -> Result<Array> {
        let img = image::open(path)?;
        let resized = img.resize_exact(
            self.config.vision_image_size as u32,
            self.config.vision_image_size as u32,
            FilterType::Triangle,
        );
        let rgb = resized.to_rgb8();

        let mut data = Vec::with_capacity(
            self.config.vision_image_size * self.config.vision_image_size * 3,
        );
        for pixel in rgb.pixels() {
            // Normalize standard ImageNet / SigLIP RGB values to [-1, 1] or [0, 1]
            data.push((pixel[0] as f32 / 255.0 - 0.5) / 0.5);
            data.push((pixel[1] as f32 / 255.0 - 0.5) / 0.5);
            data.push((pixel[2] as f32 / 255.0 - 0.5) / 0.5);
        }

        let arr = Array::from_slice(
            &data,
            &[
                1,
                self.config.vision_image_size as i32,
                self.config.vision_image_size as i32,
                3,
            ],
        );
        Ok(arr)
    }

    /// Reset KV cache for a fresh generation session
    pub fn reset(&mut self) {
        self.kv_cache.reset();
    }

    /// Encode visual input and project into text embedding dimension
    pub fn encode_image(&mut self, image: &Array) -> Result<Array> {
        let visual_features = self.vision_encoder.forward(image)?;
        self.projector.forward(&visual_features)
    }

    /// Generate text autoregressively given optional image and prompt tokens
    pub fn generate<F>(
        &mut self,
        image: Option<&Array>,
        prompt_tokens: &[i32],
        gen_config: &GenerationConfig,
        mut token_callback: F,
    ) -> Result<Vec<i32>>
    where
        F: FnMut(i32) -> bool,
    {
        self.reset();

        // 1. Process image if present
        let vision_embeds = match image {
            Some(img) => Some(self.encode_image(img)?),
            None => None,
        };

        // 2. Embed prompt text tokens
        let token_ids_arr = Array::from_slice(prompt_tokens, &[1, prompt_tokens.len() as i32]);
        let token_embeds = self.fusion.embed_tokens(&token_ids_arr)?;

        // 3. Multi-modal fusion
        let fused_embeds = self.fusion.fuse(
            prompt_tokens,
            &token_embeds,
            vision_embeds.as_ref(),
        )?;

        // 4. Prefill pass through Causal Language Model
        let logits = self.language_model.forward(&fused_embeds, Some(&mut self.kv_cache))?;

        let mut generated: Vec<i32> = Vec::new();
        let mut all_tokens: Vec<i32> = prompt_tokens.to_vec();

        // Sample first token
        let first_token = Sampler::sample(&logits, gen_config, &all_tokens)?;
        generated.push(first_token);
        all_tokens.push(first_token);

        let continue_gen = token_callback(first_token);
        if !continue_gen || gen_config.stop_token_ids.contains(&first_token) {
            return Ok(generated);
        }

        // 5. Autoregressive decoding loop using KV cache
        for _ in 1..gen_config.max_new_tokens {
            let last_token_arr = Array::from_slice(&[all_tokens[all_tokens.len() - 1]], &[1, 1]);
            let last_embed = self.fusion.embed_tokens(&last_token_arr)?;

            let next_logits = self
                .language_model
                .forward(&last_embed, Some(&mut self.kv_cache))?;

            let next_token = Sampler::sample(&next_logits, gen_config, &all_tokens)?;
            generated.push(next_token);
            all_tokens.push(next_token);

            let keep_going = token_callback(next_token);
            if !keep_going || gen_config.stop_token_ids.contains(&next_token) {
                break;
            }
        }

        Ok(generated)
    }
}
