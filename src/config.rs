use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VlmArch {
    PaliGemma,
    Qwen2VL,
    SmolVLM,
    Llava,
    Tiny,
}

pub type VlmArchitecture = VlmArch;

impl Default for VlmArch {
    fn default() -> Self {
        Self::PaliGemma
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VlmConfig {
    pub name: String,
    pub arch: VlmArch,
    // Vision encoder specs
    pub vision_hidden_size: usize,
    pub vision_num_layers: usize,
    pub vision_num_heads: usize,
    pub vision_patch_size: usize,
    pub vision_image_size: usize,
    // Language model specs
    pub text_vocab_size: usize,
    pub text_hidden_size: usize,
    pub text_num_layers: usize,
    pub text_num_heads: usize,
    pub text_intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub image_token_id: u32,
}

impl VlmConfig {
    pub fn preset(arch: VlmArch) -> Self {
        match arch {
            VlmArch::PaliGemma => Self::paligemma_3b(),
            VlmArch::Qwen2VL => Self::qwen2_vl_2b(),
            VlmArch::SmolVLM => Self::smolvlm_500m(),
            VlmArch::Llava => Self::llava_1_5(),
            VlmArch::Tiny => Self::tiny(),
        }
    }

    pub fn num_patches(&self) -> usize {
        (self.vision_image_size / self.vision_patch_size).pow(2)
    }

    pub fn paligemma_3b() -> Self {
        Self {
            name: "paligemma-3b-pt-224".to_string(),
            arch: VlmArch::PaliGemma,
            vision_hidden_size: 1152,
            vision_num_layers: 27,
            vision_num_heads: 16,
            vision_patch_size: 14,
            vision_image_size: 224,
            text_vocab_size: 257216,
            text_hidden_size: 2048,
            text_num_layers: 18,
            text_num_heads: 8,
            text_intermediate_size: 16384,
            max_position_embeddings: 8192,
            image_token_id: 257152,
        }
    }

    pub fn qwen2_vl_2b() -> Self {
        Self {
            name: "qwen2-vl-2b-instruct".to_string(),
            arch: VlmArch::Qwen2VL,
            vision_hidden_size: 1280,
            vision_num_layers: 32,
            vision_num_heads: 16,
            vision_patch_size: 14,
            vision_image_size: 448,
            text_vocab_size: 152064,
            text_hidden_size: 1536,
            text_num_layers: 28,
            text_num_heads: 12,
            text_intermediate_size: 8960,
            max_position_embeddings: 32768,
            image_token_id: 151655,
        }
    }

    pub fn smolvlm_500m() -> Self {
        Self {
            name: "smolvlm-500m-instruct".to_string(),
            arch: VlmArch::SmolVLM,
            vision_hidden_size: 768,
            vision_num_layers: 12,
            vision_num_heads: 12,
            vision_patch_size: 16,
            vision_image_size: 384,
            text_vocab_size: 49152,
            text_hidden_size: 960,
            text_num_layers: 16,
            text_num_heads: 15,
            text_intermediate_size: 2560,
            max_position_embeddings: 8192,
            image_token_id: 49150,
        }
    }

    pub fn llava_1_5() -> Self {
        Self {
            name: "llava-1.5-7b".to_string(),
            arch: VlmArch::Llava,
            vision_hidden_size: 1024,
            vision_num_layers: 24,
            vision_num_heads: 16,
            vision_patch_size: 14,
            vision_image_size: 336,
            text_vocab_size: 32000,
            text_hidden_size: 4096,
            text_num_layers: 32,
            text_num_heads: 32,
            text_intermediate_size: 11008,
            max_position_embeddings: 4096,
            image_token_id: 32000,
        }
    }

    pub fn tiny() -> Self {
        Self {
            name: "tiny-vlm".to_string(),
            arch: VlmArch::Tiny,
            vision_hidden_size: 64,
            vision_num_layers: 1,
            vision_num_heads: 2,
            vision_patch_size: 16,
            vision_image_size: 32,
            text_vocab_size: 1000,
            text_hidden_size: 64,
            text_num_layers: 1,
            text_num_heads: 2,
            text_intermediate_size: 128,
            max_position_embeddings: 128,
            image_token_id: 999,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub repetition_penalty: f32,
    pub stop_token_ids: Vec<i32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            temperature: 0.7,
            top_p: 0.9,
            top_k: Some(50),
            repetition_penalty: 1.1,
            stop_token_ids: vec![1, 2], // common EOS tokens
        }
    }
}
