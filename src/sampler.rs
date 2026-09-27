use rand::Rng;
use mlx_rs::Array;
use crate::config::GenerationConfig;
use crate::error::Result;

pub struct Sampler;

impl Sampler {
    /// Sample next token ID from logits array of shape [1, vocab_size] or [vocab_size]
    pub fn sample(logits_arr: &Array, config: &GenerationConfig, past_tokens: &[i32]) -> Result<i32> {
        let mut logits: Vec<f32> = logits_arr.as_slice::<f32>().to_vec();
        let vocab_size = logits.len();

        // Repetition penalty
        if (config.repetition_penalty - 1.0).abs() > 1e-4 {
            for &token in past_tokens {
                let idx = token as usize;
                if idx < vocab_size {
                    if logits[idx] > 0.0 {
                        logits[idx] /= config.repetition_penalty;
                    } else {
                        logits[idx] *= config.repetition_penalty;
                    }
                }
            }
        }

        // Greedy decoding if temperature is near 0
        if config.temperature <= 1e-5 {
            let max_idx = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            return Ok(max_idx as i32);
        }

        // Apply temperature
        let inv_temp = 1.0 / config.temperature;
        for val in &mut logits {
            *val *= inv_temp;
        }

        // Softmax
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        let mut probs: Vec<f32> = logits
            .iter()
            .map(|&l| {
                let e = (l - max_val).exp();
                exp_sum += e;
                e
            })
            .collect();

        if exp_sum > 0.0 {
            for p in &mut probs {
                *p /= exp_sum;
            }
        }

        // Indexed pairs (idx, prob)
        let mut indexed_probs: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
        indexed_probs.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Top-K filtering
        if let Some(top_k) = config.top_k {
            if top_k > 0 && top_k < indexed_probs.len() {
                indexed_probs.truncate(top_k);
            }
        }

        // Top-P (nucleus) filtering
        if config.top_p < 1.0 {
            let mut cum_prob = 0.0f32;
            let mut cutoff = indexed_probs.len();
            for (i, (_, p)) in indexed_probs.iter().enumerate() {
                cum_prob += p;
                if cum_prob >= config.top_p {
                    cutoff = (i + 1).min(indexed_probs.len());
                    break;
                }
            }
            indexed_probs.truncate(cutoff);
        }

        // Re-normalize probabilities
        let total_p: f32 = indexed_probs.iter().map(|(_, p)| *p).sum();
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen::<f32>() * total_p;

        let mut acc = 0.0f32;
        for (idx, p) in indexed_probs {
            acc += p;
            if r <= acc {
                return Ok(idx as i32);
            }
        }

        Ok(0)
    }
}
