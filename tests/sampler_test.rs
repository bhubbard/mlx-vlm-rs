use mlx_rs::Array;
use mlx_vlm_rs::config::GenerationConfig;
use mlx_vlm_rs::sampler::Sampler;

#[test]
fn test_greedy_sampler() -> anyhow::Result<()> {
    let logits_data = vec![0.1f32, 5.0, 0.2, 0.3];
    let logits = Array::from_slice(&logits_data, &[1, 4]);

    let config = GenerationConfig {
        temperature: 0.0,
        ..Default::default()
    };

    let sampled = Sampler::sample(&logits, &config, &[])?;
    assert_eq!(sampled, 1);
    Ok(())
}

#[test]
fn test_repetition_penalty() -> anyhow::Result<()> {
    // Token 1 has highest logit (3.0), token 0 has 2.0
    let logits_data = vec![2.0f32, 3.0, 0.1, 0.1];
    let logits = Array::from_slice(&logits_data, &[1, 4]);

    // Heavily penalize token 1 because it's in past_tokens
    let config = GenerationConfig {
        temperature: 0.0,
        repetition_penalty: 2.0, // 3.0 / 2.0 = 1.5 < 2.0 (token 0)
        ..Default::default()
    };

    let sampled = Sampler::sample(&logits, &config, &[1])?;
    assert_eq!(sampled, 0); // Token 0 should win after penalty on token 1
    Ok(())
}
