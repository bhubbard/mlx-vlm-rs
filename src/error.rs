use thiserror::Error;

#[derive(Error, Debug)]
pub enum VlmError {
    #[error("MLX error: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Shape mismatch: expected {expected}, found {found}")]
    ShapeMismatch { expected: String, found: String },

    #[error("Generation error: {0}")]
    Generation(String),
}

pub type Result<T> = std::result::Result<T, VlmError>;
