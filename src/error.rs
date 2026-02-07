use thiserror::Error;

#[derive(Error, Debug)]
pub enum SedimentError {
    #[error("Database error: {0}")]
    Database(String),

    #[error("LanceDB error: {0}")]
    LanceDb(#[from] lancedb::Error),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid content type: {0}")]
    InvalidContentType(String),

    #[error("Model loading error: {0}")]
    ModelLoading(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
}

pub type Result<T> = std::result::Result<T, SedimentError>;
