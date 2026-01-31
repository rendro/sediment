//! Retry utilities with exponential backoff
//!
//! Provides generic retry logic for transient failures in async operations.

use std::fmt::Display;
use std::future::Future;
use std::time::Duration;

use tokio::time::sleep;
use tracing::{debug, warn};

/// Default maximum number of retry attempts
const DEFAULT_MAX_ATTEMPTS: u32 = 3;

/// Default initial delay between retries (in milliseconds)
const DEFAULT_INITIAL_DELAY_MS: u64 = 100;

/// Default maximum delay between retries (in milliseconds)
const DEFAULT_MAX_DELAY_MS: u64 = 2000;

/// Configuration for retry behavior
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of attempts (including the initial attempt)
    pub max_attempts: u32,
    /// Initial delay between retries in milliseconds
    pub initial_delay_ms: u64,
    /// Maximum delay between retries in milliseconds (caps exponential growth)
    pub max_delay_ms: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: DEFAULT_MAX_ATTEMPTS,
            initial_delay_ms: DEFAULT_INITIAL_DELAY_MS,
            max_delay_ms: DEFAULT_MAX_DELAY_MS,
        }
    }
}

impl RetryConfig {
    /// Create a new retry configuration with custom values
    pub fn new(max_attempts: u32, initial_delay_ms: u64, max_delay_ms: u64) -> Self {
        Self {
            max_attempts,
            initial_delay_ms,
            max_delay_ms,
        }
    }

    /// Calculate the delay for a given attempt number (0-indexed)
    fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let delay_ms = self
            .initial_delay_ms
            .saturating_mul(1u64.checked_shl(attempt).unwrap_or(u64::MAX));
        let capped_delay_ms = delay_ms.min(self.max_delay_ms);
        Duration::from_millis(capped_delay_ms)
    }
}

/// Execute an async operation with exponential backoff retry.
///
/// The operation is retried up to `config.max_attempts` times on failure.
/// The delay between retries grows exponentially, starting at `initial_delay_ms`
/// and capped at `max_delay_ms`.
///
/// # Arguments
///
/// * `config` - Retry configuration
/// * `operation` - A closure that returns a Future yielding Result<T, E>
///
/// # Returns
///
/// The result of the successful operation, or the last error if all attempts fail.
///
/// # Example
///
/// ```ignore
/// use sediment::retry::{with_retry, RetryConfig};
///
/// let result = with_retry(&RetryConfig::default(), || async {
///     // Your fallible async operation here
///     Ok::<_, String>("success")
/// }).await;
/// ```
pub async fn with_retry<T, E, F, Fut>(config: &RetryConfig, operation: F) -> Result<T, E>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: Display,
{
    let mut last_error: Option<E> = None;

    for attempt in 0..config.max_attempts {
        match operation().await {
            Ok(result) => {
                if attempt > 0 {
                    debug!("Operation succeeded on attempt {}", attempt + 1);
                }
                return Ok(result);
            }
            Err(e) => {
                let is_last_attempt = attempt + 1 >= config.max_attempts;

                if is_last_attempt {
                    warn!(
                        "Operation failed after {} attempts: {}",
                        config.max_attempts, e
                    );
                    last_error = Some(e);
                } else {
                    let delay = config.delay_for_attempt(attempt);
                    warn!(
                        "Operation failed (attempt {}/{}): {}. Retrying in {:?}...",
                        attempt + 1,
                        config.max_attempts,
                        e,
                        delay
                    );
                    sleep(delay).await;
                    last_error = Some(e);
                }
            }
        }
    }

    // Return the last error
    Err(last_error.expect("at least one attempt should have been made"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[tokio::test]
    async fn test_retry_success_first_attempt() {
        let config = RetryConfig::default();
        let result: Result<&str, &str> = with_retry(&config, || async { Ok("success") }).await;
        assert_eq!(result, Ok("success"));
    }

    #[tokio::test]
    async fn test_retry_success_after_failures() {
        let config = RetryConfig::new(3, 10, 100); // Short delays for testing
        let attempt_count = Arc::new(AtomicU32::new(0));
        let attempt_count_clone = attempt_count.clone();

        let result: Result<&str, &str> = with_retry(&config, || {
            let count = attempt_count_clone.clone();
            async move {
                let current = count.fetch_add(1, Ordering::SeqCst);
                if current < 2 {
                    Err("transient error")
                } else {
                    Ok("success")
                }
            }
        })
        .await;

        assert_eq!(result, Ok("success"));
        assert_eq!(attempt_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retry_all_failures() {
        let config = RetryConfig::new(3, 10, 100); // Short delays for testing
        let attempt_count = Arc::new(AtomicU32::new(0));
        let attempt_count_clone = attempt_count.clone();

        let result: Result<&str, &str> = with_retry(&config, || {
            let count = attempt_count_clone.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Err("persistent error")
            }
        })
        .await;

        assert_eq!(result, Err("persistent error"));
        assert_eq!(attempt_count.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_delay_for_attempt_no_overflow() {
        // Bug #10: large attempt numbers should not panic due to overflow
        let config = RetryConfig::new(100, 100, 2000);
        // These should not panic
        let d64 = config.delay_for_attempt(64);
        let d100 = config.delay_for_attempt(99);
        // Should be capped at max_delay_ms
        assert_eq!(d64, Duration::from_millis(2000));
        assert_eq!(d100, Duration::from_millis(2000));
    }

    #[test]
    fn test_delay_calculation() {
        let config = RetryConfig::new(5, 100, 1000);

        assert_eq!(config.delay_for_attempt(0), Duration::from_millis(100));
        assert_eq!(config.delay_for_attempt(1), Duration::from_millis(200));
        assert_eq!(config.delay_for_attempt(2), Duration::from_millis(400));
        assert_eq!(config.delay_for_attempt(3), Duration::from_millis(800));
        assert_eq!(config.delay_for_attempt(4), Duration::from_millis(1000)); // Capped
    }
}
