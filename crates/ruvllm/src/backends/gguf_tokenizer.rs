//! Build a HuggingFace tokenizer from the vocabulary a GGUF file embeds, so a
//! bare `.gguf` loads without a `tokenizer.json` next to it.
//!
//! Supported: `tokenizer.ggml.model = "gpt2"` — byte-level BPE (Qwen2, Llama 3,
//! SmolLM2, GPT-2 style vocabularies): `tokenizer.ggml.tokens` +
//! `tokenizer.ggml.merges`, with control tokens (`token_type == 3`) registered as
//! special tokens. Other tokenizer models (SentencePiece `"llama"`, …) are
//! reported as unsupported so the caller can say so; a `tokenizer.json` next to
//! the file still takes precedence for every model.

use std::collections::HashMap;

use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer};

/// GGUF `token_type` of a control token (`<|im_start|>`, `<|eot_id|>`, …).
const TOKEN_TYPE_CONTROL: i32 = 3;

/// Why an embedded tokenizer could not be built.
#[derive(Debug, PartialEq, Eq)]
pub enum GgufTokenizerError {
    /// The file declares a tokenizer model this module does not build.
    Unsupported(String),
    /// Required metadata is missing or malformed.
    Invalid(String),
}

impl std::fmt::Display for GgufTokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(m) => write!(
                f,
                "GGUF embeds a '{m}' tokenizer, which ruvllm cannot build yet; \
                 put the model's tokenizer.json next to the .gguf"
            ),
            Self::Invalid(m) => write!(f, "invalid GGUF tokenizer metadata: {m}"),
        }
    }
}

/// Build a byte-level BPE tokenizer from GGUF vocabulary parts.
///
/// `tokens[i]` is token id `i`; `merges` are `"left right"` pairs in rank
/// order; `token_types`, when present, marks control tokens as special.
pub fn from_gguf_parts(
    model: &str,
    tokens: &[String],
    merges: &[String],
    token_types: Option<&[i32]>,
) -> Result<Tokenizer, GgufTokenizerError> {
    if model != "gpt2" {
        return Err(GgufTokenizerError::Unsupported(model.to_string()));
    }
    if tokens.is_empty() {
        return Err(GgufTokenizerError::Invalid(
            "tokenizer.ggml.tokens is empty".into(),
        ));
    }
    let vocab: HashMap<String, u32> = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i as u32))
        .collect();
    let merges: Vec<(String, String)> = merges
        .iter()
        .filter_map(|m| {
            let (a, b) = m.split_once(' ')?;
            Some((a.to_string(), b.to_string()))
        })
        .collect();
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| GgufTokenizerError::Invalid(e.to_string()))?;
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
    tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
    if let Some(types) = token_types {
        let special: Vec<AddedToken> = tokens
            .iter()
            .zip(types)
            .filter(|(_, &t)| t == TOKEN_TYPE_CONTROL)
            .map(|(tok, _)| AddedToken::from(tok.clone(), true))
            .collect();
        tokenizer.add_special_tokens(&special);
    }
    Ok(tokenizer)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy byte-level vocabulary: single bytes (GPT-2 byte-to-unicode map
    /// for ASCII letters and space `Ġ`), a few merges, and one control token.
    fn toy() -> (Vec<String>, Vec<String>, Vec<i32>) {
        let mut tokens: Vec<String> = "abcdehlorw".chars().map(|c| c.to_string()).collect();
        tokens.push("Ġ".into());
        for t in [
            "he", "ll", "hell", "hello", "Ġw", "or", "Ġwor", "Ġworl", "Ġworld", "<|eot|>",
        ] {
            tokens.push(t.into());
        }
        let merges = [
            "h e", "l l", "he ll", "hell o", "Ġ w", "o r", "Ġw or", "Ġwor l", "Ġworl d",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let mut types = vec![1; tokens.len()];
        *types.last_mut().unwrap() = TOKEN_TYPE_CONTROL;
        (tokens, merges, types)
    }

    #[test]
    fn builds_a_byte_level_bpe_that_round_trips() {
        let (tokens, merges, types) = toy();
        let tok = from_gguf_parts("gpt2", &tokens, &merges, Some(&types)).unwrap();
        let enc = tok.encode("hello world", false).unwrap();
        assert_eq!(enc.get_tokens(), ["hello", "Ġworld"]);
        let ids = enc.get_ids().to_vec();
        assert_eq!(tok.decode(&ids, false).unwrap(), "hello world");
    }

    #[test]
    fn control_tokens_are_special() {
        let (tokens, merges, types) = toy();
        let tok = from_gguf_parts("gpt2", &tokens, &merges, Some(&types)).unwrap();
        let enc = tok.encode("hello<|eot|>", false).unwrap();
        assert_eq!(enc.get_tokens(), ["hello", "<|eot|>"]);
        let eot = tok.token_to_id("<|eot|>").unwrap();
        assert_eq!(eot as usize, tokens.len() - 1);
    }

    #[test]
    fn sentencepiece_models_are_reported_as_unsupported() {
        let (tokens, merges, _) = toy();
        let err = from_gguf_parts("llama", &tokens, &merges, None).unwrap_err();
        assert_eq!(err, GgufTokenizerError::Unsupported("llama".into()));
        assert!(err.to_string().contains("tokenizer.json"));
    }
}
