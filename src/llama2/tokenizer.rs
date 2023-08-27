
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{self, Read, BufReader};

use byteorder::{LittleEndian, ReadBytesExt};

pub struct Tokenizer {
    vocab: Vec<String>,
    vocab_scores: Vec<f32>,
    vocab_index: HashMap<String, usize>,
    max_token_length: usize,
    byte_pieces: Vec<char>, // stores all single-byte strings
}

impl Tokenizer {

	pub fn read(file_name: &str, vocab_size: usize) -> io::Result<Tokenizer> {
        let mut vocab: Vec<String> = Vec::with_capacity(vocab_size);
        let mut vocab_scores: Vec<f32> = Vec::with_capacity(vocab_size);

        let file = File::open(file_name)?;
        let mut reader = BufReader::new(file);

        let max_token_length = reader.read_u32::<LittleEndian>()
            .map_err(|_| to_io_err("Failed to read max token length"))? as usize;

        for _ in 0..vocab_size {
            let vocab_score = reader.read_f32::<LittleEndian>()
                .map_err(|_| to_io_err("Failed to read vocab score"))?;
            vocab_scores.push(vocab_score);

            let len = reader.read_u32::<LittleEndian>()
                .map_err(|_| to_io_err("Failed to read token length"))? as usize;
            let mut token_buffer = vec![0u8; len];
            reader.read_exact(&mut token_buffer)
                .map_err(|_| to_io_err("Failed to read token"))?;
            let token = String::from_utf8(token_buffer)
                .map_err(|_| to_io_err("Failed to convert bytes to UTF-8 string"))?;

            vocab.push(token);
        }
		let byte_pieces: Vec<char> = (0..=256).map(|i| i as u8 as char).collect();

		let mut vocab_index = HashMap::new();
        for n in 0..vocab.len() {
            vocab_index.insert(vocab[n].clone(), n);
        }

        Ok(Tokenizer {
            vocab,
            vocab_scores,
            vocab_index,
            max_token_length,
            byte_pieces,
        })
    }

	pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<usize> {
		let mut tokens: Vec<usize> = Vec::new();
		if bos {
			tokens.push(1);
		}
		if !text.is_empty() {
			let dummy_prefix = self.vocab_index.get(" ").unwrap_or(&0);
			tokens.push(*dummy_prefix);
		}

		for ch in text.chars() {
			let ch_str = ch.to_string();
			match self.vocab_index.get(&ch_str) {
				Some(&id) => tokens.push(id),
				None => {
                    // byte_fallback encoding: just encode each byte as a token
                    // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
                    // so the individual bytes only start at index 3
					for byte in ch_str.as_bytes() {
						tokens.push(*byte as usize + 3);
					}
				}
			}
		}

        // merge the best consecutive pair each iteration, according the scores in vocab_scores
		loop {
			let mut best_score = f32::NEG_INFINITY;
			let mut best_id = 0;
			let mut best_idx = None;

			for i in 0..(tokens.len() - 1) {
				let pair = format!("{}{}", self.vocab[tokens[i]], self.vocab[tokens[i + 1]]);
				if let Some(&id) = self.vocab_index.get(&pair) {
					if self.vocab_scores[id] > best_score {
						best_score = self.vocab_scores[id];
						best_id = id;
						best_idx = Some(i);
					}
				}
			}

			if let Some(idx) = best_idx {
				tokens[idx] = best_id;
				tokens.remove(idx + 1);
			} else {
				break;
			}
		}

		if eos {
			tokens.push(2);
		}

		tokens
	}

    pub fn decode(&self, prev_token: usize, token: usize) -> String {
        let mut piece = self.vocab[token].as_str();
        if prev_token == 1 {
            piece = piece.trim_start();
        }
        if let Some(hex) = piece.strip_prefix("<0x") {
            if let Ok(byte) = usize::from_str_radix(&hex[..2], 16) {
                return self.byte_pieces[byte].to_string();
            }
        }
        piece.to_string()
    }
}

impl fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tokenizer with vocab size: {}", self.vocab.len())
    }
}

fn to_io_err(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}
