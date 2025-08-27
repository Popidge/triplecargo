#![forbid(unsafe_code)]
#![deny(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)] // may be revisited

pub mod types;
pub mod rules;
pub mod cards;
pub mod board;
pub mod state;
pub mod hash;

pub mod engine {
    pub mod apply;
    pub mod score;
}

pub mod solver;

// Re-exports: stable minimal API surface for external callers
pub use crate::cards::{load_cards_from_json, Card, CardsDb};
pub use crate::engine::apply::apply_move;
pub use crate::engine::score::score;
pub use crate::hash::zobrist_key;
pub use crate::rules::Rules;
pub use crate::state::{is_terminal, legal_moves, GameState, Move};
pub use crate::types::{Element, Owner};
pub use crate::board::Board;