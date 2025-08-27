use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use crate::cards::{Card, CardsDb};
use crate::rules::Rules;
use crate::state::Move;
use crate::types::{Element, Owner};

pub const FORMAT_VERSION: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ElementsMode {
    None,
    Random,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SolvedEntry {
    pub value: i8,               // minimax value from side-to-move perspective
    pub best_move: Option<Move>, // deterministic move representation
    pub depth: u8,               // remaining depth at storage time
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbHeader {
    pub version: u32,
    pub rules: Rules,
    pub elements_mode: ElementsMode,
    pub seed: u64,
    pub start_player: Owner,
    pub hands_a: [u16; 5],
    pub hands_b: [u16; 5],
    pub cards_fingerprint: u128,
    // Intentionally omit timestamp by default to preserve byte-for-byte determinism.
    // Add an optional field later if needed.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SolvedDb {
    header: DbHeader,
    entries: BTreeMap<u128, SolvedEntry>,
}

/// Compute a 128-bit fingerprint for the loaded cards database.
/// Iterates cards in stable id order and mixes key attributes deterministically.
pub fn fingerprint_cards(cards: &CardsDb) -> u128 {
    let mut a: u64 = 0xC0FF_EE00_D15E_CAFE;
    let mut b: u64 = 0xDEAD_BEEF_F00D_FACE;

    for c in cards.iter() {
        mix_card(&mut a, &mut b, c);
    }

    ((a as u128) << 64) | (b as u128)
}

#[inline]
fn mix_card(a: &mut u64, b: &mut u64, c: &Card) {
    // Pack sides into u64
    let sides: u64 =
        (c.top as u64) | ((c.right as u64) << 8) | ((c.bottom as u64) << 16) | ((c.left as u64) << 24);

    let kind: u8 = match c.kind {
        crate::cards::CardKind::Monster => 0,
        crate::cards::CardKind::Boss => 1,
        crate::cards::CardKind::GF => 2,
        crate::cards::CardKind::Player => 3,
    };

    let elem: u8 = match c.element {
        None => 255,
        Some(Element::Earth) => 0,
        Some(Element::Fire) => 1,
        Some(Element::Water) => 2,
        Some(Element::Poison) => 3,
        Some(Element::Holy) => 4,
        Some(Element::Thunder) => 5,
        Some(Element::Wind) => 6,
        Some(Element::Ice) => 7,
    };

    // Mix several tagged words to avoid accidental collisions
    mix_into(a, b, (c.id as u64) | (0x11u64 << 56), 0x9E37_79B9_7F4A_7C15);
    mix_into(a, b, (c.level as u64) | (0x12u64 << 56), 0xBF58_476D_1CE4_E5B9);
    mix_into(a, b, (kind as u64) | (0x13u64 << 56), 0x94D0_49BB_1331_11EB);
    mix_into(a, b, (elem as u64) | (0x14u64 << 56), 0xA5A5_A5A5_A5A5_A5A5);
    mix_into(a, b, sides | (0x15u64 << 56), 0xC3A5_C85C_97CB_3127);
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn mix_into(acc_a: &mut u64, acc_b: &mut u64, data: u64, salt: u64) {
    let m1 = splitmix64(data ^ salt);
    let m2 = splitmix64(m1 ^ 0xA5A5_A5A5_A5A5_A5A5);
    *acc_a ^= m1.rotate_left(17);
    *acc_b = acc_b.rotate_left(13) ^ m2;
}

/// Save database to a file as a single bincode blob.
/// The blob contains (header, entries) and preserves BTreeMap iteration order.
pub fn save_db<P: AsRef<Path>>(
    path: P,
    header: &DbHeader,
    entries: &BTreeMap<u128, SolvedEntry>,
) -> Result<(), String> {
    let db = SolvedDb {
        header: header.clone(),
        entries: entries.clone(),
    };
    let bytes = bincode::serialize(&db).map_err(|e| format!("bincode serialize error: {e}"))?;
    fs::write(path.as_ref(), bytes).map_err(|e| format!("write error: {e}"))?;
    Ok(())
}

/// Load database from a file that was written by save_db.
pub fn load_db<P: AsRef<Path>>(
    path: P,
) -> Result<(DbHeader, BTreeMap<u128, SolvedEntry>), String> {
    let bytes = fs::read(path.as_ref()).map_err(|e| format!("read error: {e}"))?;
    let db: SolvedDb = bincode::deserialize(&bytes).map_err(|e| format!("bincode deserialize error: {e}"))?;
    Ok((db.header, db.entries))
}