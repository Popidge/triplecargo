use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use triplecargo::{
    load_cards_from_json, zobrist_key, Element, GameState, Owner, Rules,
    persist::{load_db, ElementsMode},
};

#[derive(Debug, Clone, ValueEnum)]
enum ElementsOpt {
    None,
    Random,
}

#[derive(Debug, Parser)]
#[command(name = "query", about = "Triplecargo solved DB query tool")]
struct Args {
    /// Solved DB file path (produced by precompute)
    #[arg(long, default_value = "solved.bin")]
    db: PathBuf,

    /// Rules toggles as comma-separated list: elemental,same,plus,same_wall (or 'none')
    #[arg(long, default_value = "none")]
    rules: String,

    /// Hands specification: pass exactly two values: A=1,2,3,4,5 B=6,7,8,9,10
    /// Example: --hands "A=1,2,3,4,5" "B=6,7,8,9,10"
    #[arg(long, num_args = 2)]
    hands: Vec<String>,

    /// Elements mode for reconstructing the state key: none | random
    #[arg(long, value_enum, default_value_t = ElementsOpt::None)]
    elements: ElementsOpt,

    /// Seed used when --elements random is selected (deterministic)
    #[arg(long, default_value_t = 0x00C0FFEEu64)]
    seed: u64,

    /// Cards JSON path (defaults to data/cards.json) - used only for validation if desired
    #[arg(long, default_value = "data/cards.json")]
    cards: String,
}

fn parse_rules(s: &str) -> Rules {
    let mut r = Rules::default();
    let s = s.trim();
    if s.eq_ignore_ascii_case("none") || s.is_empty() {
        return r;
    }
    for tok in s.split(',') {
        match tok.trim().to_ascii_lowercase().as_str() {
            "elemental" => r.elemental = true,
            "same" => r.same = true,
            "plus" => r.plus = true,
            "same_wall" | "samewall" => r.same_wall = true,
            "" => {}
            _ => {}
        }
    }
    r
}

fn parse_hands(tokens: &[String]) -> Result<([u16; 5], [u16; 5]), String> {
    if tokens.len() != 2 {
        return Err("Expected exactly two --hands values: \"A=...\" \"B=...\"".into());
    }
    let mut a: Option<[u16; 5]> = None;
    let mut b: Option<[u16; 5]> = None;
    for t in tokens {
        let parts: Vec<&str> = t.split('=').collect();
        if parts.len() != 2 {
            return Err(format!("Invalid hands token '{}', expected A=... or B=...", t));
        }
        let side = parts[0].trim();
        let vals = parts[1].trim();
        let ids: Vec<u16> = if vals.is_empty() {
            Vec::new()
        } else {
            vals.split(',')
                .map(|x| x.trim().parse::<u16>())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("Invalid card id in '{}': {e}", t))?
        };
        if ids.len() != 5 {
            return Err(format!("Expected 5 card ids for {}, got {}", side, ids.len()));
        }
        let arr = [ids[0], ids[1], ids[2], ids[3], ids[4]];
        match side {
            "A" | "a" => a = Some(arr),
            "B" | "b" => b = Some(arr),
            _ => return Err(format!("Unknown hand side '{}', expected A or B", side)),
        }
    }
    match (a, b) {
        (Some(ha), Some(hb)) => Ok((ha, hb)),
        _ => Err("Both A=... and B=... must be provided".into()),
    }
}

// Minimal SplitMix64 for deterministic elements RNG (no rand dependency)
#[inline]
fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn gen_elements(seed: u64) -> [Option<Element>; 9] {
    let mut s = seed;
    let to_elem = |k: u8| -> Option<Element> {
        match k {
            0 => None,
            1 => Some(Element::Earth),
            2 => Some(Element::Fire),
            3 => Some(Element::Water),
            4 => Some(Element::Poison),
            5 => Some(Element::Holy),
            6 => Some(Element::Thunder),
            7 => Some(Element::Wind),
            _ => Some(Element::Ice), // 8
        }
    };
    let mut arr: [Option<Element>; 9] = [None; 9];
    for i in 0..9 {
        let r = splitmix64(&mut s);
        let k = (r % 9) as u8; // uniform over 0..=8
        arr[i] = to_elem(k);
    }
    arr
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Load DB
    let (header, map) = load_db(&args.db)
        .map_err(|e| format!("DB load error: {e}"))?;

    // Optionally, load cards just for validation or future use
    if let Err(e) = load_cards_from_json(&args.cards) {
        eprintln!("[query] Warning: could not load cards JSON: {e}");
    }

    // Build a state using provided flags (must match the one used during precompute)
    let rules = parse_rules(&args.rules);
    let (hand_a, hand_b) = parse_hands(&args.hands)
        .map_err(|e| format!("Hands parse error: {e}"))?;
    let elements_mode = match args.elements {
        ElementsOpt::None => ElementsMode::None,
        ElementsOpt::Random => ElementsMode::Random,
    };
    let cell_elements = match elements_mode {
        ElementsMode::None => None,
        ElementsMode::Random => Some(gen_elements(args.seed)),
    };
    let gs = GameState::with_hands(rules, hand_a, hand_b, cell_elements);

    let key = zobrist_key(&gs);
    println!("[query] Key {:032x}", key);

    match map.get(&key) {
        Some(entry) => {
            println!(
                "[query] Found: value={}, depth={}, best_move={}",
                entry.value,
                entry.depth,
                match entry.best_move {
                    Some(mv) => format!("{{ card_id: {}, cell: {} }}", mv.card_id, mv.cell),
                    None => "None".to_string(),
                }
            );
        }
        None => {
            println!("[query] Not found in DB");
        }
    }

    // Optional: compare with stored header metadata
    if header.elements_mode != elements_mode {
        eprintln!(
            "[query] Warning: DB elements_mode {:?} differs from query {:?}",
            header.elements_mode, elements_mode
        );
    }
    if header.seed != args.seed && header.elements_mode == ElementsMode::Random {
        eprintln!(
            "[query] Warning: DB seed {} differs from query seed {}",
            header.seed, args.seed
        );
    }

    Ok(())
}