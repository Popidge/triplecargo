use crate::types::Element;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum CardKind {
    Monster,
    Boss,
    GF,
    Player,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Card {
    pub id: u16,
    pub name: String,
    pub level: u8,
    #[serde(rename = "type")]
    pub kind: CardKind,
    pub top: u8,
    pub right: u8,
    pub bottom: u8,
    pub left: u8,
    pub element: Option<Element>,
}

impl Card {
    #[inline]
    pub fn sides(&self) -> [u8; 4] {
        [self.top, self.right, self.bottom, self.left]
    }
}

#[derive(Debug, Default)]
pub struct CardsDb {
    by_id: Vec<Option<Card>>,         // index by id (len = max_id + 1)
    name_to_id: HashMap<String, u16>, // case-sensitive names as in data
    max_id: u16,
    count: usize,
}

impl CardsDb {
    #[inline]
    pub fn get(&self, id: u16) -> Option<&Card> {
        self.by_id.get(id as usize).and_then(|c| c.as_ref())
    }

    #[inline]
    pub fn id_by_name(&self, name: &str) -> Option<u16> {
        self.name_to_id.get(name).copied()
    }

    #[inline]
    pub fn max_id(&self) -> u16 {
        self.max_id
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Card> {
        self.by_id.iter().filter_map(|o| o.as_ref())
    }
}

fn validate_card(card: &Card) -> Result<(), String> {
    let within = |v: u8| (1..=10).contains(&v);
    if !(within(card.top) && within(card.right) && within(card.bottom) && within(card.left)) {
        return Err(format!(
            "Card id {} '{}' has invalid side values (must be 1..=10)",
            card.id, card.name
        ));
    }
    Ok(())
}

/// Load cards from a JSON file (runtime), building a dense id index and name lookup.
pub fn load_cards_from_json<P: AsRef<Path>>(path: P) -> Result<CardsDb, String> {
    let data = fs::read_to_string(path.as_ref()).map_err(|e| format!("Failed to read JSON: {e}"))?;
    let raw: Vec<Card> =
        serde_json::from_str(&data).map_err(|e| format!("Failed to parse JSON: {e}"))?;
 
    if raw.is_empty() {
        return Err("No cards in JSON".to_string());
    }
 
    // Cache the JSON entry count before we consume `raw` below.
    let raw_count = raw.len();
 
    // Validate and compute max id
    let mut max_id: u16 = 0;
    for c in &raw {
        validate_card(c)?;
        max_id = max_id.max(c.id);
    }
 
    let mut by_id: Vec<Option<Card>> = vec![None; (max_id as usize) + 1];
    let mut name_to_id: HashMap<String, u16> = HashMap::with_capacity(raw_count);
 
    for c in raw {
        let id = c.id;
        let name = c.name.clone();
 
        // uniqueness checks
        if let Some(existing) = by_id.get(id as usize).and_then(|x| x.as_ref()) {
            return Err(format!(
                "Duplicate card id {} ('{}' and '{}')",
                id, existing.name, name
            ));
        }
        if let Some(prev) = name_to_id.insert(name.clone(), id) {
            return Err(format!(
                "Duplicate card name '{}' for ids {} and {}",
                name, prev, id
            ));
        }
        by_id[id as usize] = Some(c);
    }
 
    let count = by_id.iter().filter(|c| c.is_some()).count();
 
    // Sanity checks specific to the project's canonical data:
    // - Expect ids to be contiguous starting at 1 up to max_id with no gaps.
    // - Expect the number of cards in the dense index to match the JSON count.
    //
    // These checks are strict because `data/cards.json` is expected to contain
    // exactly the full canonical set (ids 1..=N). If this assumption is violated,
    // fail early with a helpful error listing the first few missing ids.
    //
    // Compute min present id
    let mut min_present: Option<u16> = None;
    for (i, slot) in by_id.iter().enumerate() {
        if slot.is_some() {
            min_present = Some(i as u16);
            break;
        }
    }
    let min_present = min_present.unwrap_or(0);
 
    if min_present != 1 {
        return Err(format!(
            "Unexpected minimum card id {} (expected 1). Check data/cards.json ids.",
            min_present
        ));
    }
 
    if count != raw_count {
        return Err(format!(
            "Card count mismatch: JSON had {} entries but indexed {} present",
            raw_count,
            count
        ));
    }
 
    if (max_id as usize) != raw_count {
        // collect a short sample of missing ids in the 1..=max_id range
        let mut missing: Vec<u16> = Vec::new();
        for i in 1..=(max_id as usize) {
            if by_id[i].is_none() {
                missing.push(i as u16);
                if missing.len() >= 8 {
                    break;
                }
            }
        }
        if !missing.is_empty() {
            return Err(format!(
                "Non-contiguous card ids detected: missing ids (first few) {:?}; max_id={} count={}. \
                 Expected contiguous ids 1..={}.",
                missing, max_id, count, raw_count
            ));
        }
    }
 
    Ok(CardsDb {
        by_id,
        name_to_id,
        max_id,
        count,
    })
}