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

    // Validate and compute max id
    let mut max_id: u16 = 0;
    for c in &raw {
        validate_card(c)?;
        max_id = max_id.max(c.id);
    }

    let mut by_id: Vec<Option<Card>> = vec![None; (max_id as usize) + 1];
    let mut name_to_id: HashMap<String, u16> = HashMap::with_capacity(raw.len());

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

    Ok(CardsDb {
        by_id,
        name_to_id,
        max_id,
        count,
    })
}