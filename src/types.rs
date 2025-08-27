use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum Element {
    Earth,
    Fire,
    Water,
    Poison,
    Holy,
    Thunder,
    Wind,
    Ice,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Owner {
    A,
    B,
}

impl Owner {
    #[inline]
    pub fn other(self) -> Self {
        match self {
            Owner::A => Owner::B,
            Owner::B => Owner::A,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dir {
    Up,
    Right,
    Down,
    Left,
}

impl Dir {
    #[inline]
    pub fn all() -> [Dir; 4] {
        [Dir::Up, Dir::Right, Dir::Down, Dir::Left]
    }

    #[inline]
    pub fn opposite(self) -> Dir {
        match self {
            Dir::Up => Dir::Down,
            Dir::Right => Dir::Left,
            Dir::Down => Dir::Up,
            Dir::Left => Dir::Right,
        }
    }
}

/// Board indexing helpers (3x3 board)
#[inline]
pub fn idx_to_rc(idx: u8) -> (u8, u8) {
    debug_assert!(idx < 9);
    (idx / 3, idx % 3)
}

#[inline]
pub fn rc_to_idx(r: u8, c: u8) -> Option<u8> {
    if r < 3 && c < 3 {
        Some(r * 3 + c)
    } else {
        None
    }
}