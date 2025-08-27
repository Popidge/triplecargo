use crate::types::{rc_to_idx, Dir, Element, Owner};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slot {
    pub owner: Owner,
    pub card_id: u16,
}

#[derive(Debug, Clone)]
pub struct Board {
    // Cells 0..=8 laid out row-major (r*3 + c)
    cells: [Option<Slot>; 9],
    // Optional per-cell element (None means no elemental on that cell)
    cell_elements: [Option<Element>; 9],
}

impl Default for Board {
    fn default() -> Self {
        Self {
            cells: [None; 9],
            cell_elements: [None; 9],
        }
    }
}

impl Board {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn from_elements(cell_elements: [Option<Element>; 9]) -> Self {
        Self {
            cells: [None; 9],
            cell_elements,
        }
    }

    #[inline]
    pub fn get(&self, idx: u8) -> Option<Slot> {
        self.cells[idx as usize]
    }

    #[inline]
    pub fn set(&mut self, idx: u8, slot: Option<Slot>) {
        self.cells[idx as usize] = slot;
    }

    #[inline]
    pub fn is_empty(&self, idx: u8) -> bool {
        self.cells[idx as usize].is_none()
    }

    #[inline]
    pub fn cell_element(&self, idx: u8) -> Option<Element> {
        self.cell_elements[idx as usize]
    }

    #[inline]
    pub fn set_cell_element(&mut self, idx: u8, elem: Option<Element>) {
        self.cell_elements[idx as usize] = elem;
    }

    #[inline]
    pub fn clear(&mut self) {
        self.cells = [None; 9];
    }

    #[inline]
    pub fn filled_count(&self) -> u8 {
        self.cells.iter().filter(|c| c.is_some()).count() as u8
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.filled_count() == 9
    }

    /// Deterministic list of neighbor indices for a cell in [Up, Right, Down, Left] order.
    /// Returns [Option<u8>; 4] where None means off-board (a wall for Same-Wall).
    #[inline]
    pub fn neighbors(&self, idx: u8) -> [Option<u8>; 4] {
        let (r, c) = crate::types::idx_to_rc(idx);
        let up = if r > 0 { Some(rc_to_idx(r - 1, c).unwrap()) } else { None };
        let right = if c < 2 { Some(rc_to_idx(r, c + 1).unwrap()) } else { None };
        let down = if r < 2 { Some(rc_to_idx(r + 1, c).unwrap()) } else { None };
        let left = if c > 0 { Some(rc_to_idx(r, c - 1).unwrap()) } else { None };
        [up, right, down, left]
    }

    /// Helper to get the direction index 0..=3 from Dir in [Up, Right, Down, Left] ordering.
    #[inline]
    pub fn dir_index(dir: Dir) -> usize {
        match dir {
            Dir::Up => 0,
            Dir::Right => 1,
            Dir::Down => 2,
            Dir::Left => 3,
        }
    }
}