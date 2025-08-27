use crate::board::Board;
use crate::rules::Rules;
use crate::types::{Element, Owner};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Move {
    pub card_id: u16,
    pub cell: u8, // 0..=8
}

#[derive(Debug, Clone)]
pub struct GameState {
    pub board: Board,
    pub hands_a: [Option<u16>; 5],
    pub hands_b: [Option<u16>; 5],
    pub next: Owner,
    pub rules: Rules,
}

impl GameState {
    #[inline]
    pub fn new_empty(rules: Rules) -> Self {
        Self {
            board: Board::new(),
            hands_a: [None; 5],
            hands_b: [None; 5],
            next: Owner::A,
            rules,
        }
    }

    #[inline]
    pub fn with_elements(rules: Rules, cell_elements: [Option<Element>; 9]) -> Self {
        Self {
            board: Board::from_elements(cell_elements),
            hands_a: [None; 5],
            hands_b: [None; 5],
            next: Owner::A,
            rules,
        }
    }

    #[inline]
    pub fn with_hands(
        rules: Rules,
        hand_a: [u16; 5],
        hand_b: [u16; 5],
        cell_elements: Option<[Option<Element>; 9]>,
    ) -> Self {
        let mut s = if let Some(elems) = cell_elements {
            Self::with_elements(rules, elems)
        } else {
            Self::new_empty(rules)
        };
        s.hands_a = hand_a.map(Some);
        s.hands_b = hand_b.map(Some);
        s
    }

    #[inline]
    pub fn current_hand(&self) -> &[Option<u16>; 5] {
        match self.next {
            Owner::A => &self.hands_a,
            Owner::B => &self.hands_b,
        }
    }

    #[inline]
    pub fn current_hand_mut(&mut self) -> &mut [Option<u16>; 5] {
        match self.next {
            Owner::A => &mut self.hands_a,
            Owner::B => &mut self.hands_b,
        }
    }

    #[inline]
    pub fn other_hand_mut(&mut self) -> &mut [Option<u16>; 5] {
        match self.next {
            Owner::A => &mut self.hands_b,
            Owner::B => &mut self.hands_a,
        }
    }

    /// Remove a card id from the current player's hand. Returns true if removed.
    #[inline]
    pub fn take_from_current_hand(&mut self, card_id: u16) -> bool {
        let hand = self.current_hand_mut();
        for slot in hand.iter_mut() {
            if slot.map_or(false, |id| id == card_id) {
                *slot = None;
                return true;
            }
        }
        false
    }

    /// Returns ordered legal moves for the current player.
    /// Order: by cell index ascending, then by card_id ascending.
    pub fn legal_moves(&self) -> Vec<Move> {
        // Collect current player's card ids and sort ascending without allocation-heavy ops.
        let mut tmp: [u16; 5] = [u16::MAX; 5];
        let mut n = 0usize;
        for &slot in self.current_hand().iter() {
            if let Some(id) = slot {
                tmp[n] = id;
                n += 1;
            }
        }
        // Simple insertion sort on small array
        for i in 1..n {
            let key = tmp[i];
            let mut j = i;
            while j > 0 && tmp[j - 1] > key {
                tmp[j] = tmp[j - 1];
                j -= 1;
            }
            tmp[j] = key;
        }

        let mut moves = Vec::with_capacity((9 - self.board.filled_count()) as usize * n);
        for cell in 0u8..9u8 {
            if self.board.is_empty(cell) {
                for k in 0..n {
                    moves.push(Move {
                        card_id: tmp[k],
                        cell,
                    });
                }
            }
        }
        moves
    }

    #[inline]
    pub fn is_terminal(&self) -> bool {
        self.board.is_full()
    }
}

/// Re-export minimal surface for callers as free functions to align with the planned API.
#[inline]
pub fn legal_moves(state: &GameState) -> Vec<Move> {
    state.legal_moves()
}

#[inline]
pub fn is_terminal(state: &GameState) -> bool {
    state.is_terminal()
}