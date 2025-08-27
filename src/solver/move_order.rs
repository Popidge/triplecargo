use crate::state::Move;
use std::cmp::Ordering;

/// Return category rank for a board cell:
/// 0 = corner (best), 1 = edge, 2 = center (worst)
#[inline]
fn cell_category(cell: u8) -> u8 {
    match cell {
        0 | 2 | 6 | 8 => 0, // corners
        4 => 2,             // center
        _ => 1,             // edges: 1,3,5,7
    }
}

/// Deterministic move ordering:
/// - TT best move first (if provided and present)
/// - Corners > Edges > Center
/// - Then ascending cell index
/// - Then ascending card_id
#[inline]
pub fn order_moves(moves: &mut Vec<Move>, tt_best: Option<Move>) {
    // Primary sort by (category, cell, card_id)
    moves.sort_by(|a, b| {
        let ca = cell_category(a.cell);
        let cb = cell_category(b.cell);
        match ca.cmp(&cb) {
            Ordering::Equal => match a.cell.cmp(&b.cell) {
                Ordering::Equal => a.card_id.cmp(&b.card_id),
                other => other,
            },
            other => other,
        }
    });

    // If TT suggests a best move, place it first
    if let Some(best) = tt_best {
        if let Some(pos) = moves.iter().position(|m| *m == best) {
            if pos != 0 {
                moves.swap(0, pos);
            }
        }
    }
}