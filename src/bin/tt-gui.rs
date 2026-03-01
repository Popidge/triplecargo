//! Triplecargo Desktop GUI - Triple Triad game against the computer
//!
//! Run with: cargo run --bin tt-gui

use eframe::egui;
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::path::Path;
use std::time::Duration;

use triplecargo::solver::{SearchLimits, Solver};
use triplecargo::{
    apply_move, is_terminal, legal_moves, load_cards_from_json, score, Card, CardsDb, Element,
    GameState, Move, Owner, Rules,
};

// ============================================================================
// Constants & Types
// ============================================================================

const CARDS_PATH: &str = "data/cards.json";

/// Difficulty levels for the computer opponent
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Difficulty {
    Easy,
    Medium,
    Hard,
}

impl Difficulty {
    fn search_depth(&self) -> u8 {
        match self {
            Difficulty::Easy => 1,   // Greedy - only immediate value
            Difficulty::Medium => 4, // Moderate lookahead
            Difficulty::Hard => 9,   // Full perfect play
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Difficulty::Easy => "Easy (Depth 1)",
            Difficulty::Medium => "Medium (Depth 4)",
            Difficulty::Hard => "Hard (Depth 9)",
        }
    }
}

/// Game state management
#[derive(Clone, PartialEq)]
enum GamePhase {
    Menu,
    Playing,
    GameOver { winner: Option<Owner>, margin: i8 },
}

/// Main application state
struct TriplecargoApp {
    // Persistent data
    cards: CardsDb,

    // Game setup
    difficulty: Difficulty,
    rules: Rules,
    game_seed: u64,

    // Game state
    phase: GamePhase,
    state: GameState,
    selected_card: Option<u16>,
    computer_thinking: bool,
    last_computer_move: Option<Move>,
    move_history: Vec<GameState>,

    // Animation
    computer_move_timer: Option<std::time::Instant>,
}

impl TriplecargoApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        // Load cards database
        let cards = load_cards_from_json(Path::new(CARDS_PATH))
            .expect("Failed to load cards from data/cards.json");

        Self {
            cards,
            difficulty: Difficulty::Medium,
            rules: Rules::default(),
            game_seed: 0xDEADBEEF,
            phase: GamePhase::Menu,
            state: GameState::new_empty(Rules::default()),
            selected_card: None,
            computer_thinking: false,
            last_computer_move: None,
            move_history: Vec::new(),
            computer_move_timer: None,
        }
    }

    // ============================================================================
    // Hand Sampling (Stratified)
    // ============================================================================

    /// Build level bands for stratified sampling: [1-2], [3-4], [5-6], [7-8], [9-10]
    fn build_level_bands(&self) -> [Vec<u16>; 5] {
        let mut bands: [Vec<u16>; 5] = Default::default();
        for c in self.cards.iter() {
            let band = match c.level {
                1 | 2 => 0,
                3 | 4 => 1,
                5 | 6 => 2,
                7 | 8 => 3,
                _ => 4, // 9 | 10
            };
            bands[band].push(c.id);
        }
        for b in bands.iter_mut() {
            b.sort_unstable();
        }
        bands
    }

    /// Sample one card from each level band (stratified)
    fn sample_hand_stratified(&self, bands: &mut [Vec<u16>; 5], rng: &mut impl Rng) -> [u16; 5] {
        let mut hand = [0u16; 5];
        for (i, band) in bands.iter_mut().enumerate() {
            if band.is_empty() {
                // Fallback: pick random from all cards
                let all_ids: Vec<u16> = self.cards.iter().map(|c| c.id).collect();
                hand[i] = all_ids[rng.gen_range(0..all_ids.len())];
            } else {
                let idx = rng.gen_range(0..band.len());
                hand[i] = band.swap_remove(idx);
            }
        }
        hand
    }

    /// Sample random hand (uniform)
    fn sample_hand_random(&self, rng: &mut impl Rng) -> [u16; 5] {
        let all_ids: Vec<u16> = self.cards.iter().map(|c| c.id).collect();
        let mut hand = [0u16; 5];
        let mut pool = all_ids;
        for i in 0..5 {
            let idx = rng.gen_range(0..pool.len());
            hand[i] = pool.swap_remove(idx);
        }
        hand
    }

    // ============================================================================
    // Game Setup
    // ============================================================================

    fn start_new_game(&mut self) {
        let mut rng = Pcg64::seed_from_u64(self.game_seed);

        // Sample hands
        let hand_a = self.sample_hand_random(&mut rng);
        let hand_b = self.sample_hand_random(&mut rng);

        // Create initial state
        self.state = GameState::with_hands(
            self.rules, hand_a, hand_b, None, // No elements for MVP
        );

        self.selected_card = None;
        self.computer_thinking = false;
        self.last_computer_move = None;
        self.move_history.clear();
        self.phase = GamePhase::Playing;
    }

    // ============================================================================
    // Game Logic
    // ============================================================================

    fn apply_player_move(&mut self, card_id: u16, cell: u8) -> Result<(), String> {
        let mv = Move { card_id, cell };

        match apply_move(&self.state, &self.cards, mv) {
            Ok(new_state) => {
                self.move_history.push(self.state.clone());
                self.state = new_state;
                self.selected_card = None;

                if is_terminal(&self.state) {
                    let final_score = score(&self.state);
                    let winner = if final_score > 0 {
                        Some(Owner::A)
                    } else if final_score < 0 {
                        Some(Owner::B)
                    } else {
                        None
                    };
                    self.phase = GamePhase::GameOver {
                        winner,
                        margin: final_score,
                    };
                } else {
                    // Trigger computer move
                    self.computer_thinking = true;
                    self.computer_move_timer = Some(std::time::Instant::now());
                }
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn apply_computer_move(&mut self) {
        if self.phase != GamePhase::Playing || self.state.next != Owner::B {
            return;
        }

        let depth = self.difficulty.search_depth();
        let remaining = 9u8 - self.state.board.filled_count();
        let search_depth = remaining.min(depth);

        // Create solver with configured depth
        let limits = SearchLimits {
            max_depth: search_depth,
            time_ms: None,
        };

        let tt = triplecargo::solver::tt_array::FixedTT::with_capacity_pow2(1 << 20);
        let mut solver = Solver::with_tt(Box::new(tt), limits);

        let result = solver.search(&self.state, &self.cards);

        if let Some(best_move) = result.best_move {
            let mv = best_move;
            self.last_computer_move = Some(mv);

            match apply_move(&self.state, &self.cards, mv) {
                Ok(new_state) => {
                    self.move_history.push(self.state.clone());
                    self.state = new_state;

                    if is_terminal(&self.state) {
                        let final_score = score(&self.state);
                        let winner = if final_score > 0 {
                            Some(Owner::A)
                        } else if final_score < 0 {
                            Some(Owner::B)
                        } else {
                            None
                        };
                        self.phase = GamePhase::GameOver {
                            winner,
                            margin: final_score,
                        };
                    }
                }
                Err(e) => {
                    eprintln!("Computer move error: {}", e);
                }
            }
        }

        self.computer_thinking = false;
        self.computer_move_timer = None;
    }

    fn get_score(&self) -> (u8, u8) {
        let diff = score(&self.state);
        // Convert difference to ownership totals: A + B = 10, A - B = diff
        // A = (10 + diff) / 2, B = (10 - diff) / 2
        let a_count = ((10i16 + diff as i16) / 2) as u8;
        let b_count = 10 - a_count;
        (a_count, b_count)
    }

    // ============================================================================
    // UI Helpers
    // ============================================================================

    fn card_by_id(&self, id: u16) -> Option<&Card> {
        self.cards.get(id)
    }

    fn player_hand_cards(&self) -> Vec<(u16, &Card)> {
        self.state
            .hands_a
            .iter()
            .filter_map(|&o| {
                if let Some(id) = o {
                    self.card_by_id(id).map(|card| (id, card))
                } else {
                    None
                }
            })
            .collect()
    }

    fn computer_hand_cards(&self) -> Vec<(u16, &Card)> {
        self.state
            .hands_b
            .iter()
            .filter_map(|&o| {
                if let Some(id) = o {
                    self.card_by_id(id).map(|card| (id, card))
                } else {
                    None
                }
            })
            .collect()
    }

    fn legal_moves_for_card(&self, card_id: u16) -> Vec<u8> {
        self.state
            .legal_moves()
            .iter()
            .filter(|m| m.card_id == card_id)
            .map(|m| m.cell)
            .collect()
    }
}

// ============================================================================
// eframe Implementation
// ============================================================================

impl eframe::App for TriplecargoApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Handle computer move timing
        if self.computer_thinking {
            // Small delay for UX
            if let Some(start) = self.computer_move_timer {
                if start.elapsed() > Duration::from_millis(300) {
                    self.apply_computer_move();
                }
            }
            // Request continuous repaint
            ctx.request_repaint();
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            // Clone phase to avoid borrow conflicts
            let phase = self.phase.clone();
            match phase {
                GamePhase::Menu => {
                    self.draw_menu(ui);
                }
                GamePhase::Playing => {
                    self.draw_game(ui);
                }
                GamePhase::GameOver { winner, margin } => {
                    self.draw_game_over(ui, &winner, margin);
                }
            }
        });
    }
}

// ============================================================================
// UI Drawing Functions
// ============================================================================

impl TriplecargoApp {
    fn draw_menu(&mut self, ui: &mut egui::Ui) {
        ui.heading("♠️ Triplecargo");
        ui.label("Triple Triad against the Computer");
        ui.separator();

        ui.add_space(20.0);
        ui.label("Difficulty:");
        for (i, diff) in [Difficulty::Easy, Difficulty::Medium, Difficulty::Hard]
            .iter()
            .enumerate()
        {
            ui.radio_value(&mut self.difficulty, *diff, diff.label());
        }

        ui.add_space(20.0);
        ui.label("Rules:");
        ui.checkbox(&mut self.rules.elemental, "Elemental");
        ui.checkbox(&mut self.rules.same, "Same");
        ui.checkbox(&mut self.rules.plus, "Plus");
        ui.checkbox(&mut self.rules.same_wall, "Same Wall");

        ui.add_space(20.0);
        ui.horizontal(|ui| {
            ui.label("Seed:");
            ui.add(egui::DragValue::new(&mut self.game_seed).clamp_range(0..=u64::MAX));
        });

        ui.add_space(30.0);
        ui.label("Click New Game to start!");
        ui.add_space(10.0);

        if ui.button("🎮 New Game").clicked() {
            self.start_new_game();
        }

        ui.add_space(20.0);
        ui.separator();
        ui.label("Card pool loaded from data/cards.json");
    }

    fn draw_game(&mut self, ui: &mut egui::Ui) {
        egui::TopBottomPanel::top("header").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Triplecargo");
                ui.add_space(20.0);
                let (a_score, b_score) = self.get_score();
                ui.label(format!("Score - You: {}  Computer: {}", a_score, b_score));
                ui.add_space(20.0);
                let turn_label = match self.state.next {
                    Owner::A => "Your Turn",
                    Owner::B => "Computer's Turn",
                };
                ui.label(turn_label);
            });
        });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.columns(3, |columns| {
                columns[0].vertical(|ui| {
                    ui.label("Your Hand - Click a card to select, then click a board cell");
                    self.draw_hand_in_column(ui, true);
                });

                columns[1].vertical(|ui| {
                    ui.label("Board");
                    self.draw_board_in_column(ui);
                });

                columns[2].vertical(|ui| {
                    ui.label("Computer's Hand");
                    self.draw_computer_hand_in_column(ui);
                });
            });

            ui.add_space(10.0);
            self.draw_active_rules(ui);

            ui.add_space(10.0);

            if ui.button("Main Menu").clicked() {
                self.phase = GamePhase::Menu;
            }
        });
    }

    fn draw_board(&mut self, ui: &mut egui::Ui) {
        // Calculate board size based on available width
        let board_size = ui.available_width().min(400.0);
        let cell_size = board_size / 3.0;
        let card_size = cell_size - 4.0;

        // Board visual
        egui::Grid::new("board")
            .spacing(egui::vec2(2.0, 2.0))
            .show(ui, |ui| {
                for row in 0..3 {
                    for col in 0..3 {
                        let cell_idx = (row * 3 + col) as u8;
                        let cell_rect = ui.available_rect_before_wrap();

                        // Draw cell background
                        let response =
                            ui.interact(cell_rect, egui::Id::new(cell_idx), egui::Sense::click());

                        let (bg_color, border_color) = if self.state.board.is_empty(cell_idx) {
                            // Empty cell
                            (egui::Color32::from_rgb(60, 60, 60), egui::Color32::GRAY)
                        } else {
                            // Occupied cell
                            let slot = self.state.board.get(cell_idx).unwrap();
                            match slot.owner {
                                Owner::A => (
                                    egui::Color32::from_rgb(100, 50, 50),
                                    egui::Color32::LIGHT_RED,
                                ),
                                Owner::B => (
                                    egui::Color32::from_rgb(50, 50, 100),
                                    egui::Color32::LIGHT_BLUE,
                                ),
                            }
                        };

                        let painter = ui.painter();
                        painter.rect_filled(cell_rect, 2.0, bg_color);
                        painter.rect_stroke(cell_rect, 2.0, egui::Stroke::new(1.0, border_color));

                        // Draw card if occupied
                        if let Some(slot) = self.state.board.get(cell_idx) {
                            self.draw_board_card(
                                ui,
                                cell_rect.min,
                                card_size,
                                slot.card_id,
                                slot.owner,
                            );
                        } else if self.state.next == Owner::A && self.selected_card.is_some() {
                            // Highlight valid cells for selected card
                            let legal_cells =
                                self.legal_moves_for_card(self.selected_card.unwrap());
                            if legal_cells.contains(&cell_idx) {
                                painter.rect_filled(
                                    cell_rect.shrink(4.0),
                                    2.0,
                                    egui::Color32::from_rgba_unmultiplied(100, 200, 100, 100),
                                );
                            }

                            // Handle click on empty cell
                            if response.clicked() {
                                if let Some(card_id) = self.selected_card {
                                    if legal_cells.contains(&cell_idx) {
                                        let _ = self.apply_player_move(card_id, cell_idx);
                                    }
                                }
                            }
                        }

                        // Cell number label (top-left corner)
                        ui.painter().text(
                            cell_rect.left_top() + egui::vec2(2.0, 2.0),
                            egui::Align2::LEFT_TOP,
                            format!("{}", cell_idx),
                            egui::FontId::monospace(10.0),
                            egui::Color32::GRAY,
                        );
                    }
                    ui.end_row();
                }
            });

        if self.computer_thinking {
            ui.label("🤔 Computer is thinking...");
        }
    }

    fn draw_board_card(
        &self,
        ui: &egui::Ui,
        pos: egui::Pos2,
        size: f32,
        card_id: u16,
        owner: Owner,
    ) {
        if let Some(card) = self.card_by_id(card_id) {
            let rect =
                egui::Rect::from_min_size(pos + egui::vec2(2.0, 2.0), egui::vec2(size, size));

            // Card background
            let bg_color = match owner {
                Owner::A => egui::Color32::from_rgb(180, 100, 100),
                Owner::B => egui::Color32::from_rgb(100, 100, 180),
            };
            ui.painter().rect_filled(rect, 4.0, bg_color);

            // Card name (truncated)
            ui.painter().text(
                rect.center() + egui::vec2(0.0, -size * 0.25),
                egui::Align2::CENTER_CENTER,
                &card.name[..card.name.len().min(8)],
                egui::FontId::monospace(size * 0.12),
                egui::Color32::WHITE,
            );

            // Stats (small text)
            let stats = format!("{}/{}\n{}/{}", card.top, card.right, card.bottom, card.left);
            ui.painter().text(
                rect.center() + egui::vec2(0.0, size * 0.15),
                egui::Align2::CENTER_CENTER,
                stats,
                egui::FontId::monospace(size * 0.1),
                egui::Color32::YELLOW,
            );

            // Element indicator
            if let Some(elem) = card.element {
                let elem_char = match elem {
                    Element::Fire => "🔥",
                    Element::Ice => "❄️",
                    Element::Thunder => "⚡",
                    Element::Water => "💧",
                    Element::Earth => "🌍",
                    Element::Poison => "☠️",
                    Element::Holy => "✨",
                    Element::Wind => "💨",
                };
                ui.painter().text(
                    rect.right_top() + egui::vec2(-12.0, 2.0),
                    egui::Align2::RIGHT_TOP,
                    elem_char,
                    egui::FontId::monospace(12.0),
                    egui::Color32::WHITE,
                );
            }
        }
    }

    fn draw_hand(&mut self, ui: &mut egui::Ui, interactive: bool) {
        // Copy selected card ID first, before any borrows
        let selected_id = self.selected_card;

        // Get cards - this borrows self
        let player_cards = self.player_hand_cards();

        // Clone card IDs to avoid borrow conflicts
        let card_ids: Vec<u16> = player_cards.iter().map(|(id, _)| *id).collect();

        // Detect clicks - we'll store which card to select
        let mut to_select: Option<u16> = None;

        // Use Grid for proper layout
        egui::Grid::new("hand_grid")
            .spacing(egui::vec2(10.0, 5.0))
            .show(ui, |ui| {
                for (i, &card_id) in card_ids.iter().enumerate() {
                    let card = player_cards[i].1;
                    let is_selected = selected_id == Some(card_id);

                    // Reserve space for card and draw it
                    ui.scope(|ui| {
                        ui.set_min_width(130.0);
                        ui.set_min_height(90.0);
                        self.draw_hand_card(ui, card, is_selected);
                    });

                    // Draw select button next to card
                    if interactive {
                        if ui.button("Select").clicked() {
                            to_select = Some(card_id);
                        }
                    }

                    ui.end_row();
                }
            });

        if player_cards.is_empty() {
            ui.label("(no cards)");
        }

        // Apply selection after all borrows are gone
        drop(player_cards);
        if let Some(id) = to_select {
            self.selected_card = Some(id);
        }
    }

    fn draw_computer_hand(&mut self, ui: &mut egui::Ui) {
        let computer_cards = self.computer_hand_cards();

        egui::Grid::new("computer_hand_grid")
            .spacing(egui::vec2(10.0, 5.0))
            .show(ui, |ui| {
                for (card_id, card) in &computer_cards {
                    ui.scope(|ui| {
                        ui.set_min_width(130.0);
                        ui.set_min_height(90.0);
                        // Draw visible card (open rule)
                        self.draw_hand_card(ui, card, false);
                    });
                    ui.end_row();
                }
            });

        if computer_cards.is_empty() {
            ui.label("(no cards)");
        }
    }

    fn draw_hand_in_column(&mut self, ui: &mut egui::Ui, interactive: bool) {
        let selected_id = self.selected_card;
        let player_cards = self.player_hand_cards();
        let card_ids: Vec<u16> = player_cards.iter().map(|(id, _)| *id).collect();
        let mut to_select: Option<u16> = None;

        egui::Grid::new("hand_grid")
            .spacing(egui::vec2(10.0, 5.0))
            .show(ui, |ui| {
                for (i, &card_id) in card_ids.iter().enumerate() {
                    let card = player_cards[i].1;
                    let is_selected = selected_id == Some(card_id);

                    ui.scope(|ui| {
                        ui.set_min_width(130.0);
                        ui.set_min_height(90.0);

                        let rect = ui.available_rect_before_wrap();
                        let response = ui.interact(
                            rect,
                            egui::Id::new(("hand", card_id)),
                            egui::Sense::click(),
                        );

                        self.draw_hand_card(ui, card, is_selected);

                        if interactive && response.clicked() {
                            to_select = Some(card_id);
                        }
                    });

                    ui.end_row();
                }
            });

        if player_cards.is_empty() {
            ui.label("(no cards)");
        }

        drop(player_cards);
        if let Some(id) = to_select {
            self.selected_card = Some(id);
        }
    }

    fn draw_computer_hand_in_column(&mut self, ui: &mut egui::Ui) {
        let computer_cards = self.computer_hand_cards();

        egui::Grid::new("computer_hand_grid")
            .spacing(egui::vec2(10.0, 5.0))
            .show(ui, |ui| {
                for (_card_id, card) in &computer_cards {
                    ui.scope(|ui| {
                        ui.set_min_width(130.0);
                        ui.set_min_height(90.0);
                        self.draw_hand_card(ui, card, false);
                    });
                    ui.end_row();
                }
            });

        if computer_cards.is_empty() {
            ui.label("(no cards)");
        }
    }

    fn draw_board_in_column(&mut self, ui: &mut egui::Ui) {
        let board_size = 300.0_f32.min(ui.available_width());
        let cell_size = board_size / 3.0;
        let card_size = cell_size - 8.0;
        let expected_size = egui::vec2(cell_size, cell_size);

        egui::Grid::new("board")
            .spacing(egui::vec2(4.0, 4.0))
            .show(ui, |ui| {
                for row in 0..3 {
                    for col in 0..3 {
                        let cell_idx = (row * 3 + col) as u8;

                        let cell_pos = ui.available_rect_before_wrap().min;
                        let cell_rect = egui::Rect::from_min_size(cell_pos, expected_size);
                        ui.set_min_size(expected_size);
                        ui.set_max_size(expected_size);

                        let response = ui.interact(
                            cell_rect,
                            egui::Id::new(("board", cell_idx)),
                            egui::Sense::click(),
                        );

                        let (bg_color, border_color) = if self.state.board.is_empty(cell_idx) {
                            (egui::Color32::from_rgb(80, 80, 80), egui::Color32::GRAY)
                        } else {
                            let slot = self.state.board.get(cell_idx).unwrap();
                            match slot.owner {
                                Owner::A => (
                                    egui::Color32::from_rgb(120, 60, 60),
                                    egui::Color32::LIGHT_RED,
                                ),
                                Owner::B => (
                                    egui::Color32::from_rgb(60, 60, 120),
                                    egui::Color32::LIGHT_BLUE,
                                ),
                            }
                        };

                        let painter = ui.painter();
                        painter.rect_filled(cell_rect, 4.0, bg_color);
                        painter.rect_stroke(cell_rect, 2.0, egui::Stroke::new(2.0, border_color));

                        if let Some(slot) = self.state.board.get(cell_idx) {
                            self.draw_board_card(
                                ui,
                                cell_pos + egui::vec2(4.0, 4.0),
                                card_size,
                                slot.card_id,
                                slot.owner,
                            );
                        } else if self.state.next == Owner::A && self.selected_card.is_some() {
                            let legal_cells =
                                self.legal_moves_for_card(self.selected_card.unwrap());
                            if legal_cells.contains(&cell_idx) {
                                painter.rect_filled(
                                    cell_rect.shrink(4.0),
                                    4.0,
                                    egui::Color32::from_rgba_unmultiplied(100, 200, 100, 150),
                                );
                            }

                            if response.clicked() {
                                if let Some(card_id) = self.selected_card {
                                    if legal_cells.contains(&cell_idx) {
                                        let _ = self.apply_player_move(card_id, cell_idx);
                                    }
                                }
                            }
                        }

                        painter.text(
                            cell_pos + egui::vec2(4.0, 4.0),
                            egui::Align2::LEFT_TOP,
                            format!("{}", cell_idx),
                            egui::FontId::monospace(12.0),
                            egui::Color32::from_gray(180),
                        );
                    }
                    ui.end_row();
                }
            });

        if self.computer_thinking {
            ui.label("Computer is thinking...");
        }
    }

    fn draw_hand_card(&self, ui: &mut egui::Ui, card: &Card, selected: bool) {
        let card_width: f32 = 120.0;
        let card_height: f32 = 80.0;

        let rect = ui.available_rect_before_wrap();
        let w = card_width.min(rect.width());
        let h = card_height.min(rect.height());
        let card_rect = egui::Rect::from_min_size(rect.min, egui::vec2(w, h));

        let bg_color = if selected {
            egui::Color32::from_rgb(60, 140, 60)
        } else {
            egui::Color32::from_rgb(50, 50, 60)
        };
        ui.painter().rect_filled(card_rect, 4.0, bg_color);

        if selected {
            ui.painter()
                .rect_stroke(card_rect, 4.0, egui::Stroke::new(3.0, egui::Color32::GREEN));
        }

        ui.painter().text(
            card_rect.left_top() + egui::vec2(5.0, 5.0),
            egui::Align2::LEFT_TOP,
            format!("Lv.{}", card.level),
            egui::FontId::monospace(10.0),
            egui::Color32::WHITE,
        );

        // Card info
        ui.painter().text(
            card_rect.left_top() + egui::vec2(5.0, 5.0),
            egui::Align2::LEFT_TOP,
            format!("Lv.{}", card.level),
            egui::FontId::monospace(10.0),
            egui::Color32::WHITE,
        );

        ui.painter().text(
            card_rect.center() + egui::vec2(0.0, -15.0),
            egui::Align2::CENTER_CENTER,
            &card.name[..card.name.len().min(12)],
            egui::FontId::monospace(11.0),
            egui::Color32::WHITE,
        );

        // Stats
        let stats = format!(
            "↑{} →{} ↓{} ←{}",
            card.top, card.right, card.bottom, card.left
        );
        ui.painter().text(
            card_rect.center() + egui::vec2(0.0, 10.0),
            egui::Align2::CENTER_CENTER,
            stats,
            egui::FontId::monospace(10.0),
            egui::Color32::from_rgb(200, 200, 100),
        );

        // Element
        if let Some(elem) = card.element {
            let elem_str = match elem {
                Element::Fire => "Fire",
                Element::Ice => "Ice",
                Element::Thunder => "Thunder",
                Element::Water => "Water",
                Element::Earth => "Earth",
                Element::Poison => "Poison",
                Element::Holy => "Holy",
                Element::Wind => "Wind",
            };
            ui.painter().text(
                card_rect.left_bottom() + egui::vec2(5.0, -5.0),
                egui::Align2::LEFT_BOTTOM,
                elem_str,
                egui::FontId::monospace(9.0),
                egui::Color32::from_rgb(150, 150, 255),
            );
        }
    }

    fn draw_hidden_card(&self, ui: &mut egui::Ui, _card_id: u16) {
        let card_width: f32 = 120.0;
        let card_height: f32 = 80.0;

        // Use the current allocation
        let rect = ui.available_rect_before_wrap();
        let w = card_width.min(rect.width());
        let h = card_height.min(rect.height());
        let card_rect = egui::Rect::from_min_size(rect.min, egui::vec2(w, h));

        // Card back design
        ui.painter()
            .rect_filled(card_rect, 4.0, egui::Color32::from_rgb(40, 40, 80));
        ui.painter().rect_stroke(
            card_rect,
            4.0,
            egui::Stroke::new(2.0, egui::Color32::from_rgb(80, 80, 120)),
        );

        // Pattern
        ui.painter().text(
            card_rect.center(),
            egui::Align2::CENTER_CENTER,
            "?",
            egui::FontId::monospace(30.0),
            egui::Color32::from_rgb(100, 100, 150),
        );
    }

    fn draw_active_rules(&self, ui: &mut egui::Ui) {
        ui.label("Active Rules:");
        ui.label(if self.rules.elemental {
            "✓ Elemental"
        } else {
            "○ Elemental"
        });
        ui.label(if self.rules.same {
            "✓ Same"
        } else {
            "○ Same"
        });
        ui.label(if self.rules.plus {
            "✓ Plus"
        } else {
            "○ Plus"
        });
        ui.label(if self.rules.same_wall {
            "✓ Same Wall"
        } else {
            "○ Same Wall"
        });
    }

    fn draw_game_over(&mut self, ui: &mut egui::Ui, winner: &Option<Owner>, margin: i8) {
        ui.heading("Game Over!");
        ui.separator();
        ui.add_space(20.0);

        let result_text = match winner {
            Some(Owner::A) => "🎉 You Win!",
            Some(Owner::B) => "💀 Computer Wins!",
            None => "🤝 It's a Draw!",
        };
        ui.label(result_text);

        let (a_score, b_score) = self.get_score();
        ui.label(format!("Final Score: A:{} - B:{}", a_score, b_score));
        if margin != 0 {
            ui.label(format!("Margin: {} cards", margin.abs()));
        }

        ui.add_space(30.0);

        ui.horizontal(|ui| {
            if ui.button("🔄 Play Again").clicked() {
                self.start_new_game();
            }
            if ui.button("🏠 Main Menu").clicked() {
                self.phase = GamePhase::Menu;
            }
        });
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([900.0, 600.0])
            .with_title("Triplecargo - Triple Triad"),
        ..Default::default()
    };

    eframe::run_native(
        "Triplecargo",
        options,
        Box::new(|cc| Ok(Box::new(TriplecargoApp::new(cc)))),
    )
}
