#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rules {
    pub elemental: bool,
    pub same: bool,
    pub plus: bool,
    pub same_wall: bool,
}

impl Default for Rules {
    fn default() -> Self {
        // Defaults per spec:
        // elemental=false, same=false, plus=false, same_wall=false
        Self {
            elemental: false,
            same: false,
            plus: false,
            same_wall: false,
        }
    }
}

impl Rules {
    #[inline]
    pub const fn new(elemental: bool, same: bool, plus: bool, same_wall: bool) -> Self {
        Self {
            elemental,
            same,
            plus,
            same_wall,
        }
    }

    #[inline]
    pub const fn basic_only() -> Self {
        Self {
            elemental: false,
            same: false,
            plus: false,
            same_wall: false,
        }
    }

    #[inline]
    pub const fn all_enabled() -> Self {
        Self {
            elemental: true,
            same: true,
            plus: true,
            same_wall: true,
        }
    }
}