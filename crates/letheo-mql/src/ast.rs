//! MQL AST. Seven verbs: PERCEIVE · DISTILL · EVOKE · FADE · IMPRINT · RECALL · REINFORCE.
//! Formal grammar: the seven MQL verbs, no SQL CRUD.

use std::collections::BTreeMap;

/// An MQL statement.
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Perceive(Perceive),
    Distill(Distill),
    Evoke(Evoke),
    Fade(Fade),
    Imprint(Imprint),
    /// Layer-1: lossless directed retrieval of episodic facts by resonance.
    Recall(Recall),
    /// Layer-1: reinforcement / spaced-repetition of facts (resets their decay).
    Reinforce(Reinforce),
}

/// Duration as an entropy coefficient (not a timestamp). Normalised to seconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Duration {
    pub seconds: f64,
}

impl Duration {
    pub fn from_value_unit(value: f64, unit: &str) -> Option<Self> {
        let mult = match unit {
            "m" | "min" => 60.0,
            "h" | "hour" => 3600.0,
            "d" | "day" => 86_400.0,
            "w" | "week" => 604_800.0,
            "month" => 2_592_000.0,
            "y" | "year" => 31_536_000.0,
            _ => return None,
        };
        Some(Duration {
            seconds: value * mult,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Perceive {
    pub subject: String,
    pub traits: BTreeMap<String, String>,
    pub salience: Option<f64>,
    pub halflife: Option<Duration>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Distill {
    pub subject: String,
    pub compressing_by_variance: bool,
    pub retaining: Vec<String>,
    /// Optional `WHERE` filter over the perceptions to distil.
    pub filter: Option<Predicate>,
}

// ─────────────────────────────────────────────────────────────────────────────
// WHERE predicates: the language evaluates the filter for real against each perception.
// ─────────────────────────────────────────────────────────────────────────────

/// Comparison operator of a predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

/// The field being compared: a physical property of the memory or an arbitrary trait.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Field {
    /// Current weight of the memory (`weight now`). Numeric, time-dependent.
    Weight,
    /// Initial charge of the stimulus. Numeric.
    Salience,
    /// Age in seconds since last contact (`Δt`). Numeric.
    Age,
    /// Resonance (cosine) of the item with the statement's query. Numeric, in `[-1, 1]`.
    /// Requires the statement to provide a query (`RESONATING WITH { … }`); without a query evaluates to false.
    Resonance,
    /// An arbitrary trait from the trait map (e.g. `domain`, `mood`).
    Trait(String),
}

/// Literal value to compare against.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Num(f64),
    Text(String),
}

/// A boolean `WHERE` predicate: comparisons combined with AND/OR/NOT.
#[derive(Debug, Clone, PartialEq)]
pub enum Predicate {
    Cmp {
        field: Field,
        op: CmpOp,
        value: Value,
    },
    And(Box<Predicate>, Box<Predicate>),
    Or(Box<Predicate>, Box<Predicate>),
    Not(Box<Predicate>),
}

/// Source of facts against which a predicate is evaluated. Implemented by whoever holds the data
/// (the executor, over a `Perception`), keeping `letheo-mql` decoupled from `letheo-core`.
pub trait Facts {
    /// Numeric value of a physical field (`Weight`/`Salience`/`Age`) in the current context.
    fn numeric(&self, field: &Field) -> Option<f64>;
    /// Text value of a trait from the trait map.
    fn text(&self, key: &str) -> Option<String>;
}

impl Predicate {
    /// Evaluates the predicate against a facts source.
    pub fn eval(&self, f: &dyn Facts) -> bool {
        match self {
            Predicate::And(a, b) => a.eval(f) && b.eval(f),
            Predicate::Or(a, b) => a.eval(f) || b.eval(f),
            Predicate::Not(a) => !a.eval(f),
            Predicate::Cmp { field, op, value } => eval_cmp(field, *op, value, f),
        }
    }
}

fn eval_cmp(field: &Field, op: CmpOp, value: &Value, f: &dyn Facts) -> bool {
    match (field, value) {
        // Physical field/resonance vs number → numeric comparison.
        (Field::Weight | Field::Salience | Field::Age | Field::Resonance, Value::Num(rhs)) => {
            match f.numeric(field) {
                Some(lhs) => cmp_num(lhs, op, *rhs),
                None => false,
            }
        }
        // Trait vs number → try to parse the trait as a number.
        (Field::Trait(k), Value::Num(rhs)) => match f.text(k).and_then(|s| s.parse::<f64>().ok()) {
            Some(lhs) => cmp_num(lhs, op, *rhs),
            None => false,
        },
        // Trait vs text → string comparison (lexicographic order for </>).
        (Field::Trait(k), Value::Text(rhs)) => match f.text(k) {
            Some(lhs) => cmp_text(&lhs, op, rhs),
            None => false,
        },
        // Physical field/resonance vs text: no semantic meaning → false.
        (Field::Weight | Field::Salience | Field::Age | Field::Resonance, Value::Text(_)) => false,
    }
}

fn cmp_num(lhs: f64, op: CmpOp, rhs: f64) -> bool {
    match op {
        CmpOp::Lt => lhs < rhs,
        CmpOp::Le => lhs <= rhs,
        CmpOp::Gt => lhs > rhs,
        CmpOp::Ge => lhs >= rhs,
        CmpOp::Eq => lhs == rhs,
        CmpOp::Ne => lhs != rhs,
    }
}

fn cmp_text(lhs: &str, op: CmpOp, rhs: &str) -> bool {
    match op {
        CmpOp::Lt => lhs < rhs,
        CmpOp::Le => lhs <= rhs,
        CmpOp::Gt => lhs > rhs,
        CmpOp::Ge => lhs >= rhs,
        CmpOp::Eq => lhs == rhs,
        CmpOp::Ne => lhs != rhs,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EssenceKind {
    Essence,
    Archetype,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Resolution {
    Arc,
    Point,
    Summary,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Projection {
    Trajectory,
    Snapshot,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Evoke {
    pub kind: EssenceKind,
    pub subject: String,
    pub span: Option<Duration>,
    pub resonating_with: Vec<String>,
    pub resolution: Option<Resolution>,
    pub projecting: Option<Projection>,
    pub token_budget: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Fade {
    pub target: String,
    pub preserving_archetype: bool,
    /// Optional `WHERE` filter: which perceptions are candidates for fading.
    pub filter: Option<Predicate>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Resilience {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Imprint {
    pub archetype: String,
    pub resilience: Option<Resilience>,
}

/// `RECALL` — directed retrieval of episodic facts (layer-1), **lossless** and **read-only**.
/// The query is formed with `RESONATING WITH { … }` (embedded). Optional `WHERE` accepts
/// `resonates`/`weight`/`age`/`salience`. `WITHIN k N` caps the top-k (default 3).
#[derive(Debug, Clone, PartialEq)]
pub struct Recall {
    pub subject: String,
    pub resonating_with: Vec<String>,
    pub k: usize,
    pub filter: Option<Predicate>,
}

/// `REINFORCE` — reinforcement / spaced-repetition of facts that resonate with the query (resets
/// their decay → gains permanence). Mutates layer-1. `WITHIN k N` caps how many to reinforce (default 3).
#[derive(Debug, Clone, PartialEq)]
pub struct Reinforce {
    pub subject: String,
    pub resonating_with: Vec<String>,
    pub k: usize,
}
