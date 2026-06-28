//! # letheo-mql · Mnemonic Query Language
//!
//! Lexer + parser of the biological verbs: PERCEIVE · DISTILL · EVOKE · FADE · IMPRINT.
//! There is **no** SELECT / INSERT / UPDATE / DELETE.
//!
//! ```
//! use letheo_mql::parse;
//! let prog = parse(r#"EVOKE essence OF "user:Xolotl" WITHIN budget 800 tokens"#).unwrap();
//! assert_eq!(prog.len(), 1);
//! ```

pub mod ast;
pub mod lexer;
pub mod parser;
pub mod validate;

pub use ast::{CmpOp, Facts, Field, Predicate, Statement, Value};
pub use parser::{parse, ParseError};
pub use validate::{validate, SemanticError, SemanticErrorKind};
