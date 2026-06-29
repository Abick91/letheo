//! Recursive-descent MQL parser → AST.
//!
//! Keywords are compared case-insensitively. Optional clauses are recognised by their guide keyword,
//! so the user's flexible ordering does not break parsing.

use crate::ast::*;
use crate::lexer::{lex, Token};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
}

impl ParseError {
    fn new(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
        }
    }
}

type PResult<T> = Result<T, ParseError>;

struct Parser {
    toks: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> &Token {
        &self.toks[self.pos]
    }

    fn next(&mut self) -> Token {
        let t = self.toks[self.pos].clone();
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn at_eof(&self) -> bool {
        matches!(self.peek(), Token::Eof)
    }

    /// Is the next token the keyword `kw` (case-insensitive)? Non-consuming.
    fn is_kw(&self, kw: &str) -> bool {
        matches!(self.peek(), Token::Word(w) if w.eq_ignore_ascii_case(kw))
    }

    /// Consume the expected keyword or error.
    fn expect_kw(&mut self, kw: &str) -> PResult<()> {
        if self.is_kw(kw) {
            self.next();
            Ok(())
        } else {
            Err(ParseError::new(format!(
                "expected '{kw}', found {:?}",
                self.peek()
            )))
        }
    }

    /// Consume the keyword if present; returns whether it was there.
    fn eat_kw(&mut self, kw: &str) -> bool {
        if self.is_kw(kw) {
            self.next();
            true
        } else {
            false
        }
    }

    fn expect_string(&mut self) -> PResult<String> {
        match self.next() {
            Token::Str(s) => Ok(s),
            t => Err(ParseError::new(format!("expected a string, found {t:?}"))),
        }
    }

    fn expect_number(&mut self) -> PResult<f64> {
        match self.next() {
            Token::Number(n) => Ok(n),
            t => Err(ParseError::new(format!("expected a number, found {t:?}"))),
        }
    }

    fn expect_word(&mut self) -> PResult<String> {
        match self.next() {
            Token::Word(w) => Ok(w),
            t => Err(ParseError::new(format!(
                "expected an identifier, found {t:?}"
            ))),
        }
    }

    /// Duration: `<number> <unit-word>`.
    fn parse_duration(&mut self) -> PResult<Duration> {
        let val = self.expect_number()?;
        let unit = self.expect_word()?;
        Duration::from_value_unit(val, &unit)
            .ok_or_else(|| ParseError::new(format!("invalid duration unit: {unit}")))
    }

    /// trait_map: `{ k: v, k: v }`.
    fn parse_trait_map(&mut self) -> PResult<BTreeMap<String, String>> {
        let mut map = BTreeMap::new();
        if !matches!(self.next(), Token::LBrace) {
            return Err(ParseError::new("expected '{'"));
        }
        while !matches!(self.peek(), Token::RBrace) {
            let key = self.expect_word()?;
            if !matches!(self.next(), Token::Colon) {
                return Err(ParseError::new("expected ':' in trait map"));
            }
            let value = match self.next() {
                Token::Str(s) => s,
                Token::Word(w) => w,
                Token::Number(n) => n.to_string(),
                t => {
                    return Err(ParseError::new(format!(
                        "invalid value in trait map: {t:?}"
                    )))
                }
            };
            map.insert(key, value);
            if !matches!(self.peek(), Token::RBrace) && !matches!(self.next(), Token::Comma) {
                return Err(ParseError::new("expected ',' or '}' in trait map"));
            }
        }
        self.next(); // consume '}'
        Ok(map)
    }

    /// trait_set: `{ a, b, c }`.
    fn parse_trait_set(&mut self) -> PResult<Vec<String>> {
        let mut set = Vec::new();
        if !matches!(self.next(), Token::LBrace) {
            return Err(ParseError::new("expected '{'"));
        }
        while !matches!(self.peek(), Token::RBrace) {
            set.push(self.expect_word()?);
            if !matches!(self.peek(), Token::RBrace) && !matches!(self.next(), Token::Comma) {
                return Err(ParseError::new("expected ',' or '}' in trait set"));
            }
        }
        self.next(); // consume '}'
        Ok(set)
    }

    fn parse_statement(&mut self) -> PResult<Statement> {
        if self.is_kw("PERCEIVE") {
            self.parse_perceive().map(Statement::Perceive)
        } else if self.is_kw("DISTILL") {
            self.parse_distill().map(Statement::Distill)
        } else if self.is_kw("EVOKE") {
            self.parse_evoke().map(Statement::Evoke)
        } else if self.is_kw("FADE") {
            self.parse_fade().map(Statement::Fade)
        } else if self.is_kw("IMPRINT") {
            self.parse_imprint().map(Statement::Imprint)
        } else if self.is_kw("RECALL") {
            self.parse_recall().map(Statement::Recall)
        } else if self.is_kw("REINFORCE") {
            self.parse_reinforce().map(Statement::Reinforce)
        } else {
            Err(ParseError::new(format!(
                "unknown MQL verb: {:?}",
                self.peek()
            )))
        }
    }

    fn parse_perceive(&mut self) -> PResult<Perceive> {
        self.expect_kw("PERCEIVE")?;
        self.expect_kw("interaction")?;
        self.expect_kw("FROM")?;
        self.expect_kw("subject")?;
        let subject = self.expect_string()?;
        self.expect_kw("AS")?;
        let traits = self.parse_trait_map()?;

        let mut salience = None;
        let mut halflife = None;
        if self.eat_kw("WITH") {
            self.expect_kw("salience")?;
            salience = Some(self.expect_number()?);
        }
        if self.eat_kw("DECAYS") {
            self.expect_kw("halflife")?;
            halflife = Some(self.parse_duration()?);
        }
        Ok(Perceive {
            subject,
            traits,
            salience,
            halflife,
        })
    }

    fn parse_distill(&mut self) -> PResult<Distill> {
        self.expect_kw("DISTILL")?;
        self.expect_kw("subject")?;
        let subject = self.expect_string()?;
        let mut filter = None;
        if self.eat_kw("FROM") {
            self.expect_kw("perceptions")?;
            // Optional WHERE clause: parsed as a real predicate.
            if self.eat_kw("WHERE") {
                filter = Some(self.parse_predicate()?);
            }
        }
        self.expect_kw("INTO")?;
        self.expect_kw("intention_vector")?;
        let mut compressing_by_variance = false;
        if self.eat_kw("COMPRESSING") {
            self.expect_kw("BY")?;
            self.expect_kw("semantic_variance")?;
            compressing_by_variance = true;
        }
        let mut retaining = Vec::new();
        if self.eat_kw("RETAINING") {
            retaining = self.parse_trait_set()?;
        }
        Ok(Distill {
            subject,
            compressing_by_variance,
            retaining,
            filter,
        })
    }

    // ── WHERE predicates (precedence: OR < AND < NOT < comparison/parentheses) ──

    /// `predicate := and_expr ( OR and_expr )*`
    fn parse_predicate(&mut self) -> PResult<Predicate> {
        let mut left = self.parse_and()?;
        while self.eat_kw("OR") {
            let right = self.parse_and()?;
            left = Predicate::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    /// `and_expr := not_expr ( AND not_expr )*`
    fn parse_and(&mut self) -> PResult<Predicate> {
        let mut left = self.parse_not()?;
        while self.eat_kw("AND") {
            let right = self.parse_not()?;
            left = Predicate::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    /// `not_expr := NOT not_expr | atom`
    fn parse_not(&mut self) -> PResult<Predicate> {
        if self.eat_kw("NOT") {
            Ok(Predicate::Not(Box::new(self.parse_not()?)))
        } else {
            self.parse_atom()
        }
    }

    /// `atom := '(' predicate ')' | comparison`
    fn parse_atom(&mut self) -> PResult<Predicate> {
        if matches!(self.peek(), Token::LParen) {
            self.next();
            let p = self.parse_predicate()?;
            if !matches!(self.next(), Token::RParen) {
                return Err(ParseError::new("expected ')' in predicate"));
            }
            return Ok(p);
        }
        self.parse_comparison()
    }

    /// `comparison := field ( op )? value`. No operator ⇒ implicit equality (`domain "x"`).
    fn parse_comparison(&mut self) -> PResult<Predicate> {
        let field = self.parse_field()?;
        let op = self.parse_cmp_op(); // None ⇒ implicit equality
        let value = self.parse_value()?;
        Ok(Predicate::Cmp {
            field,
            op: op.unwrap_or(CmpOp::Eq),
            value,
        })
    }

    /// `field := 'weight' 'now' | 'salience' | 'age' | <trait-word>`
    fn parse_field(&mut self) -> PResult<Field> {
        let w = self.expect_word()?;
        Ok(match w.to_ascii_lowercase().as_str() {
            "weight" => {
                self.eat_kw("now"); // `weight now` ↔ `weight`: weight is always "now"
                Field::Weight
            }
            "salience" => Field::Salience,
            "age" => Field::Age,
            "resonates" | "resonance" => Field::Resonance,
            _ => Field::Trait(w),
        })
    }

    /// Comparison operator if the next token is one; otherwise `None` (implicit equality).
    fn parse_cmp_op(&mut self) -> Option<CmpOp> {
        let op = match self.peek() {
            Token::Op(s) => match s.as_str() {
                "<" => CmpOp::Lt,
                "<=" => CmpOp::Le,
                ">" => CmpOp::Gt,
                ">=" => CmpOp::Ge,
                "==" | "=" => CmpOp::Eq,
                "!=" => CmpOp::Ne,
                _ => return None,
            },
            _ => return None,
        };
        self.next();
        Some(op)
    }

    fn parse_value(&mut self) -> PResult<Value> {
        match self.next() {
            Token::Number(n) => Ok(Value::Num(n)),
            Token::Str(s) => Ok(Value::Text(s)),
            Token::Word(w) => Ok(Value::Text(w)),
            t => Err(ParseError::new(format!(
                "expected a value in predicate, found {t:?}"
            ))),
        }
    }

    fn parse_evoke(&mut self) -> PResult<Evoke> {
        self.expect_kw("EVOKE")?;
        let kind = if self.eat_kw("essence") {
            EssenceKind::Essence
        } else if self.eat_kw("archetype") {
            EssenceKind::Archetype
        } else {
            return Err(ParseError::new("expected 'essence' or 'archetype'"));
        };
        self.expect_kw("OF")?;
        let subject = self.expect_string()?;

        let mut ev = Evoke {
            kind,
            subject,
            span: None,
            resonating_with: Vec::new(),
            resolution: None,
            projecting: None,
            token_budget: None,
        };

        // Optional clauses in any order.
        loop {
            if self.eat_kw("ACROSS") {
                self.expect_kw("span")?;
                ev.span = Some(self.parse_duration()?);
            } else if self.eat_kw("RESONATING") {
                self.expect_kw("WITH")?;
                ev.resonating_with = self.parse_trait_set()?;
            } else if self.eat_kw("RESOLUTION") {
                ev.resolution = Some(if self.eat_kw("arc") {
                    Resolution::Arc
                } else if self.eat_kw("point") {
                    Resolution::Point
                } else if self.eat_kw("summary") {
                    Resolution::Summary
                } else {
                    return Err(ParseError::new("invalid RESOLUTION"));
                });
            } else if self.eat_kw("PROJECTING") {
                ev.projecting = Some(if self.eat_kw("trajectory") {
                    Projection::Trajectory
                } else if self.eat_kw("snapshot") {
                    Projection::Snapshot
                } else {
                    return Err(ParseError::new("invalid PROJECTING"));
                });
            } else if self.eat_kw("WITHIN") {
                self.expect_kw("budget")?;
                let n = self.expect_number()?;
                self.expect_kw("tokens")?;
                ev.token_budget = Some(n as usize);
            } else if self.eat_kw("RETURN") {
                self.expect_kw("compressed_context")?;
            } else {
                break;
            }
        }
        Ok(ev)
    }

    fn parse_fade(&mut self) -> PResult<Fade> {
        self.expect_kw("FADE")?;
        let target = self.expect_word()?; // "noise" | "interaction"
                                          // Optional WHERE: now a real predicate (stops at PRESERVING / next statement).
        let mut filter = None;
        if self.eat_kw("WHERE") {
            filter = Some(self.parse_predicate()?);
        }
        let mut preserving_archetype = false;
        if self.eat_kw("PRESERVING") {
            self.expect_kw("archetype_contribution")?;
            preserving_archetype = true;
        }
        Ok(Fade {
            target,
            preserving_archetype,
            filter,
        })
    }

    fn parse_imprint(&mut self) -> PResult<Imprint> {
        self.expect_kw("IMPRINT")?;
        self.expect_kw("archetype")?;
        let archetype = self.expect_string()?;
        self.expect_kw("FROM")?;
        self.expect_kw("intention_vector")?;
        let mut resilience = None;
        if self.eat_kw("RESILIENCE") {
            resilience = Some(if self.eat_kw("low") {
                Resilience::Low
            } else if self.eat_kw("medium") {
                Resilience::Medium
            } else if self.eat_kw("high") {
                Resilience::High
            } else {
                return Err(ParseError::new("invalid RESILIENCE"));
            });
        }
        Ok(Imprint {
            archetype,
            resilience,
        })
    }

    /// `RECALL facts FROM subject "u" [RESONATING WITH {..}] [WHERE pred] [WITHIN k N]`.
    fn parse_recall(&mut self) -> PResult<Recall> {
        self.expect_kw("RECALL")?;
        self.expect_kw("facts")?;
        self.expect_kw("FROM")?;
        self.expect_kw("subject")?;
        let subject = self.expect_string()?;
        let mut r = Recall {
            subject,
            resonating_with: Vec::new(),
            k: 3,
            filter: None,
        };
        loop {
            if self.eat_kw("RESONATING") {
                self.expect_kw("WITH")?;
                r.resonating_with = self.parse_trait_set()?;
            } else if self.eat_kw("WHERE") {
                r.filter = Some(self.parse_predicate()?);
            } else if self.eat_kw("WITHIN") {
                self.expect_kw("k")?;
                r.k = self.expect_number()? as usize;
            } else {
                break;
            }
        }
        Ok(r)
    }

    /// `REINFORCE facts FROM subject "u" [RESONATING WITH {..}] [WITHIN k N]`.
    fn parse_reinforce(&mut self) -> PResult<Reinforce> {
        self.expect_kw("REINFORCE")?;
        self.expect_kw("facts")?;
        self.expect_kw("FROM")?;
        self.expect_kw("subject")?;
        let subject = self.expect_string()?;
        let mut r = Reinforce {
            subject,
            resonating_with: Vec::new(),
            k: 3,
        };
        loop {
            if self.eat_kw("RESONATING") {
                self.expect_kw("WITH")?;
                r.resonating_with = self.parse_trait_set()?;
            } else if self.eat_kw("WITHIN") {
                self.expect_kw("k")?;
                r.k = self.expect_number()? as usize;
            } else {
                break;
            }
        }
        Ok(r)
    }
}

/// Parses a complete MQL program (one or more statements) into a list of [`Statement`].
pub fn parse(src: &str) -> PResult<Vec<Statement>> {
    let toks = lex(src).map_err(|e| ParseError::new(format!("lex: {} @ {}", e.message, e.pos)))?;
    let mut p = Parser { toks, pos: 0 };
    let mut stmts = Vec::new();
    while !p.at_eof() {
        stmts.push(p.parse_statement()?);
    }
    Ok(stmts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_perceive() {
        let src = r#"
            PERCEIVE interaction
              FROM subject "user:Xolotl"
              AS { act: purchase, object: "running_shoes", hue: nocturnal, urgency: high }
              WITH salience 0.2
              DECAYS halflife 18h
        "#;
        let stmts = parse(src).unwrap();
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Statement::Perceive(p) => {
                assert_eq!(p.subject, "user:Xolotl");
                assert_eq!(p.salience, Some(0.2));
                assert_eq!(p.traits.get("act").unwrap(), "purchase");
                assert!((p.halflife.unwrap().seconds - 18.0 * 3600.0).abs() < 1e-6);
            }
            other => panic!("expected Perceive, got {other:?}"),
        }
    }

    #[test]
    fn parse_distill() {
        let src = r#"DISTILL subject "user:Xolotl"
            FROM perceptions WHERE domain "ecommerce"
            INTO intention_vector
            COMPRESSING BY semantic_variance
            RETAINING { trajectory, rhythm, affect, drift }"#;
        let stmts = parse(src).unwrap();
        match &stmts[0] {
            Statement::Distill(d) => {
                assert!(d.compressing_by_variance);
                assert_eq!(d.retaining, vec!["trajectory", "rhythm", "affect", "drift"]);
            }
            other => panic!("expected Distill, got {other:?}"),
        }
    }

    #[test]
    fn parse_evoke() {
        let src = r#"EVOKE essence
            OF "user:Xolotl"
            ACROSS span 1 year
            RESONATING WITH { mood, interest, intent_drift }
            RESOLUTION arc
            PROJECTING trajectory
            WITHIN budget 800 tokens
            RETURN compressed_context"#;
        let stmts = parse(src).unwrap();
        match &stmts[0] {
            Statement::Evoke(e) => {
                assert_eq!(e.kind, EssenceKind::Essence);
                assert_eq!(e.subject, "user:Xolotl");
                assert_eq!(e.token_budget, Some(800));
                assert_eq!(e.resolution, Some(Resolution::Arc));
                assert_eq!(e.projecting, Some(Projection::Trajectory));
                assert!((e.span.unwrap().seconds - 31_536_000.0).abs() < 1e-3);
            }
            other => panic!("expected Evoke, got {other:?}"),
        }
    }

    #[test]
    fn parse_fade() {
        let src = r#"FADE noise WHERE weight now < 0.05 PRESERVING archetype_contribution"#;
        let stmts = parse(src).unwrap();
        match &stmts[0] {
            Statement::Fade(f) => {
                assert_eq!(f.target, "noise");
                assert!(f.preserving_archetype);
            }
            other => panic!("expected Fade, got {other:?}"),
        }
    }

    #[test]
    fn parse_distill_where_predicate() {
        let src = r#"DISTILL subject "u:X"
            FROM perceptions WHERE domain "ecommerce" AND salience >= 0.5
            INTO intention_vector"#;
        let stmts = parse(src).unwrap();
        match &stmts[0] {
            Statement::Distill(d) => {
                let expected = Predicate::And(
                    Box::new(Predicate::Cmp {
                        field: Field::Trait("domain".into()),
                        op: CmpOp::Eq,
                        value: Value::Text("ecommerce".into()),
                    }),
                    Box::new(Predicate::Cmp {
                        field: Field::Salience,
                        op: CmpOp::Ge,
                        value: Value::Num(0.5),
                    }),
                );
                assert_eq!(d.filter, Some(expected));
            }
            other => panic!("expected Distill, got {other:?}"),
        }
    }

    #[test]
    fn parse_fade_where_real_predicate() {
        let src = r#"FADE noise WHERE weight now < 0.05 PRESERVING archetype_contribution"#;
        match &parse(src).unwrap()[0] {
            Statement::Fade(f) => {
                assert!(f.preserving_archetype);
                assert_eq!(
                    f.filter,
                    Some(Predicate::Cmp {
                        field: Field::Weight,
                        op: CmpOp::Lt,
                        value: Value::Num(0.05),
                    })
                );
            }
            other => panic!("expected Fade, got {other:?}"),
        }
    }

    #[test]
    fn parse_predicate_precedence_and_parens() {
        // a OR b AND c  ≡  a OR (b AND c);  parentheses force (a OR b) AND c.
        let src = r#"FADE noise WHERE (mood "low" OR mood "flat") AND NOT age > 3600"#;
        match &parse(src).unwrap()[0] {
            Statement::Fade(f) => {
                let inner_or = Predicate::Or(
                    Box::new(Predicate::Cmp {
                        field: Field::Trait("mood".into()),
                        op: CmpOp::Eq,
                        value: Value::Text("low".into()),
                    }),
                    Box::new(Predicate::Cmp {
                        field: Field::Trait("mood".into()),
                        op: CmpOp::Eq,
                        value: Value::Text("flat".into()),
                    }),
                );
                let not_age = Predicate::Not(Box::new(Predicate::Cmp {
                    field: Field::Age,
                    op: CmpOp::Gt,
                    value: Value::Num(3600.0),
                }));
                let expected = Predicate::And(Box::new(inner_or), Box::new(not_age));
                assert_eq!(f.filter, Some(expected));
            }
            other => panic!("expected Fade, got {other:?}"),
        }
    }

    #[test]
    fn parse_imprint() {
        let src = r#"IMPRINT archetype "Xolotl::self"
            FROM intention_vector
            RESILIENCE high"#;
        let stmts = parse(src).unwrap();
        match &stmts[0] {
            Statement::Imprint(i) => {
                assert_eq!(i.archetype, "Xolotl::self");
                assert_eq!(i.resilience, Some(Resilience::High));
            }
            other => panic!("expected Imprint, got {other:?}"),
        }
    }

    #[test]
    fn parse_all_five_verbs_in_one_program() {
        let src = r#"
            PERCEIVE interaction FROM subject "u" AS { a: b }
            DISTILL subject "u" INTO intention_vector COMPRESSING BY semantic_variance
            IMPRINT archetype "u::self" FROM intention_vector RESILIENCE high
            EVOKE essence OF "u" WITHIN budget 800 tokens
            FADE noise PRESERVING archetype_contribution
        "#;
        let stmts = parse(src).unwrap();
        assert_eq!(stmts.len(), 5);
    }

    #[test]
    fn parse_recall_with_vector_predicate() {
        let src = r#"RECALL facts FROM subject "u"
            RESONATING WITH { vacation, paris }
            WHERE resonates > 0.5
            WITHIN k 3"#;
        match &parse(src).unwrap()[0] {
            Statement::Recall(r) => {
                assert_eq!(r.subject, "u");
                assert_eq!(r.resonating_with, vec!["vacation", "paris"]);
                assert_eq!(r.k, 3);
                assert_eq!(
                    r.filter,
                    Some(Predicate::Cmp {
                        field: Field::Resonance,
                        op: CmpOp::Gt,
                        value: Value::Num(0.5),
                    })
                );
            }
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    #[test]
    fn parse_reinforce_defaults_k_to_three() {
        match &parse(r#"REINFORCE facts FROM subject "u" RESONATING WITH { vacation }"#).unwrap()[0]
        {
            Statement::Reinforce(r) => {
                assert_eq!(r.subject, "u");
                assert_eq!(r.resonating_with, vec!["vacation"]);
                assert_eq!(r.k, 3, "WITHIN k omitted ⇒ default 3");
            }
            other => panic!("expected Reinforce, got {other:?}"),
        }
    }
}
