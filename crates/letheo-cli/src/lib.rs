//! # letheo-cli · MQL REPL
//!
//! Closes the product experience: *touch* the language without writing Rust or Python. Reads MQL,
//! executes it against an `Executor<P: Provider>` (real Candle in the binary) and displays the
//! resolved context. Carries a logical clock advanced with `:tick`, and persists memory with `:save`/`:load`.
//!
//! Logic lives here (testable without a TTY); `main.rs` only connects stdin/stdout.

use std::path::PathBuf;

use letheo_core::{CognitiveRuntime, RuntimeConfig, Tick};
use letheo_exec::{ExecError, ExecResult, Executor};
use letheo_inference::{CandleProvider, Provider};
use letheo_mql::{parse, validate};

/// Result of evaluating a REPL line.
#[derive(Debug, PartialEq)]
pub enum Eval {
    /// Text to display to the user.
    Output(String),
    /// The user requested to quit.
    Quit,
}

/// REPL state: the runtime, the logical clock and the optional persistence path.
/// Generic over `Provider`: the binary uses `CandleProvider` (real); tests use `MockProvider`.
pub struct Repl<P: Provider> {
    exec: Executor<P>,
    now: Tick,
    persist: Option<PathBuf>,
}

/// Production REPL: real embeddings (Candle).
pub type RealRepl = Repl<CandleProvider>;

impl Repl<CandleProvider> {
    /// Creates the production REPL with a real Candle provider. Requires `LETHEO_MODEL_DIR`.
    pub fn real(persist: Option<PathBuf>) -> std::io::Result<Self> {
        let provider = CandleProvider::load().map_err(|e| std::io::Error::other(e.to_string()))?;
        Self::with_provider(provider, persist)
    }
}

impl<P: Provider> Repl<P> {
    /// Creates a REPL with the given `provider`. If `persist` points to snapshots, rehydrates memory.
    pub fn with_provider(provider: P, persist: Option<PathBuf>) -> std::io::Result<Self> {
        let mut exec = Executor::new(CognitiveRuntime::new(RuntimeConfig::default()), provider);
        if let Some(dir) = &persist {
            let store = letheo_persist::load_store(dir)?;
            *exec.runtime_mut().long_term_mut() = store;
        }
        Ok(Self {
            exec,
            now: 0.0,
            persist,
        })
    }

    pub fn now(&self) -> Tick {
        self.now
    }

    /// Evaluates a line: meta-command (`:`) or MQL program.
    pub fn eval(&mut self, input: &str) -> Eval {
        let line = input.trim();
        if line.is_empty() {
            return Eval::Output(String::new());
        }
        if let Some(cmd) = line.strip_prefix(':') {
            return self.meta(cmd.trim());
        }
        self.run_mql(line)
    }

    fn meta(&mut self, cmd: &str) -> Eval {
        let mut parts = cmd.splitn(2, char::is_whitespace);
        let verb = parts.next().unwrap_or("");
        let arg = parts.next().map(str::trim).filter(|s| !s.is_empty());

        match verb {
            "q" | "quit" | "exit" => Eval::Quit,
            "help" | "h" | "?" => Eval::Output(HELP.to_string()),
            "now" => Eval::Output(format!("now = {:.0}s", self.now)),
            "tick" => match arg.and_then(|a| a.parse::<f64>().ok()) {
                Some(s) if s >= 0.0 => {
                    self.now += s;
                    Eval::Output(format!("⏱  now = {:.0}s", self.now))
                }
                _ => Eval::Output("usage: :tick <seconds ≥ 0>".into()),
            },
            "state" => Eval::Output(format!(
                "short-term: {} perceptions · long-term: {} archetypes",
                self.exec.runtime().short_term_len(),
                self.exec.runtime().long_term_len(),
            )),
            "subjects" => {
                let subs: Vec<&str> = self
                    .exec
                    .runtime()
                    .long_term()
                    .iter()
                    .map(|a| a.subject.as_str())
                    .collect();
                Eval::Output(if subs.is_empty() {
                    "(no consolidated archetypes)".into()
                } else {
                    subs.join("\n")
                })
            }
            "save" => {
                let dir = arg.map(PathBuf::from).or_else(|| self.persist.clone());
                match dir {
                    None => Eval::Output("usage: :save <dir>  (or start with --persist)".into()),
                    Some(d) => {
                        match letheo_persist::save_store(&d, self.exec.runtime().long_term()) {
                            Ok(n) => {
                                Eval::Output(format!("💾 saved {n} archetypes to {}", d.display()))
                            }
                            Err(e) => Eval::Output(format!("save error: {e}")),
                        }
                    }
                }
            }
            "load" => {
                let dir = arg.map(PathBuf::from).or_else(|| self.persist.clone());
                match dir {
                    None => Eval::Output("usage: :load <dir>  (or start with --persist)".into()),
                    Some(d) => match letheo_persist::load_store(&d) {
                        Ok(store) => {
                            let n = store.len();
                            *self.exec.runtime_mut().long_term_mut() = store;
                            Eval::Output(format!("📂 loaded {n} archetypes from {}", d.display()))
                        }
                        Err(e) => Eval::Output(format!("load error: {e}")),
                    },
                }
            }
            other => Eval::Output(format!("unknown command: ':{other}' — try :help")),
        }
    }

    fn run_mql(&mut self, src: &str) -> Eval {
        let stmts = match parse(src) {
            Ok(s) => s,
            Err(e) => return Eval::Output(format!("⚠ syntax error: {}", e.message)),
        };
        // Semantic validation: if the program makes no sense, don't execute it halfway.
        let problems = validate(&stmts);
        if !problems.is_empty() {
            let msg: Vec<String> = problems.iter().map(|p| format!("⚠ {p}")).collect();
            return Eval::Output(msg.join("\n"));
        }
        let mut lines = Vec::new();
        for stmt in &stmts {
            lines.push(match self.exec.execute(stmt, self.now) {
                Ok(r) => format_result(&r),
                Err(e) => format!("⚠ {}", format_error(&e)),
            });
        }
        Eval::Output(lines.join("\n"))
    }

    /// Saves memory if a persistence path is configured (called on exit).
    pub fn autosave(&self) -> Option<std::io::Result<usize>> {
        self.persist
            .as_ref()
            .map(|dir| letheo_persist::save_store(dir, self.exec.runtime().long_term()))
    }
}

fn format_result(r: &ExecResult) -> String {
    match r {
        ExecResult::Perceived { subject } => format!("· perceived «{subject}»"),
        ExecResult::Dreamed(b) => format!(
            "· dreamed: {} subject(s) consolidated, {} perception(s) absorbed, {} faded",
            b.distilled_subjects, b.perceptions_absorbed, b.faded
        ),
        ExecResult::Evoked(c) => format!(
            "· evoked «{}»: {} events → {} vectors · ~{} tokens · compression {:.1}:1",
            c.subject,
            c.represented,
            c.vectors_returned,
            c.token_estimate,
            c.compression_ratio()
        ),
        ExecResult::Faded { swept } => format!("· faded {swept} perception(s)"),
        ExecResult::Imprinted { archetype, .. } => {
            format!("· imprinted «{archetype}» (consolidated essence)")
        }
        ExecResult::Recalled(facts) => {
            if facts.is_empty() {
                "· recall: no resonating facts".to_string()
            } else {
                let items: Vec<String> = facts
                    .iter()
                    .map(|f| format!("«{}» ({:.2})", f.text, f.score))
                    .collect();
                format!("· recalled {} fact(s): {}", facts.len(), items.join(", "))
            }
        }
        ExecResult::Reinforced { count } => format!("· reinforced {count} fact(s) (decay reset)"),
    }
}

fn format_error(e: &ExecError) -> String {
    match e {
        ExecError::NoSuchSubject(s) => format!("no live archetype for «{s}»"),
        ExecError::MissingBudget => "EVOKE requires WITHIN budget N tokens".into(),
    }
}

pub const HELP: &str = "\
MQL verbs — type them directly:
  PERCEIVE interaction FROM subject \"u:X\" AS { act: buy, object: shoes }
  DISTILL  subject \"u:X\" INTO intention_vector COMPRESSING BY semantic_variance
  EVOKE    essence OF \"u:X\" WITHIN budget 800 tokens
  EVOKE    essence OF \"u:X\" RESONATING WITH { nostalgia } WITHIN budget 800 tokens
  FADE     noise WHERE weight now < 0.05 PRESERVING archetype_contribution
  IMPRINT  archetype \"u:X\" FROM intention_vector RESILIENCE high
  RECALL   facts FROM subject \"u:X\" RESONATING WITH { allergy } WHERE resonates > 0.6 WITHIN k 3
  REINFORCE facts FROM subject \"u:X\" RESONATING WITH { allergy } WITHIN k 3

Meta-commands:
  :tick <s>     advance the logical clock <s> seconds
  :now          show the clock
  :state        short/long-term memory size
  :subjects     consolidated archetypes
  :save [dir]   persist memory (one JSON per subject)
  :load [dir]   rehydrate memory from disk
  :help         this help
  :quit         exit";

#[cfg(test)]
mod tests {
    use super::*;
    use letheo_inference::MockProvider;

    fn repl() -> Repl<MockProvider> {
        Repl::with_provider(MockProvider::new(), None).unwrap()
    }

    fn out(e: Eval) -> String {
        match e {
            Eval::Output(s) => s,
            Eval::Quit => "<quit>".into(),
        }
    }

    #[test]
    fn full_session_perceive_distill_evoke() {
        let mut r = repl();
        out(r.eval(r#"PERCEIVE interaction FROM subject "u:X" AS { act: buy }"#));
        out(r.eval(r#"PERCEIVE interaction FROM subject "u:X" AS { act: buy }"#));
        let dreamed = out(r.eval(r#"DISTILL subject "u:X" INTO intention_vector"#));
        assert!(dreamed.contains("consolidated"), "{dreamed}");
        let evoked = out(r.eval(r#"EVOKE essence OF "u:X" WITHIN budget 800 tokens"#));
        assert!(
            evoked.contains("evoked") && evoked.contains("compression"),
            "{evoked}"
        );
    }

    #[test]
    fn tick_advances_logical_clock() {
        let mut r = repl();
        assert_eq!(r.now(), 0.0);
        out(r.eval(":tick 3600"));
        assert_eq!(r.now(), 3600.0);
        out(r.eval(":tick 60"));
        assert_eq!(r.now(), 3660.0);
    }

    #[test]
    fn syntax_error_is_reported_not_panicked() {
        let mut r = repl();
        let o = out(r.eval("PERCEIVE wat"));
        assert!(o.contains("syntax error"), "{o}");
    }

    #[test]
    fn quit_command() {
        let mut r = repl();
        assert_eq!(r.eval(":quit"), Eval::Quit);
        assert_eq!(r.eval(":q"), Eval::Quit);
    }

    #[test]
    fn save_then_load_via_meta_commands() {
        let mut dir = std::env::temp_dir();
        dir.push(format!("letheo_cli_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.to_str().unwrap().to_string();

        // Session 1: consolidate and save.
        let mut r = repl();
        out(r.eval(r#"PERCEIVE interaction FROM subject "u:X" AS { act: buy }"#));
        out(r.eval(r#"DISTILL subject "u:X" INTO intention_vector"#));
        let saved = out(r.eval(&format!(":save {path}")));
        assert!(saved.contains("saved 1"), "{saved}");

        // Session 2: load and evoke what was learned before.
        let mut r2 = repl();
        let loaded = out(r2.eval(&format!(":load {path}")));
        assert!(loaded.contains("loaded 1"), "{loaded}");
        let subs = out(r2.eval(":subjects"));
        assert!(subs.contains("u:X"), "{subs}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn semantic_error_blocks_execution() {
        let mut r = repl();
        // budget 0 is syntactically valid but semantically nonsensical: must not execute.
        let o = out(r.eval(r#"EVOKE essence OF "u:X" WITHIN budget 0 tokens"#));
        assert!(o.contains("budget") && o.contains("> 0"), "{o}");
    }

    #[test]
    fn unknown_meta_command_is_friendly() {
        let mut r = repl();
        let o = out(r.eval(":wat"));
        assert!(o.contains("unknown"), "{o}");
    }
}
