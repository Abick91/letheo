//! Binary `letheo`: interactive MQL REPL.
//!
//! Usage:
//!   letheo                      interactive REPL
//!   letheo --persist ./mem      REPL that auto-loads/saves memory
//!   letheo --exec "<mql>"       executes a program and exits (non-interactive)

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use letheo_cli::{Eval, RealRepl, HELP};

fn main() {
    let mut persist: Option<PathBuf> = None;
    let mut exec_src: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--persist" | "-p" => persist = args.next().map(PathBuf::from),
            "--exec" | "-e" => exec_src = args.next(),
            "--help" | "-h" => {
                println!("{HELP}");
                return;
            }
            other => {
                eprintln!("unknown argument: {other} (try --help)");
                std::process::exit(2);
            }
        }
    }

    let mut repl = match RealRepl::real(persist.clone()) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("startup failed (is LETHEO_MODEL_DIR set to the all-MiniLM-L6-v2 model?): {e}");
            std::process::exit(1);
        }
    };

    // Modo no interactivo: ejecuta y sale.
    if let Some(src) = exec_src {
        if let Eval::Output(s) = repl.eval(&src) {
            if !s.is_empty() {
                println!("{s}");
            }
        }
        autosave(&repl);
        return;
    }

    // Modo interactivo.
    println!("Letheo · MQL REPL — :help for help, :quit to exit");
    if persist.is_some() {
        println!(
            "(persistence active: {} archetypes loaded)",
            repl.eval(":subjects").describe_count()
        );
    }

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    loop {
        print!("mql> ");
        let _ = stdout.flush();

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break, // EOF (Ctrl-D)
            Ok(_) => {}
            Err(e) => {
                eprintln!("read error: {e}");
                break;
            }
        }

        match repl.eval(&line) {
            Eval::Quit => break,
            Eval::Output(s) => {
                if !s.is_empty() {
                    println!("{s}");
                }
            }
        }
    }

    autosave(&repl);
    println!("goodbye.");
}

fn autosave(repl: &RealRepl) {
    if let Some(result) = repl.autosave() {
        match result {
            Ok(n) => eprintln!("💾 memory saved ({n} archetypes)"),
            Err(e) => eprintln!("⚠ autosave failed: {e}"),
        }
    }
}

/// Small helper for the banner: counts non-empty lines in output.
trait DescribeCount {
    fn describe_count(self) -> usize;
}
impl DescribeCount for Eval {
    fn describe_count(self) -> usize {
        match self {
            Eval::Output(s) if s.starts_with('(') => 0, // "(no archetypes...)"
            Eval::Output(s) => s.lines().filter(|l| !l.is_empty()).count(),
            Eval::Quit => 0,
        }
    }
}
