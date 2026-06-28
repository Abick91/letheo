//! Calibration report: sweeps thresholds over labelled synthetic data and prints
//! the Pareto frontier + recommendation. Deterministic — `cargo run -p letheo-calibration`.

use letheo_calibration::*;

const HOUR: f64 = 3600.0;
const DAY: f64 = 24.0 * HOUR;

fn main() {
    println!("════════════════════════════════════════════════════════════════════");
    println!("  Letheo · Empirical threshold sweep (labelled synthetic data)");
    println!("════════════════════════════════════════════════════════════════════\n");

    semantic_report();
    println!();
    fade_report();

    println!("\nRuntime defaults: θ_red=0.92  θ_anom=0.30  θ_fade=0.05");
    println!("(Deterministic report — fixed seeds. See crates/letheo-calibration.)");
}

fn semantic_report() {
    // 2000 redundant, 2000 signal, 1000 anomaly: realistic mix (repeated behaviour dominates).
    let events = synth_semantic(101, 2000, 2000, 1000);
    let reds = [0.80f32, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96];
    let anoms = [0.10f32, 0.20, 0.30, 0.40, 0.50, 0.60];
    let scores = sweep_semantic(&events, &reds, &anoms);

    println!("── θ_redundancy / θ_anomaly ────────────────────────────────────────");
    println!("  population: 2000 redundant · 2000 signal · 1000 anomaly\n");

    // Pareto frontier over (F1 redundancy, F1 anomaly).
    let pts: Vec<(f64, f64)> = scores
        .iter()
        .map(|s| (s.redundancy.f1(), s.anomaly.f1()))
        .collect();
    let mut front = pareto_front(&pts);
    front.sort_by(|&a, &b| pts[b].0.partial_cmp(&pts[a].0).unwrap());

    println!("  Pareto frontier (F1_red ↔ F1_anom, no point dominates another):");
    println!("    θ_red  θ_anom |  F1_red  F1_anom | fade_signal | objective");
    println!("    ───────────────┼──────────────────┼────────────┼─────────");
    for &i in &front {
        let s = &scores[i];
        println!(
            "    {:>5.2}  {:>5.2}  |  {:>5.3}  {:>6.3}  |   {:>6.1}%  | {:>6.3}",
            s.theta_redundancy,
            s.theta_anomaly,
            s.redundancy.f1(),
            s.anomaly.f1(),
            s.signal_fade_rate * 100.0,
            s.objective(),
        );
    }

    let best = scores
        .iter()
        .max_by(|a, b| a.objective().partial_cmp(&b.objective()).unwrap())
        .unwrap();
    let default = score_semantic(&events, 0.92, 0.30);
    println!(
        "\n  ► Optimum (objective): θ_red={:.2} θ_anom={:.2} → {:.3}",
        best.theta_redundancy,
        best.theta_anomaly,
        best.objective()
    );
    println!(
        "  ► Default (0.92/0.30):                 → {:.3}",
        default.objective()
    );
    verdict(best.objective(), default.objective());
}

fn fade_report() {
    let events = synth_decay(202, 3000, 3000);
    let thetas = [0.01f64, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20];
    let horizon = 3.0 * DAY;
    let scores = sweep_fade(&events, &thetas, horizon);

    println!("── θ_fade (horizon = 3 days) ───────────────────────────────────────");
    println!("  population: 3000 noise (half-life hours) · 3000 memory (days)\n");
    println!("    θ_fade |  P_noise  R_noise    F1 | amnesia | objective");
    println!("    ───────┼─────────────────────────┼─────────┼─────────");
    for s in &scores {
        println!(
            "    {:>6.3} |   {:>5.3}    {:>5.3}  {:>5.3} | {:>5.1}%  | {:>6.3}",
            s.theta_fade,
            s.fade.precision(),
            s.fade.recall(),
            s.fade.f1(),
            s.memory_loss_rate * 100.0,
            s.objective(),
        );
    }
    let best = scores
        .iter()
        .max_by(|a, b| a.objective().partial_cmp(&b.objective()).unwrap())
        .unwrap();
    let default = score_fade(&events, 0.05, horizon);
    println!(
        "\n  ► Optimum (objective): θ_fade={:.3} → {:.3}",
        best.theta_fade,
        best.objective()
    );
    println!(
        "  ► Default (0.05):             → {:.3}",
        default.objective()
    );
    verdict(best.objective(), default.objective());
}

fn verdict(best: f64, default: f64) {
    if best - default < 0.01 {
        println!("  ✓ Default is essentially at the optimum (Δ < 0.01): confirmed.");
    } else {
        println!(
            "  ⚠ Sweep improves default by Δ={:.3}: consider re-calibrating.",
            best - default
        );
    }
}
