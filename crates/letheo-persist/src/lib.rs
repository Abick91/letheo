//! # letheo-persist · Local-first persistence of long-term memory
//!
//! Letheo's persistence layer: makes the distilled essence **survive process restarts**. Saves one
//! *snapshot per subject* (one JSON file per archetype) in a directory, and rehydrates it.
//!
//! Design decisions:
//! - **Separate crate with serde**, not in `letheo-core`: the core stays free of serialization
//!   dependencies. Mirror DTOs + conversions from/to core types live here.
//! - **One file per subject** (`{sanitized-subject}-{hash}.json`): independent snapshots, easy to
//!   inspect, diff, and migrate; the name includes a hash to avoid sanitization collisions.
//! - **Human-readable JSON**: an agent's memory must be auditable by hand (not an opaque blob).

use std::fs;
use std::io;
use std::path::Path;

use letheo_core::archetype::ArcMilestone;
use letheo_core::entropy::EntropyTrace;
use letheo_core::factstore::{Fact, FactStore};
use letheo_core::modes::Mode;
use letheo_core::{Archetype, ArchetypeStore};
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};

/// On-disk format version. Allows future migrations without breaking old snapshots.
pub const SNAPSHOT_VERSION: u32 = 1;

// ─────────────────────────────────────────────────────────────────────────────
// Mirror DTOs (serde) — decouple the on-disk format from the runtime types.
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct TraceDto {
    salience: f64,
    lambda: f64,
    reinforcement: f64,
    last_touch: f64,
}

#[derive(Serialize, Deserialize)]
struct MilestoneDto {
    at: f64,
    direction: Vec<f32>,
    absorbed: usize,
    #[serde(default)]
    label: String,
    #[serde(default)]
    label_histogram: Vec<(String, usize)>,
}

/// DTO for one archetype mode (multi-modal). Its physics (`trace`) is persisted just like the
/// archetype's, so per-mode forgetting survives restarts — lossless round-trip.
#[derive(Serialize, Deserialize)]
struct ModeDto {
    centroid: Vec<f32>,
    /// Birth direction of the mode (drift origin). `default` for pre-multi-modal snapshots: if empty
    /// on load, the centroid is used (drift 0 — honest: no birth origin was recorded).
    #[serde(default)]
    origin: Vec<f32>,
    #[serde(default)]
    label: String,
    #[serde(default)]
    label_histogram: Vec<(String, usize)>,
    absorbed: usize,
    trace: TraceDto,
}

#[derive(Serialize, Deserialize)]
struct ArchetypeDto {
    version: u32,
    subject: String,
    core: Vec<f32>,
    anomalies: Vec<Vec<f32>>,
    #[serde(default)]
    anomaly_labels: Vec<String>,
    #[serde(default)]
    core_label: String,
    represented: usize,
    arc: Vec<MilestoneDto>,
    trace: TraceDto,
    /// Archetype modes. `default` to load pre-multi-modal snapshots without error.
    #[serde(default)]
    modes: Vec<ModeDto>,
}

impl From<&Archetype> for ArchetypeDto {
    fn from(a: &Archetype) -> Self {
        ArchetypeDto {
            version: SNAPSHOT_VERSION,
            subject: a.subject.clone(),
            core: a.core.clone(),
            anomalies: a.anomalies.clone(),
            anomaly_labels: a.anomaly_labels.clone(),
            core_label: a.core_label.clone(),
            represented: a.represented,
            arc: a
                .arc
                .iter()
                .map(|m| MilestoneDto {
                    at: m.at,
                    direction: m.direction.clone(),
                    absorbed: m.absorbed,
                    label: m.label.clone(),
                    label_histogram: m.label_histogram.clone(),
                })
                .collect(),
            trace: TraceDto {
                salience: a.trace.salience,
                lambda: a.trace.lambda,
                reinforcement: a.trace.reinforcement,
                last_touch: a.trace.last_touch,
            },
            modes: a
                .modes
                .iter()
                .map(|m| ModeDto {
                    centroid: m.centroid.clone(),
                    origin: m.origin.clone(),
                    label: m.label.clone(),
                    label_histogram: m.label_histogram.clone(),
                    absorbed: m.absorbed,
                    trace: TraceDto {
                        salience: m.trace.salience,
                        lambda: m.trace.lambda,
                        reinforcement: m.trace.reinforcement,
                        last_touch: m.trace.last_touch,
                    },
                })
                .collect(),
        }
    }
}

impl From<ArchetypeDto> for Archetype {
    fn from(d: ArchetypeDto) -> Self {
        Archetype {
            subject: d.subject,
            core: d.core,
            modes: d
                .modes
                .into_iter()
                .map(|m| Mode {
                    // Legacy: snapshots without `origin` → use centroid (drift 0, honest).
                    origin: if m.origin.is_empty() {
                        m.centroid.clone()
                    } else {
                        m.origin
                    },
                    centroid: m.centroid,
                    label: m.label,
                    label_histogram: m.label_histogram,
                    absorbed: m.absorbed,
                    trace: EntropyTrace {
                        salience: m.trace.salience,
                        lambda: m.trace.lambda,
                        reinforcement: m.trace.reinforcement,
                        last_touch: m.trace.last_touch,
                    },
                })
                .collect(),
            anomalies: d.anomalies,
            anomaly_labels: d.anomaly_labels,
            core_label: d.core_label,
            represented: d.represented,
            arc: d
                .arc
                .into_iter()
                .map(|m| ArcMilestone {
                    at: m.at,
                    direction: m.direction,
                    absorbed: m.absorbed,
                    label: m.label,
                    label_histogram: m.label_histogram,
                })
                .collect(),
            trace: EntropyTrace {
                salience: d.trace.salience,
                lambda: d.trace.lambda,
                reinforcement: d.trace.reinforcement,
                last_touch: d.trace.last_touch,
            },
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Episodic layer DTOs (layer-1): verbatim facts with forgetting physics.
// ─────────────────────────────────────────────────────────────────────────────

/// DTO for one episodic fact. Its `trace` is persisted like the archetype's: forgetting (and
/// reinforcement gained through recall/repetition) survives restarts — lossless round-trip.
#[derive(Serialize, Deserialize)]
struct FactDto {
    subject: String,
    text: String,
    embedding: Vec<f32>,
    #[serde(default)]
    provenance: String,
    created_at: f64,
    trace: TraceDto,
}

/// DTO for the full `FactStore` (a single `facts.json` file). Subject sharding and on-disk indexing
/// come with the embedded storage engine; here a human-readable diffable snapshot is sufficient.
#[derive(Serialize, Deserialize)]
struct FactStoreDto {
    version: u32,
    theta_dedup: f32,
    facts: Vec<FactDto>,
}

impl From<&FactStore> for FactStoreDto {
    fn from(s: &FactStore) -> Self {
        FactStoreDto {
            version: SNAPSHOT_VERSION,
            theta_dedup: s.theta_dedup(),
            facts: s
                .iter()
                .map(|f| FactDto {
                    subject: f.subject.clone(),
                    text: f.text.clone(),
                    embedding: f.embedding.clone(),
                    provenance: f.provenance.clone(),
                    created_at: f.created_at,
                    trace: TraceDto {
                        salience: f.trace.salience,
                        lambda: f.trace.lambda,
                        reinforcement: f.trace.reinforcement,
                        last_touch: f.trace.last_touch,
                    },
                })
                .collect(),
        }
    }
}

impl From<FactStoreDto> for FactStore {
    fn from(d: FactStoreDto) -> Self {
        let mut store = FactStore::with_dedup(d.theta_dedup);
        for f in d.facts {
            store.insert(Fact {
                subject: f.subject,
                text: f.text,
                embedding: f.embedding,
                provenance: f.provenance,
                created_at: f.created_at,
                trace: EntropyTrace {
                    salience: f.trace.salience,
                    lambda: f.trace.lambda,
                    reinforcement: f.trace.reinforcement,
                    last_touch: f.trace.last_touch,
                },
            });
        }
        store
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stable filename per subject
// ─────────────────────────────────────────────────────────────────────────────

/// 64-bit FNV-1a hash — deterministic, no dependencies. Used only for filename generation.
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Snapshot filename for a subject: human-readable prefix + collision-resistant hash.
pub fn snapshot_filename(subject: &str) -> String {
    let safe: String = subject
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    // Truncate the readable prefix to avoid giant filenames.
    let safe: String = safe.chars().take(48).collect();
    format!("{safe}-{:016x}.json", fnv1a(subject))
}

fn json_err(e: serde_json::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Saves a single archetype as a snapshot in `dir`. Creates the directory if it does not exist.
pub fn save_archetype(dir: impl AsRef<Path>, a: &Archetype) -> io::Result<()> {
    let dir = dir.as_ref();
    fs::create_dir_all(dir)?;
    let dto = ArchetypeDto::from(a);
    let json = serde_json::to_string_pretty(&dto).map_err(json_err)?;
    let path = dir.join(snapshot_filename(&a.subject));
    fs::write(path, json)
}

/// Persists the entire long-term memory: one file per subject. Returns how many were saved.
pub fn save_store(dir: impl AsRef<Path>, store: &ArchetypeStore) -> io::Result<usize> {
    let dir = dir.as_ref();
    fs::create_dir_all(dir)?;
    let mut n = 0;
    for a in store.iter() {
        save_archetype(dir, a)?;
        n += 1;
    }
    Ok(n)
}

/// Loads a single snapshot from a `.json` file.
pub fn load_archetype(path: impl AsRef<Path>) -> io::Result<Archetype> {
    let bytes = fs::read(path)?;
    let dto: ArchetypeDto = serde_json::from_slice(&bytes).map_err(json_err)?;
    Ok(dto.into())
}

/// Rehydrates an `ArchetypeStore` from a directory of snapshots. Ignores non-`.json` files.
/// A non-existent directory is treated as an empty store (first startup).
pub fn load_store(dir: impl AsRef<Path>) -> io::Result<ArchetypeStore> {
    let dir = dir.as_ref();
    let mut store = ArchetypeStore::new();
    if !dir.exists() {
        return Ok(store);
    }
    // Stable sort by filename → deterministic restoration. Exclude `facts.json` (layer-1 lives in
    // the same directory but is NOT an archetype: loaded separately by `load_facts`).
    let mut paths: Vec<_> = fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
        .filter(|p| p.file_name().and_then(|n| n.to_str()) != Some(FACTS_FILENAME))
        .collect();
    paths.sort();
    for p in paths {
        store.insert(load_archetype(p)?);
    }
    Ok(store)
}

/// Filename for the episodic layer inside the memory directory. Does not collide with per-subject
/// snapshots (`{prefix}-{hash}.json`) because it carries no hash suffix.
pub const FACTS_FILENAME: &str = "facts.json";

/// Persists the full episodic memory (layer-1) as a single `facts.json`. Returns how many facts
/// were saved. Creates the directory if it does not exist.
pub fn save_facts(dir: impl AsRef<Path>, store: &FactStore) -> io::Result<usize> {
    let dir = dir.as_ref();
    fs::create_dir_all(dir)?;
    let dto = FactStoreDto::from(store);
    let n = dto.facts.len();
    let json = serde_json::to_string_pretty(&dto).map_err(json_err)?;
    fs::write(dir.join(FACTS_FILENAME), json)?;
    Ok(n)
}

/// Rehydrates episodic memory from `facts.json`. A missing file is treated as an empty store
/// (first startup), same as [`load_store`].
pub fn load_facts(dir: impl AsRef<Path>) -> io::Result<FactStore> {
    let path = dir.as_ref().join(FACTS_FILENAME);
    if !path.exists() {
        return Ok(FactStore::new());
    }
    let bytes = fs::read(path)?;
    let dto: FactStoreDto = serde_json::from_slice(&bytes).map_err(json_err)?;
    Ok(dto.into())
}

// ─────────────────────────────────────────────────────────────────────────────
// Embedded store (redb) — transactional KV, single-file, ACID, pure-Rust.
//
// The per-subject JSON (above) is inspectable and diffable but not transactional or multi-tenant at
// scale. This embedded store is a single redb file with archetypes **keyed by subject** (one can be
// updated without rewriting the others) and layer-1 as a blob. Every write is an atomic ACID
// transaction — a crash mid-write leaves no corrupt state. Same serde DTOs are reused; the JSON
// export is kept for inspection.
// ─────────────────────────────────────────────────────────────────────────────

const ARCHETYPES: TableDefinition<&str, &[u8]> = TableDefinition::new("archetypes");
const META: TableDefinition<&str, &[u8]> = TableDefinition::new("meta");

fn db_err<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::other(format!("redb: {e}"))
}

/// Transactional embedded store for both memory layers in a single redb file.
pub struct DbStore {
    db: Database,
}

impl DbStore {
    /// Opens (or creates) the store at `path` (a file, e.g. `memory.redb`).
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let db = Database::create(path).map_err(db_err)?;
        Ok(Self { db })
    }

    /// Upserts **a single subject** in an atomic transaction — multi-tenant: does not touch others.
    pub fn write_archetype(&self, a: &Archetype) -> io::Result<()> {
        let wtxn = self.db.begin_write().map_err(db_err)?;
        {
            let mut t = wtxn.open_table(ARCHETYPES).map_err(db_err)?;
            let bytes = serde_json::to_vec(&ArchetypeDto::from(a)).map_err(json_err)?;
            t.insert(a.subject.as_str(), bytes.as_slice())
                .map_err(db_err)?;
        }
        wtxn.commit().map_err(db_err)?;
        Ok(())
    }

    /// Upserts the **entire** layer-2 in a single transaction. Returns how many archetypes were saved.
    pub fn write_store(&self, store: &ArchetypeStore) -> io::Result<usize> {
        let wtxn = self.db.begin_write().map_err(db_err)?;
        let mut n = 0;
        {
            let mut t = wtxn.open_table(ARCHETYPES).map_err(db_err)?;
            for a in store.iter() {
                let bytes = serde_json::to_vec(&ArchetypeDto::from(a)).map_err(json_err)?;
                t.insert(a.subject.as_str(), bytes.as_slice())
                    .map_err(db_err)?;
                n += 1;
            }
        }
        wtxn.commit().map_err(db_err)?;
        Ok(n)
    }

    /// Rehydrates the full layer-2 from the store.
    pub fn read_store(&self) -> io::Result<ArchetypeStore> {
        let mut store = ArchetypeStore::new();
        let rtxn = self.db.begin_read().map_err(db_err)?;
        let table = match rtxn.open_table(ARCHETYPES) {
            Ok(t) => t,
            Err(_) => return Ok(store), // table not yet created → first startup
        };
        for row in table.iter().map_err(db_err)? {
            let (_k, v) = row.map_err(db_err)?;
            let dto: ArchetypeDto = serde_json::from_slice(v.value()).map_err(json_err)?;
            store.insert(dto.into());
        }
        Ok(store)
    }

    /// Persists layer-1 (facts) as a transactional blob. Returns how many facts were saved.
    pub fn write_facts(&self, store: &FactStore) -> io::Result<usize> {
        let dto = FactStoreDto::from(store);
        let n = dto.facts.len();
        let bytes = serde_json::to_vec(&dto).map_err(json_err)?;
        let wtxn = self.db.begin_write().map_err(db_err)?;
        {
            let mut t = wtxn.open_table(META).map_err(db_err)?;
            t.insert("factstore", bytes.as_slice()).map_err(db_err)?;
        }
        wtxn.commit().map_err(db_err)?;
        Ok(n)
    }

    /// Rehydrates layer-1 from the store.
    pub fn read_facts(&self) -> io::Result<FactStore> {
        let rtxn = self.db.begin_read().map_err(db_err)?;
        let table = match rtxn.open_table(META) {
            Ok(t) => t,
            Err(_) => return Ok(FactStore::new()),
        };
        match table.get("factstore").map_err(db_err)? {
            Some(v) => {
                let dto: FactStoreDto = serde_json::from_slice(v.value()).map_err(json_err)?;
                Ok(dto.into())
            }
            None => Ok(FactStore::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use letheo_core::synthesis::IntentionVector;
    use letheo_core::Resilience;
    use std::env;

    fn tmp_dir(tag: &str) -> std::path::PathBuf {
        let mut d = env::temp_dir();
        d.push(format!("letheo_persist_{tag}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    fn iv(subject: &str, c: Vec<f32>, absorbed: usize) -> IntentionVector {
        IntentionVector {
            subject: subject.into(),
            centroid: c,
            anomalies: vec![vec![0.0, 1.0]],
            core_label: "core".into(),
            anomaly_labels: vec!["novelty".into()],
            absorbed,
            redundant: 0,
            label_histogram: vec![("core".into(), absorbed)],
            modes: vec![],
        }
    }

    fn sample_store() -> ArchetypeStore {
        let mut s = ArchetypeStore::new();
        s.imprint(
            &iv("user:Xolotl", vec![1.0, 0.0], 1000),
            Resilience::High,
            0.0,
        );
        s.imprint(
            &iv("user:Xolotl", vec![0.0, 1.0], 500),
            Resilience::High,
            3600.0,
        ); // evolves
        s.imprint(
            &iv("agent:Tlaloc", vec![0.3, 0.7], 42),
            Resilience::Medium,
            0.0,
        );
        s
    }

    #[test]
    fn filename_is_collision_resistant() {
        // Subjects that sanitize to the same prefix must differ by hash.
        assert_ne!(snapshot_filename("user:X"), snapshot_filename("user_X"));
    }

    #[test]
    fn roundtrip_preserves_every_field() {
        let dir = tmp_dir("roundtrip");
        let original = sample_store();
        let saved = save_store(&dir, &original).unwrap();
        assert_eq!(saved, 2, "two distinct subjects → two files");

        let restored = load_store(&dir).unwrap();
        assert_eq!(restored.len(), 2);

        let a = original.get("user:Xolotl").unwrap();
        let b = restored.get("user:Xolotl").unwrap();
        assert_eq!(a.subject, b.subject);
        assert_eq!(a.represented, b.represented);
        assert_eq!(a.core, b.core);
        assert_eq!(a.anomalies, b.anomalies);
        assert_eq!(a.arc.len(), b.arc.len(), "evolutionary arc survives");
        assert_eq!(a.arc[1].absorbed, b.arc[1].absorbed);
        // Exact archetype physics preserved (including half-life gained through consolidation).
        assert_eq!(a.trace.lambda, b.trace.lambda);
        assert_eq!(a.trace.reinforcement, b.trace.reinforcement);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn resonance_survives_restart() {
        let dir = tmp_dir("resonance");
        save_store(&dir, &sample_store()).unwrap();
        let restored = load_store(&dir).unwrap();
        // Rehydrated memory still resonates: a query close to Xolotl retrieves them.
        let top = restored.resonate(
            &[0.6, 0.4],
            1,
            7200.0,
            letheo_core::entropy::DEFAULT_THETA_FADE,
        );
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].subject, "user:Xolotl");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn runtime_long_term_survives_restart() {
        use letheo_core::{CognitiveRuntime, EvokeRequest, Perception, RuntimeConfig};
        let dir = tmp_dir("runtime");

        // A runtime lives, dreams, and consolidates an essence…
        let mut rt = CognitiveRuntime::new(RuntimeConfig::default());
        for _ in 0..50 {
            rt.perceive(Perception::new(
                "user:X",
                vec![1.0, 0.0],
                1.0,
                86_400.0,
                0.0,
            ));
        }
        rt.breathe(&["user:X"], 0.0);
        save_store(&dir, rt.long_term()).unwrap();
        drop(rt); // process "restarts".

        // A new runtime rehydrates its long-term memory from disk…
        let mut reborn = CognitiveRuntime::new(RuntimeConfig::default());
        assert_eq!(reborn.long_term_len(), 0);
        *reborn.long_term_mut() = load_store(&dir).unwrap();

        // …and can evoke what it learned in its "previous life".
        let ctx = reborn
            .evoke(&EvokeRequest::new("user:X", 800), 0.0)
            .expect("essence survived restart");
        assert_eq!(ctx.represented, 50);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_dir_loads_empty() {
        let dir = tmp_dir("nonexistent_xyz");
        let store = load_store(&dir).unwrap();
        assert!(store.is_empty(), "first startup: empty store, no error");
    }

    #[test]
    fn factstore_roundtrip_preserves_facts_and_physics() {
        use letheo_core::FactStore;
        let dir = tmp_dir("facts");

        let mut fs = FactStore::new();
        fs.remember(
            "user:X",
            "allergic to peanuts",
            vec![0.0, 1.0],
            "agentA",
            1.0,
            86_400.0,
            0.0,
        );
        fs.remember(
            "user:X",
            "drives a red car",
            vec![1.0, 0.0],
            "agentB",
            0.8,
            86_400.0,
            1.0,
        );
        // Recall one: gains reinforcement + consolidated λ → physics is no longer trivial.
        let hits = fs.recall(
            "user:X",
            &[1.0, 0.0],
            1,
            2.0,
            letheo_core::entropy::DEFAULT_THETA_FADE,
        );
        assert_eq!(hits[0].text, "drives a red car");

        let n = save_facts(&dir, &fs).unwrap();
        assert_eq!(n, 2, "two distinct facts → two entries");

        let restored = load_facts(&dir).unwrap();
        assert_eq!(restored.len(), 2);

        // Lossless round-trip: verbatim text, embedding, provenance, and exact physics (including
        // reinforcement and the half-life consolidated by evocation).
        let orig: Vec<_> = fs.iter().collect();
        let back: Vec<_> = restored.iter().collect();
        for (a, b) in orig.iter().zip(back.iter()) {
            assert_eq!(a.text, b.text, "exact fact survives verbatim");
            assert_eq!(a.subject, b.subject);
            assert_eq!(a.provenance, b.provenance);
            assert_eq!(a.embedding, b.embedding);
            assert_eq!(a.created_at, b.created_at);
            assert_eq!(
                a.trace.lambda, b.trace.lambda,
                "consolidated half-life is preserved"
            );
            assert_eq!(a.trace.reinforcement, b.trace.reinforcement);
            assert_eq!(a.trace.last_touch, b.trace.last_touch);
        }

        // Rehydrated layer-1 still answers nominal: exact fact is retrieved after restart.
        let mut restored = restored;
        let after = restored.recall(
            "user:X",
            &[0.0, 1.0],
            1,
            3.0,
            letheo_core::entropy::DEFAULT_THETA_FADE,
        );
        assert_eq!(
            after[0].text, "allergic to peanuts",
            "fact survives restart and is evoked"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn both_layers_coexist_in_one_dir() {
        use letheo_core::FactStore;
        // Both layers coexist in the SAME directory (as the binding does on `save`).
        let dir = tmp_dir("both_layers");
        save_store(&dir, &sample_store()).unwrap();
        let mut facts = FactStore::new();
        facts.remember(
            "user:Xolotl",
            "allergic to peanuts",
            vec![0.0, 1.0],
            "agent",
            1.0,
            86_400.0,
            0.0,
        );
        save_facts(&dir, &facts).unwrap();

        // `load_store` must IGNORE `facts.json` (not parse it as an archetype); facts load separately.
        let restored = load_store(&dir).unwrap();
        assert_eq!(
            restored.len(),
            2,
            "archetypes load; facts.json is ignored"
        );
        assert_eq!(
            load_facts(&dir).unwrap().len(),
            1,
            "facts load independently"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    fn tmp_file(tag: &str) -> std::path::PathBuf {
        let mut p = env::temp_dir();
        p.push(format!("letheo_db_{tag}_{}.redb", std::process::id()));
        let _ = fs::remove_file(&p);
        p
    }

    #[test]
    fn db_roundtrips_both_layers_and_survives_reopen() {
        use letheo_core::FactStore;
        let path = tmp_file("roundtrip");
        {
            let db = DbStore::open(&path).unwrap();
            assert_eq!(db.write_store(&sample_store()).unwrap(), 2);
            let mut facts = FactStore::new();
            facts.remember(
                "user:Xolotl",
                "allergic to peanuts",
                vec![0.0, 1.0],
                "agent",
                1.0,
                86_400.0,
                0.0,
            );
            assert_eq!(db.write_facts(&facts).unwrap(), 1);
        } // store is dropped (releases the file lock)

        // Reopen: both memory layers survive — ACID durability.
        let db = DbStore::open(&path).unwrap();
        let store = db.read_store().unwrap();
        assert_eq!(store.len(), 2);
        assert_eq!(store.get("user:Xolotl").unwrap().represented, 1500);
        assert_eq!(db.read_facts().unwrap().len(), 1);
        drop(db);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn db_is_multi_tenant_per_subject() {
        let path = tmp_file("multitenant");
        let db = DbStore::open(&path).unwrap();
        db.write_store(&sample_store()).unwrap(); // user:Xolotl (1500) + agent:Tlaloc (42)

        // Updating ONLY one subject must not touch the other (key is the subject).
        let mut one = ArchetypeStore::new();
        one.imprint(
            &iv("agent:Tlaloc", vec![1.0, 0.0], 999),
            Resilience::High,
            0.0,
        );
        db.write_archetype(one.get("agent:Tlaloc").unwrap())
            .unwrap();

        let store = db.read_store().unwrap();
        assert_eq!(store.len(), 2, "both subjects still present");
        assert_eq!(
            store.get("agent:Tlaloc").unwrap().represented,
            999,
            "updated subject changed"
        );
        assert_eq!(
            store.get("user:Xolotl").unwrap().represented,
            1500,
            "other subject untouched"
        );
        drop(db);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn multimodal_archetype_survives_restart_with_physics() {
        use letheo_core::{CognitiveRuntime, Perception, RuntimeConfig};
        let dir = tmp_dir("multimodal");

        // A subject with three distinct behaviours: the engine distils three modes.
        let mut rt = CognitiveRuntime::new(RuntimeConfig::default());
        for _ in 0..10 {
            rt.perceive(
                Perception::new("u", vec![1.0, 0.0, 0.0], 1.0, 86_400.0, 0.0)
                    .with_trait("act", "noir"),
            );
            rt.perceive(
                Perception::new("u", vec![0.0, 1.0, 0.0], 1.0, 86_400.0, 0.0)
                    .with_trait("act", "docs"),
            );
            rt.perceive(
                Perception::new("u", vec![0.0, 0.0, 1.0], 1.0, 86_400.0, 0.0)
                    .with_trait("act", "scifi"),
            );
        }
        rt.breathe(&["u"], 0.0);
        let modes_before = rt.long_term().get("u").unwrap().modes.len();
        assert_eq!(modes_before, 3, "three behaviours → three modes");
        save_store(&dir, rt.long_term()).unwrap();
        drop(rt);

        // After "restart", modes and their physics (half-life, label) survive without loss.
        let restored = load_store(&dir).unwrap();
        let a = restored.get("u").unwrap();
        assert_eq!(a.modes.len(), 3, "modes survive restart");
        let labels: Vec<&str> = a.modes.iter().map(|m| m.label.as_str()).collect();
        assert!(labels.contains(&"noir") && labels.contains(&"docs") && labels.contains(&"scifi"));
        // Multi-modal resonance still works after rehydration.
        assert!(
            (a.resonance(&[1.0, 0.0, 0.0]) - 1.0).abs() < 1e-3,
            "correct mode resonates after restart"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn mode_origin_and_drift_survive_restart() {
        use letheo_core::{CognitiveRuntime, Perception, RuntimeConfig};
        let dir = tmp_dir("mode_drift");
        let mut rt = CognitiveRuntime::new(RuntimeConfig::default());
        // Cycle 1: behaviour at [1,0] (short half-life so it does not bleed into cycle 2).
        for _ in 0..5 {
            rt.perceive(Perception::new("u", vec![1.0, 0.0], 1.0, 1.0, 0.0).with_trait("act", "x"));
        }
        rt.breathe(&["u"], 0.0);
        // Cycle 2: SAME mode but shifted to [0.6,0.8] (cos 0.6 ≥ θ → merges) → mode drifts.
        for _ in 0..5 {
            rt.perceive(
                Perception::new("u", vec![0.6, 0.8], 1.0, 86_400.0, 0.0).with_trait("act", "x"),
            );
        }
        rt.breathe(&["u"], 100.0);
        let drift_before = rt.long_term().get("u").unwrap().modes[0].drift();
        assert!(
            drift_before > 0.0,
            "mode drifted from its origin: {drift_before}"
        );

        save_store(&dir, rt.long_term()).unwrap();
        drop(rt);
        let restored = load_store(&dir).unwrap();
        let drift_after = restored.get("u").unwrap().modes[0].drift();
        assert!(
            (drift_after - drift_before).abs() < 1e-6,
            "origin (and drift) survive restart"
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
