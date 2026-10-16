//! `ellm-lattice` — the JSON-lines side-car service over ewm-state-machine.
//!
//! Protocol (Phase 1):
//!
//! ```text
//! request  → {"id": "<n>", "cmd": "<name>", "args": {…}}     (one line, stdin)
//! response → {"id": "<n>", "ok": true, "data": {…}}          (one line, stdout)
//!          | {"id": "<n>", "ok": false, "error": "<msg>"}
//! ```
//!
//! Commands:
//! - `ping`        → `{"pong": true, "version": "…"}`
//! - `ingest`      → one [UM] turn: ingest → join → ring-push → commit → door
//!                   round-trip (`args: {"ids": [u32, …]}`)
//! - `materialize` → pure door demo, no commit (`args: {"ids": [u32, …]}`)
//! - `state`       → `LatticeState` snapshot of the live service
//! - `commit`      → force-commit the current working set (`args: {"message": "…"}`)
//! - `hllset_lut`  → the per-ewm-sm HLLSet registry (gate + confirmation)
//! - `setkey`      → Mode B scores of one query HLLSet vs candidate HLLSets
//! - `gate`        → per-host codebook gates (`op: put|list|get`)
//! - `to_prompt`   → materialize-to-prompt through a codebook gate (the door)
//! - `door_probe`  → door-guard verdict on a broadcast HLLSet (hex bytes)
//! - `setkey_hex`  → Mode B scores over HLLSet hex bytes
//! - `policy_put`  → publish the active policy into the store's named slot
//!                   `@policy` (the trainer's commit-as-publication channel)
//! - `policy_get`  → read the active policy (the server's hot-swap read)
//!
//! The service is generic over the object store: `Service<MemoryStore>` for
//! tests/demos, `Service<LooseStore>` when started as `ellm-lattice <repo>`
//! so two processes (server + trainer) share one persisted ewm-git store.

use std::collections::BTreeSet;
use std::io::{self, BufRead, Write};

use ewm_app::{StateCache, StateMachine};
use ewm_git::{view, LooseStore, MemoryStore, ObjectId, ObjectStore};
use hllset_contracts::token::{parse_token_id, token_in_bytes, TokenId};
use hllset_core::HLLSet;
use hllset_morphisms::{ingest, ingest_gated_set, materialize, Ingest};
use serde::Deserialize;
use serde_json::{json, Value};

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// One request line.
#[derive(Deserialize)]
struct Request {
    id: String,
    cmd: String,
    #[serde(default)]
    args: Value,
}

#[derive(Deserialize)]
struct IdsArgs {
    ids: Vec<TokenId>,
}

#[derive(Deserialize)]
struct CommitArgs {
    #[serde(default)]
    message: Option<String>,
}

#[derive(Deserialize)]
struct SetKeyArgs {
    query: Vec<String>,
    nodes: Vec<Vec<String>>,
}

#[derive(Deserialize)]
struct GatePutArgs {
    user: String,
    tokens: Vec<String>,
}

#[derive(Deserialize)]
struct GateGetArgs {
    user: String,
}

#[derive(Deserialize)]
struct ToPromptArgs {
    ids: Vec<TokenId>,
    #[serde(default)]
    user: Option<String>,
}

#[derive(Deserialize)]
struct DoorProbeArgs {
    /// HLLSet bytes as hex (the broadcast candidate).
    hllset: String,
    #[serde(default)]
    user: Option<String>,
}

#[derive(Deserialize)]
struct SetKeyHexArgs {
    /// Query HLLSet bytes as hex.
    query: String,
    /// Candidate HLLSet bytes as hex.
    nodes: Vec<String>,
}

#[derive(Deserialize)]
struct PolicyPutArgs {
    policy: Value,
}

/// The side-car service: the [UM] harness plus the shared cache.
///
/// Generic over the object store so one binary serves both the in-memory
/// demo (`Service<MemoryStore>`) and the two-process split
/// (`Service<LooseStore>`, sharing a persisted ewm-git store on disk).
struct Service<S: ObjectStore> {
    um: StateMachine<S>,
    cache: StateCache,
}

impl Service<MemoryStore> {
    fn new() -> Self {
        Self {
            um: StateMachine::new(MemoryStore::default()),
            cache: StateCache::empty(),
        }
    }
}

impl<S: ObjectStore> Service<S> {
    fn with_store(store: S) -> Self {
        Self {
            um: StateMachine::new(store),
            cache: StateCache::empty(),
        }
    }

    /// Recovery: open the persisted store and restore the cache from HEAD.
    fn open_store(store: S) -> Self {
        let um = StateMachine::open(store);
        let cache = StateCache::restore(um.repo());
        Self { um, cache }
    }

    fn handle(&mut self, line: &str) -> String {
        let request: Request = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => return error(String::new(), format!("parse: {e}")),
        };

        match request.cmd.as_str() {
            "ping" => ok(
                request.id,
                json!({ "pong": true, "version": VERSION, "cmd": "ping" }),
            ),
            "ingest" => self.handle_ingest(request.id, request.args),
            "materialize" => self.handle_materialize(request.id, request.args),
            "state" => self.handle_state(request.id),
            "commit" => self.handle_commit(request.id, request.args),
            "hllset_lut" => self.handle_hllset_lut(request.id),
            "gate" => self.handle_gate(request.id, request.args),
            "to_prompt" => self.handle_to_prompt(request.id, request.args),
            "door_probe" => self.handle_door_probe(request.id, request.args),
            "policy_put" => self.handle_policy_put(request.id, request.args),
            "policy_get" => self.handle_policy_get(request.id),
            "setkey" => handle_setkey(request.id, request.args),
            "setkey_hex" => handle_setkey_hex(request.id, request.args),
            other => error(request.id, format!("unknown command: {other}")),
        }
    }

    fn handle_ingest(&mut self, id: String, args: Value) -> String {
        let args: IdsArgs = match serde_json::from_value(args) {
            Ok(a) => a,
            Err(e) => return error(id, format!("ingest args: {e}")),
        };

        match self.um.run_turn(&mut self.cache, &args.ids) {
            Ok(outcome) => {
                let dnr = outcome
                    .commit_view
                    .as_ref()
                    .map(|v| {
                        json!({
                            "departed": v.departed.popcount(),
                            "retained": v.retained.popcount(),
                            "new": v.new.popcount(),
                        })
                    })
                    .unwrap_or(Value::Null);
                ok(
                    id,
                    json!({
                        "commit": outcome.commit.as_ref().map(|c| c.as_str()),
                        "head": outcome.head.as_ref().map(|c| c.as_str()),
                        "dnr": dnr,
                        "restored_ids": outcome.restored_ids,
                        "ring": {
                            "residual": outcome.ring_stats.residual,
                            "in_span": outcome.ring_stats.in_span,
                            "dimension": outcome.ring_stats.dimension,
                            "rotation_count": outcome.ring_stats.rotation_count,
                        },
                    }),
                )
            }
            Err(e) => error(id, format!("ingest: {e}")),
        }
    }

    fn handle_materialize(&mut self, id: String, args: Value) -> String {
        let args: IdsArgs = match serde_json::from_value(args) {
            Ok(a) => a,
            Err(e) => return error(id, format!("materialize args: {e}")),
        };

        let bytes: Vec<Vec<u8>> = args.ids.iter().map(|&n| token_in_bytes(n)).collect();
        let image = materialize(&ingest(&bytes));
        let restored_ids: Vec<TokenId> = image.iter().filter_map(|b| parse_token_id(b)).collect();

        ok(
            id,
            json!({
                "restored_ids": restored_ids,
                "restored_tokens": image.iter().map(|b| String::from_utf8_lossy(b).to_string()).collect::<Vec<_>>(),
            }),
        )
    }

    fn handle_state(&self, id: String) -> String {
        let head = self.um.head().cloned();
        let dnr = head
            .as_ref()
            .and_then(|cid| view(self.um.repo(), cid).ok())
            .map(|v| {
                json!({
                    "departed": v.departed.popcount(),
                    "retained": v.retained.popcount(),
                    "new": v.new.popcount(),
                })
            })
            .unwrap_or(Value::Null);

        let working_projection = self.cache.working[0]
            .union(&self.cache.working[1])
            .union(&self.cache.working[2]);

        ok(
            id,
            json!({
                "head": head.as_ref().map(|c| c.as_str()),
                "turn": self.cache.turn,
                "working_popcounts": [
                    self.cache.working[0].popcount(),
                    self.cache.working[1].popcount(),
                    self.cache.working[2].popcount(),
                ],
                "working_hex": hex::encode(working_projection.to_bytes()),
                "dnr": dnr,
                "ring": {
                    "dimension": self.cache.ring.dimension(),
                    "generation": self.cache.ring.generation(),
                },
                "births": self.cache.births.len(),
                "hllset_lut_len": self.um.repo().hllset_lut().len(),
            }),
        )
    }

    fn handle_commit(&mut self, id: String, args: Value) -> String {
        let args: CommitArgs = serde_json::from_value(args).unwrap_or(CommitArgs { message: None });
        let state = ewm_git::LatticeState {
            g1: self.cache.working[0].clone(),
            g2: self.cache.working[1].clone(),
            g3: self.cache.working[2].clone(),
            tf: self.cache.current_tf(),
        };
        let parents: Vec<ObjectId> = self.cache.tip.clone().into_iter().collect();
        let new_bits = self.um.repo().new_bits(&state);
        if new_bits == 0 {
            return ok(
                id,
                json!({ "commit": Value::Null, "skipped": true, "reason": "no new bits" }),
            );
        }
        let message = args
            .message
            .unwrap_or_else(|| format!("manual commit at turn {}", self.cache.turn));
        match self.um.repo_mut().commit(&state, &parents, &message) {
            Ok(commit) => {
                self.cache.tip = self.um.repo().head().cloned();
                ok(
                    id,
                    json!({ "commit": commit.as_str(), "skipped": false }),
                )
            }
            Err(e) => error(id, format!("commit: {e}")),
        }
    }

    fn handle_hllset_lut(&self, id: String) -> String {
        let entries: Vec<Value> = self
            .um
            .repo()
            .hllset_lut()
            .ranked()
            .into_iter()
            .map(|(cid, th)| json!({ "cid": cid.as_str(), "th": th }))
            .collect();
        ok(id, json!({ "entries": entries }))
    }

    /// `gate put {user, tokens}` / `gate list` / `gate get {user}` — the
    /// per-host codebook gates (host encodings as languages).
    fn handle_gate(&mut self, id: String, args: Value) -> String {
        let request: Value = match serde_json::from_value(args) {
            Ok(v) => v,
            Err(e) => return error(id, format!("gate args: {e}")),
        };

        match request.get("op").and_then(|v| v.as_str()) {
            Some("put") => {
                let args: GatePutArgs = match serde_json::from_value(request) {
                    Ok(a) => a,
                    Err(e) => return error(id, format!("gate put args: {e}")),
                };
                match self.um.repo_mut().put_gate(&args.user, &args.tokens) {
                    Ok(catalog) => ok(
                        id,
                        json!({ "user": args.user, "catalog_key": catalog.as_str(), "tokens": args.tokens.len() }),
                    ),
                    Err(e) => error(id, format!("gate put: {e}")),
                }
            }
            Some("list") => ok(
                id,
                json!({ "gates": self.um.repo().gate_names() }),
            ),
            Some("get") => {
                let args: GateGetArgs = match serde_json::from_value(request) {
                    Ok(a) => a,
                    Err(e) => return error(id, format!("gate get args: {e}")),
                };
                match self.um.repo().gate(&args.user) {
                    Ok((catalog_key, tokens)) => ok(
                        id,
                        json!({ "user": args.user, "catalog_key": catalog_key, "tokens": tokens }),
                    ),
                    Err(e) => error(id, format!("gate get: {e}")),
                }
            }
            other => error(id, format!("gate: unknown op {other:?} (use put|list|get)")),
        }
    }

    /// `to_prompt {ids, user?}` — materialize-to-prompt through the named
    /// codebook gate, guarded by the round-trip coverage invariant.
    ///
    /// With `user` set, the query HLLSet is gated by that user's codebook:
    /// `coverage = |query ∩ gate| / |query|` must be 1.0, otherwise the door
    /// is refused (`RoundTripGuard`). Without `user`, the plain door runs.
    fn handle_to_prompt(&mut self, id: String, args: Value) -> String {
        let args: ToPromptArgs = match serde_json::from_value(args) {
            Ok(a) => a,
            Err(e) => return error(id, format!("to_prompt args: {e}")),
        };

        let bytes: Vec<Vec<u8>> = args.ids.iter().map(|&n| token_in_bytes(n)).collect();

        match args.user {
            Some(user) => {
                let (catalog_key, codebook) = match self.um.repo().gate(&user) {
                    Ok(g) => g,
                    Err(e) => return error(id, format!("to_prompt gate: {e}")),
                };
                let cb_bytes: Vec<Vec<u8>> =
                    codebook.iter().map(|t| t.as_bytes().to_vec()).collect();

                // The codebook gate uses the **flat n-seed** scheme (per-token
                // atoms, seeds 0/1/2) — the same scheme `put_gate` uses for its
                // catalog key. A vocabulary gate covers a query exactly when
                // every query token's atoms are in the gate.
                let mut query_ingest = Ingest::new();
                query_ingest.ingest_tokens(bytes.iter().map(|b| b.as_slice()));
                let mut gate_ingest = Ingest::new();
                gate_ingest.ingest_tokens(cb_bytes.iter().map(|b| b.as_slice()));
                let gate_set = gate_ingest.projection();

                let coverage = coverage_of(&query_ingest.projection(), &gate_set);
                if coverage < 1.0 {
                    return ok(
                        id,
                        json!({
                            "refused": true,
                            "coverage": coverage,
                            "gate": user,
                            "reason": "round-trip coverage < 1.0 (RoundTripGuard)",
                        }),
                    );
                }

                let cb_set: BTreeSet<Vec<u8>> = cb_bytes.iter().cloned().collect();
                let tokens = ingest_gated_set(&query_ingest, &gate_set, Some(&cb_set));
                let restored_ids: Vec<TokenId> =
                    tokens.iter().filter_map(|b| parse_token_id(b)).collect();
                ok(
                    id,
                    json!({
                        "refused": false,
                        "coverage": 1.0,
                        "gate": user,
                        "catalog_key": catalog_key,
                        "restored_ids": restored_ids,
                        "tokens": tokens.iter().map(|b| String::from_utf8_lossy(b).to_string()).collect::<Vec<_>>(),
                    }),
                )
            }
            None => {
                let ingested = ingest(&bytes);
                let tokens = materialize(&ingested);
                let restored_ids: Vec<TokenId> =
                    tokens.iter().filter_map(|b| parse_token_id(b)).collect();
                ok(
                    id,
                    json!({
                        "refused": false,
                        "coverage": 1.0,
                        "gate": Value::Null,
                        "restored_ids": restored_ids,
                        "tokens": tokens.iter().map(|b| String::from_utf8_lossy(b).to_string()).collect::<Vec<_>>(),
                    }),
                )
            }
        }
    }

    /// `door_probe {hllset: hex, user?}` — the door guard's verdict on a
    /// broadcast HLLSet candidate. Probes the named codebook gate, or every
    /// registered gate when `user` is omitted.
    fn handle_door_probe(&self, id: String, args: Value) -> String {
        let args: DoorProbeArgs = match serde_json::from_value(args) {
            Ok(a) => a,
            Err(e) => return error(id, format!("door_probe args: {e}")),
        };
        let hllset = match decode_hllset(&args.hllset) {
            Ok(h) => h,
            Err(e) => return error(id, format!("door_probe hllset: {e}")),
        };

        let mut verdicts: Vec<Value> = Vec::new();
        match args.user {
            Some(user) => match self.um.repo().gate(&user) {
                Ok((_, codebook)) => {
                    let gate_set = n_seed_projection(&codebook);
                    let coverage = coverage_of(&hllset, &gate_set);
                    verdicts.push(verdict_json(&user, coverage));
                }
                Err(e) => return error(id, format!("door_probe gate: {e}")),
            },
            None => {
                for user in self.um.repo().gate_names() {
                    if let Ok((_, codebook)) = self.um.repo().gate(&user) {
                        let gate_set = n_seed_projection(&codebook);
                        let coverage = coverage_of(&hllset, &gate_set);
                        verdicts.push(verdict_json(&user, coverage));
                    }
                }
            }
        }

        ok(id, json!({ "verdicts": verdicts }))
    }

    /// `policy_put {policy}` — the trainer publishes the active policy into
    /// the store's named slot `@policy`. This is commit-as-publication: the
    /// server never sees the policy until it is written here.
    fn handle_policy_put(&mut self, id: String, args: Value) -> String {
        let args: PolicyPutArgs = match serde_json::from_value(args) {
            Ok(a) => a,
            Err(e) => return error(id, format!("policy_put args: {e}")),
        };
        let text = serde_json::to_string(&args.policy).unwrap_or_else(|_| "null".to_string());
        match self.um.repo_mut().store_mut().put_gate("@policy", &text) {
            Ok(()) => ok(id, json!({ "published": true, "slot": "@policy" })),
            Err(e) => error(id, format!("policy_put: {e}")),
        }
    }

    /// `policy_get` — the server's hot-swap read of the active policy.
    fn handle_policy_get(&self, id: String) -> String {
        match self.um.repo().store().get_gate("@policy") {
            Ok(text) => match serde_json::from_str::<Value>(&text) {
                Ok(policy) => ok(id, json!({ "policy": policy })),
                Err(_) => ok(id, json!({ "policy": Value::Null, "corrupt": true })),
            },
            Err(_) => ok(id, json!({ "policy": Value::Null })),
        }
    }
}

/// Fraction of the query's set bits covered by the gate.
fn coverage_of(query: &HLLSet, gate_set: &HLLSet) -> f64 {
    let total = query.popcount() as f64;
    if total == 0.0 {
        1.0
    } else {
        query.intersection(gate_set).popcount() as f64 / total
    }
}

/// Decode an HLLSet from its hex serialization (Roaring bitmap bytes).
fn decode_hllset(hex_bytes: &str) -> Result<HLLSet, String> {
    let bytes = hex::decode(hex_bytes).map_err(|e| format!("bad hex: {e}"))?;
    HLLSet::from_bytes(&bytes).ok_or_else(|| "bad hllset bytes".to_string())
}

/// The flat n-seed projection of a codebook — the gate HLLSet.
fn n_seed_projection(codebook: &[String]) -> HLLSet {
    let mut ingest = Ingest::new();
    let bytes: Vec<Vec<u8>> = codebook.iter().map(|t| t.as_bytes().to_vec()).collect();
    ingest.ingest_tokens(bytes.iter().map(|b| b.as_slice()));
    ingest.projection()
}

/// One door verdict: `{user, coverage, can_restore}`.
fn verdict_json(user: &str, coverage: f64) -> Value {
    json!({
        "user": user,
        "coverage": coverage,
        "can_restore": coverage >= 1.0,
    })
}

/// Mode B scores over raw HLLSet bytes (hex) — the discovery scorer.
fn handle_setkey_hex(id: String, args: Value) -> String {
    let args: SetKeyHexArgs = match serde_json::from_value(args) {
        Ok(a) => a,
        Err(e) => return error(id, format!("setkey_hex args: {e}")),
    };
    let query = match decode_hllset(&args.query) {
        Ok(q) => q,
        Err(e) => return error(id, format!("setkey_hex query: {e}")),
    };
    let nodes: Vec<HLLSet> = match args.nodes.iter().map(|n| decode_hllset(n)).collect() {
        Ok(n) => n,
        Err(e) => return error(id, format!("setkey_hex node: {e}")),
    };

    let scores = ellm_attn::setkey::score_many(&query, &nodes);
    ok(
        id,
        json!({
            "query_bits": query.popcount(),
            "scores": scores,
        }),
    )
}

fn handle_setkey(id: String, args: Value) -> String {
    let args: SetKeyArgs = match serde_json::from_value(args) {
        Ok(a) => a,
        Err(e) => return error(id, format!("setkey args: {e}")),
    };

    let query = HLLSet::from_tokens(&args.query);
    let contexts: Vec<HLLSet> = args
        .nodes
        .iter()
        .map(|node| HLLSet::from_tokens(node))
        .collect();

    let scores = ellm_attn::setkey::score_many(&query, &contexts);
    ok(
        id,
        json!({
            "query_bits": query.popcount(),
            "scores": scores,
        }),
    )
}

fn ok(id: String, data: Value) -> String {
    json!({ "id": id, "ok": true, "data": data }).to_string()
}

fn error(id: String, message: String) -> String {
    json!({ "id": id, "ok": false, "error": message }).to_string()
}

fn main() {
    let repo_path = std::env::args().nth(1);
    match repo_path {
        Some(path) => run(Service::open_store(LooseStore::new(path))),
        None => run(Service::new()),
    }
}

fn run<S: ObjectStore>(mut service: Service<S>) {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let response = service.handle(&line);
        if writeln!(out, "{response}").is_err() {
            break;
        }
        let _ = out.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call<S: ObjectStore>(service: &mut Service<S>, cmd: &str, args: Value) -> Value {
        let line = json!({ "id": "t", "cmd": cmd, "args": args }).to_string();
        serde_json::from_str(&service.handle(&line)).unwrap()
    }

    #[test]
    fn ping_returns_pong() {
        let mut service = Service::new();
        let response = call(&mut service, "ping", json!({}));
        assert_eq!(response["ok"], true);
        assert_eq!(response["data"]["pong"], true);
    }

    #[test]
    fn ingest_turn_commits_and_restores_ids() {
        let mut service = Service::new();
        let response = call(&mut service, "ingest", json!({ "ids": [1, 2, 3] }));
        assert_eq!(response["ok"], true);
        assert!(response["data"]["commit"].is_string());
        assert_eq!(response["data"]["restored_ids"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn state_reports_working_set_and_ring() {
        let mut service = Service::new();
        call(&mut service, "ingest", json!({ "ids": [1, 2, 3] }));
        let response = call(&mut service, "state", json!({}));
        assert_eq!(response["ok"], true);
        assert_eq!(response["data"]["turn"], 1);
        assert!(response["data"]["working_popcounts"][0].as_u64().unwrap() > 0);
        assert!(response["data"]["ring"]["dimension"].as_u64().unwrap() >= 1);
    }

    #[test]
    fn materialize_roundtrips_without_committing() {
        let mut service = Service::new();
        let response = call(&mut service, "materialize", json!({ "ids": [7, 8] }));
        assert_eq!(response["ok"], true);
        assert_eq!(response["data"]["restored_ids"].as_array().unwrap().len(), 2);
        let state = call(&mut service, "state", json!({}));
        assert_eq!(state["data"]["turn"], 0, "materialize must not commit");
    }

    #[test]
    fn hllset_lut_lists_registered_hllsets() {
        let mut service = Service::new();
        call(&mut service, "ingest", json!({ "ids": [1, 2] }));
        let response = call(&mut service, "hllset_lut", json!({}));
        assert_eq!(response["ok"], true);
        assert!(response["data"]["entries"].as_array().unwrap().len() >= 1);
    }

    #[test]
    fn setkey_returns_bss_scores() {
        let mut service = Service::new();
        let response = call(
            &mut service,
            "setkey",
            json!({ "query": ["alpha"], "nodes": [["alpha", "beta"], ["gamma"]] }),
        );
        assert_eq!(response["ok"], true);
        let scores = response["data"]["scores"].as_array().unwrap();
        assert!((scores[0].as_f64().unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn gate_put_and_list_register_codebook() {
        let mut service = Service::new();
        let put = call(
            &mut service,
            "gate",
            json!({ "op": "put", "user": "ellm", "tokens": ["tid1", "tid2", "tid3"] }),
        );
        assert_eq!(put["ok"], true);
        assert!(put["data"]["catalog_key"].is_string());

        let list = call(&mut service, "gate", json!({ "op": "list" }));
        assert!(list["data"]["gates"].as_array().unwrap().contains(&json!("ellm")));

        let get = call(&mut service, "gate", json!({ "op": "get", "user": "ellm" }));
        assert_eq!(get["ok"], true);
        assert_eq!(get["data"]["tokens"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn to_prompt_gated_covers_in_codebook_tokens() {
        let mut service = Service::new();
        call(
            &mut service,
            "gate",
            json!({ "op": "put", "user": "ellm", "tokens": ["tid1", "tid2", "tid3"] }),
        );
        let response = call(&mut service, "to_prompt", json!({ "ids": [1, 2], "user": "ellm" }));
        assert_eq!(response["ok"], true);
        assert_eq!(response["data"]["refused"], false);
        assert!((response["data"]["coverage"].as_f64().unwrap() - 1.0).abs() < 1e-9);
        assert_eq!(response["data"]["restored_ids"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn to_prompt_refuses_unknown_tokens_outside_gate() {
        let mut service = Service::new();
        call(
            &mut service,
            "gate",
            json!({ "op": "put", "user": "ellm", "tokens": ["tid1", "tid2", "tid3"] }),
        );
        let response = call(
            &mut service,
            "to_prompt",
            json!({ "ids": [1, 99], "user": "ellm" }),
        );
        assert_eq!(response["ok"], true);
        assert_eq!(response["data"]["refused"], true);
        assert!(response["data"]["coverage"].as_f64().unwrap() < 1.0);
    }

    #[test]
    fn to_prompt_ungated_plain_door() {
        let mut service = Service::new();
        let response = call(&mut service, "to_prompt", json!({ "ids": [7, 8] }));
        assert_eq!(response["ok"], true);
        assert_eq!(response["data"]["refused"], false);
        assert_eq!(response["data"]["gate"], Value::Null);
        assert_eq!(response["data"]["restored_ids"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn door_probe_accepts_candidate_inside_gate() {
        let mut service = Service::new();
        call(
            &mut service,
            "gate",
            json!({ "op": "put", "user": "ellm", "tokens": ["tid1", "tid2", "tid3"] }),
        );
        let candidate = HLLSet::from_tokens(&["tid1".as_bytes(), "tid2".as_bytes()]);
        let response = call(
            &mut service,
            "door_probe",
            json!({ "hllset": hex::encode(candidate.to_bytes()), "user": "ellm" }),
        );
        assert_eq!(response["ok"], true);
        let verdicts = response["data"]["verdicts"].as_array().unwrap();
        assert_eq!(verdicts.len(), 1);
        assert_eq!(verdicts[0]["can_restore"], true);
    }

    #[test]
    fn door_probe_refuses_candidate_outside_gate() {
        let mut service = Service::new();
        call(
            &mut service,
            "gate",
            json!({ "op": "put", "user": "ellm", "tokens": ["tid1", "tid2"] }),
        );
        let candidate = HLLSet::from_tokens(&["tid1".as_bytes(), "tid99".as_bytes()]);
        let response = call(
            &mut service,
            "door_probe",
            json!({ "hllset": hex::encode(candidate.to_bytes()) }),
        );
        let verdicts = response["data"]["verdicts"].as_array().unwrap();
        assert_eq!(verdicts[0]["can_restore"], false);
    }

    #[test]
    fn door_probe_answers_all_gates_when_user_omitted() {
        let mut service = Service::new();
        call(
            &mut service,
            "gate",
            json!({ "op": "put", "user": "ellm", "tokens": ["tid1", "tid2"] }),
        );
        call(
            &mut service,
            "gate",
            json!({ "op": "put", "user": "host_b", "tokens": ["tid1", "tid2"] }),
        );
        let candidate = HLLSet::from_tokens(&["tid1".as_bytes()]);
        let response = call(
            &mut service,
            "door_probe",
            json!({ "hllset": hex::encode(candidate.to_bytes()) }),
        );
        let verdicts = response["data"]["verdicts"].as_array().unwrap();
        assert_eq!(verdicts.len(), 2);
        assert!(verdicts.iter().all(|v| v["can_restore"] == true));
    }

    #[test]
    fn setkey_hex_scores_hllset_bytes() {
        let mut service = Service::new();
        let query = HLLSet::from_tokens(&["tid1".as_bytes()]);
        let node = HLLSet::from_tokens(&["tid1".as_bytes(), "tid2".as_bytes()]);
        let response = call(
            &mut service,
            "setkey_hex",
            json!({
                "query": hex::encode(query.to_bytes()),
                "nodes": [hex::encode(node.to_bytes())],
            }),
        );
        assert_eq!(response["ok"], true);
        let scores = response["data"]["scores"].as_array().unwrap();
        assert!((scores[0].as_f64().unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn state_exposes_working_hex() {
        let mut service = Service::new();
        call(&mut service, "ingest", json!({ "ids": [1, 2] }));
        let response = call(&mut service, "state", json!({}));
        let hex_str = response["data"]["working_hex"].as_str().unwrap();
        let parsed = decode_hllset(hex_str).expect("working_hex must round-trip");
        assert!(parsed.popcount() > 0);
    }

    #[test]
    fn policy_put_and_get_roundtrip() {
        let mut service = Service::new();
        let put = call(
            &mut service,
            "policy_put",
            json!({ "policy": { "theta": 0.4, "door": "surprise-v0" } }),
        );
        assert_eq!(put["data"]["published"], true);

        let get = call(&mut service, "policy_get", json!({}));
        assert_eq!(get["data"]["policy"]["theta"], 0.4);
        assert_eq!(get["data"]["policy"]["door"], "surprise-v0");
    }

    #[test]
    fn loose_store_is_shared_between_two_services() {
        let dir = std::env::temp_dir().join(format!("ellm-lattice-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let mut writer = Service::with_store(LooseStore::new(&dir));
        call(
            &mut writer,
            "gate",
            json!({ "op": "put", "user": "ellm", "tokens": ["tid1", "tid2"] }),
        );
        call(
            &mut writer,
            "policy_put",
            json!({ "policy": { "generation": 7 } }),
        );

        // A second process's side-car opens the same store and sees both.
        let mut reader = Service::open_store(LooseStore::new(&dir));
        let gates = call(&mut reader, "gate", json!({ "op": "list" }));
        assert!(gates["data"]["gates"].as_array().unwrap().contains(&json!("ellm")));
        let policy = call(&mut reader, "policy_get", json!({}));
        assert_eq!(policy["data"]["policy"]["generation"], 7);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unknown_command_is_an_error() {
        let mut service = Service::new();
        let response = call(&mut service, "nope", json!({}));
        assert_eq!(response["ok"], false);
    }
}
