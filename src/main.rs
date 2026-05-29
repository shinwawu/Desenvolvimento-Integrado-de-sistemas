use std::collections::HashMap;
use std::fs::File;
use std::sync::{Arc, Mutex};

use ndarray::{Array1, ArrayView1};
use ndarray_npy::NpzReader;
use zip::ZipArchive;
use poem::{
    handler,
    http::StatusCode,
    listener::TcpListener,
    post,
    web::{Data, Json, Path, Query},
    EndpointExt, IntoResponse, Response, Route, Server,
};
use serde::Deserialize;
use serde_json::json;
use sprs::CsMat;
use tokio::sync::Semaphore;

type F = f32;

const MAX_ITER: usize = 10;
const TOL: F = 1e-4;
const RECON_TIMEOUT_S: u64 = 30;

struct Model {
    h: CsMat<F>,
    ht: CsMat<F>, // precomputed CSR transpose for fast matvec
    s: usize,
    n: usize,
    shape: (usize, usize),
}

#[derive(Default)]
struct ClientBuffer {
    algorithm: String,
    g_parts: Vec<F>,
}

struct AppState {
    models: HashMap<String, Arc<Model>>,
    client_signals: Mutex<HashMap<String, ClientBuffer>>,
    inflight_sem: Arc<Semaphore>,
}

struct ModelCfg {
    s: usize,
    n: usize,
    shape: (usize, usize),
    path: &'static str,
}

fn model_configs() -> Vec<(&'static str, ModelCfg)> {
    vec![
        (
            "60x60",
            ModelCfg { s: 50816, n: 3600, shape: (60, 60), path: "data/H-1.npz" },
        ),
        (
            "30x30",
            ModelCfg { s: 27904, n: 900, shape: (30, 30), path: "data/H-2.npz" },
        ),
    ]
}

/// Parse the raw bytes of a 0-d `|S3` npy file (scipy sparse format field) into a string.
fn parse_npy_s3_string(npy: &[u8]) -> Result<String, Box<dyn std::error::Error>> {
    if npy.len() < 10 || &npy[0..6] != b"\x93NUMPY" {
        return Err("not a valid npy file".into());
    }
    let major = npy[6];
    let data_offset = if major == 1 {
        let hl = u16::from_le_bytes([npy[8], npy[9]]) as usize;
        10 + hl
    } else {
        let hl = u32::from_le_bytes([npy[8], npy[9], npy[10], npy[11]]) as usize;
        12 + hl
    };
    let data = npy.get(data_offset..).ok_or("npy file truncated")?;
    let s = std::str::from_utf8(data)?.trim_end_matches('\0').trim().to_string();
    Ok(s)
}

/// Load a scipy.sparse .npz file as a CSR matrix of f32.
fn load_scipy_sparse_npz(path: &str) -> Result<CsMat<F>, Box<dyn std::error::Error>> {
    use std::io::{Read, Seek, SeekFrom};

    let file = File::open(path)?;
    let mut archive = ZipArchive::new(file)?;
    let fmt = {
        let mut entry = archive.by_name("format.npy")?;
        let mut buf = Vec::new();
        entry.read_to_end(&mut buf)?;
        parse_npy_s3_string(&buf)?
    };

    let mut file = archive.into_inner();
    file.seek(SeekFrom::Start(0))?;
    let mut npz = NpzReader::new(file)?;

    let shape_arr: Array1<i64> = npz.by_name("shape.npy")?;
    // tenta f32 primeiro (formato do matrix_converter atual); cai para f64 se falhar
    let data: Vec<F> = {
        let try_f32: Result<Array1<f32>, _> = npz.by_name("data.npy");
        match try_f32 {
            Ok(arr) => arr.iter().copied().collect(),
            Err(_) => {
                let arr: Array1<f64> = npz.by_name("data.npy")?;
                arr.iter().map(|&x| x as F).collect()
            }
        }
    };
    let indices_arr: Array1<i32> = npz.by_name("indices.npy")?;
    let indptr_arr: Array1<i32> = npz.by_name("indptr.npy")?;

    let nrows = shape_arr[0] as usize;
    let ncols = shape_arr[1] as usize;

    let indices: Vec<usize> = indices_arr.iter().map(|&x| x as usize).collect();
    let indptr: Vec<usize> = indptr_arr.iter().map(|&x| x as usize).collect();

    let mat = if fmt.starts_with("csr") {
        CsMat::new((nrows, ncols), indptr, indices, data)
    } else if fmt.starts_with("csc") {
        CsMat::new_csc((nrows, ncols), indptr, indices, data).to_other_storage()
    } else {
        return Err(format!("unsupported sparse format: {}", fmt).into());
    };

    Ok(mat)
}

fn load_model(name: &str, cfg: &ModelCfg) -> Result<Model, Box<dyn std::error::Error>> {
    let h = load_scipy_sparse_npz(cfg.path)?;
    if h.shape() != (cfg.s, cfg.n) {
        return Err(
            format!("{}: shape {:?}, expected ({}, {})", name, h.shape(), cfg.s, cfg.n).into(),
        );
    }
    let ht = h.transpose_view().to_other_storage();
    println!("loaded {}: nnz={}, transpose nnz={}", name, h.nnz(), ht.nnz());
    Ok(Model { h, ht, s: cfg.s, n: cfg.n, shape: cfg.shape })
}

/// CGNR: same algorithm as the Python server.
fn cgnr(model: &Model, g: ArrayView1<F>, max_iter: usize, tol: F) -> (Array1<F>, usize, F) {
    let n = model.n;
    let tol_sq = tol * tol;
    let mut f = Array1::<F>::zeros(n);
    let mut r = g.to_owned();
    let mut z: Array1<F> = &model.ht * &r;
    let mut p = z.clone();
    let mut norm_z_sq = z.dot(&z);

    for k in 0..max_iter {
        let w: Array1<F> = &model.h * &p;
        let norm_w_sq = w.dot(&w);
        if norm_w_sq == 0.0 {
            return (f, k, r.dot(&r).sqrt());
        }
        let alpha = norm_z_sq / norm_w_sq;
        f.scaled_add(alpha, &p);
        r.scaled_add(-alpha, &w);

        let norm_r_sq = r.dot(&r);
        if norm_r_sq < tol_sq {
            return (f, k + 1, norm_r_sq.sqrt());
        }
        z = &model.ht * &r;
        let norm_z_new_sq = z.dot(&z);
        if norm_z_sq == 0.0 {
            return (f, k + 1, norm_r_sq.sqrt());
        }
        let beta = norm_z_new_sq / norm_z_sq;
        p *= beta;
        p += &z;
        norm_z_sq = norm_z_new_sq;
    }
    (f, max_iter, r.dot(&r).sqrt())
}

/// CGNE: matches the Python server implementation.
fn cgne(model: &Model, g: ArrayView1<F>, max_iter: usize, tol: F) -> (Array1<F>, usize, F) {
    let n = model.n;
    let tol_sq = tol * tol;
    let mut f = Array1::<F>::zeros(n);
    let mut r = g.to_owned();              // r = g - H*0 = g
    let mut p: Array1<F> = &model.ht * &r;
    let mut rtr = r.dot(&r);

    for k in 0..max_iter {
        let hp: Array1<F> = &model.h * &p;
        let ptp = p.dot(&p);
        if ptp == 0.0 {
            return (f, k, rtr.sqrt());
        }
        let alpha = rtr / ptp;
        f.scaled_add(alpha, &p);
        r.scaled_add(-alpha, &hp);
        let new_rtr = r.dot(&r);
        if new_rtr < tol_sq {
            return (f, k + 1, new_rtr.sqrt());
        }
        let beta = new_rtr / rtr;
        let htr: Array1<F> = &model.ht * &r;
        p *= beta;
        p += &htr;
        rtr = new_rtr;
    }
    (f, max_iter, rtr.sqrt())
}

/// Min-max normalize in place. Matches Python (img - min) / (max - min).
fn minmax_normalize(v: &mut [F]) {
    let (mut lo, mut hi) = (F::INFINITY, F::NEG_INFINITY);
    for &x in v.iter() {
        if x < lo { lo = x; }
        if x > hi { hi = x; }
    }
    let span = hi - lo;
    if span > 0.0 {
        for x in v.iter_mut() { *x = (*x - lo) / span; }
    } else {
        for x in v.iter_mut() { *x = 0.0; }
    }
}

/// Run reconstruction by algorithm name. Returns (f_normalized, iters, err).
fn run_reconstruction(
    model: &Model,
    algorithm: &str,
    g: Array1<F>,
) -> Result<(Array1<F>, usize, F), String> {
    let (mut f, iters, err) = match algorithm {
        "CGNR" => cgnr(model, g.view(), MAX_ITER, TOL),
        "CGNE" => cgne(model, g.view(), MAX_ITER, TOL),
        other => return Err(format!("algoritmo '{}' nao suportado", other)),
    };
    minmax_normalize(f.as_slice_mut().unwrap());
    Ok((f, iters, err))
}

// ---------- HTTP API (chunked protocol, mirrors the Python server) ----------

#[derive(Deserialize)]
struct ReconstructQuery {
    cliente_id: String,
    algorithm: String,
    model_id: String,
    #[serde(default, deserialize_with = "deserialize_bool_loose")]
    complete: bool,
}

// Python's `requests` envia bool como "True"/"False" (com inicial maiuscula);
// o default do serde so aceita "true"/"false". Aqui aceitamos ambos.
fn deserialize_bool_loose<'de, D>(deserializer: D) -> Result<bool, D::Error>
where D: serde::Deserializer<'de> {
    use serde::Deserialize;
    let s: String = String::deserialize(deserializer)?;
    match s.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" | "" => Ok(false),
        other => Err(serde::de::Error::custom(format!("bool invalido: {}", other))),
    }
}

#[derive(Deserialize)]
struct Sinal {
    g: Vec<F>,
}

fn iso_now() -> String {
    chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%.3f").to_string()
}

#[handler]
async fn reconstruct(
    Path(model_id_in_path): Path<String>,
    Query(params): Query<ReconstructQuery>,
    Data(state): Data<&Arc<AppState>>,
    Json(sinal): Json<Sinal>,
) -> Response {
    // accumula chunks no buffer do cliente; so executa quando complete=true.
    let (g_arr, algo_choice) = {
        let mut signals = state.client_signals.lock().unwrap();
        let buf = signals.entry(params.cliente_id.clone()).or_insert_with(|| ClientBuffer {
            algorithm: params.algorithm.clone(),
            g_parts: Vec::new(),
        });
        buf.g_parts.extend_from_slice(&sinal.g);

        if !params.complete {
            return Json(json!({
                "message": format!("sinal g recebido para {}, aguardando mais partes...", model_id_in_path)
            })).into_response();
        }

        // consome o buffer e libera o lock antes da reconstrucao
        let algo = buf.algorithm.clone();
        let g_taken = std::mem::take(&mut buf.g_parts);
        (Array1::from(g_taken), algo)
    };

    let model = match state.models.get(&model_id_in_path) {
        Some(m) => m.clone(),
        None => {
            return Json(json!({
                "error": format!("modelo '{}' nao encontrado. modelos disponiveis: {:?}",
                    model_id_in_path, state.models.keys().collect::<Vec<_>>())
            })).into_response();
        }
    };

    if g_arr.len() != model.s {
        return Json(json!({
            "error": format!("tamanho do sinal g={} diferente do esperado {} para o modelo {}",
                g_arr.len(), model.s, model_id_in_path)
        })).into_response();
    }

    // admission control: limita reconstrucoes simultaneas a 2*cpu (mesma logica do Python).
    let _permit = state.inflight_sem.clone().acquire_owned().await.unwrap();

    let start_dt = iso_now();
    let t0 = std::time::Instant::now();

    // clona o Arc para que possamos acessar model.shape apos o spawn_blocking
    let model_for_task = model.clone();
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(RECON_TIMEOUT_S),
        tokio::task::spawn_blocking(move || run_reconstruction(&model_for_task, &algo_choice, g_arr)),
    ).await;

    let elapsed = t0.elapsed().as_secs_f64();
    let end_dt = iso_now();

    let (f, iters, err) = match result {
        Ok(Ok(Ok(triple))) => triple,
        Ok(Ok(Err(msg))) => {
            return Json(json!({ "error": format!("erro na reconstrucao: {}", msg) })).into_response();
        }
        Ok(Err(_)) => {
            return (StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": "task panicked" }))).into_response();
        }
        Err(_) => {
            return (StatusCode::GATEWAY_TIMEOUT,
                    Json(json!({ "error": format!("timeout > {}s na reconstrucao", RECON_TIMEOUT_S) }))).into_response();
        }
    };

    println!("reconstrucao {} completa: iters={}, erro final={:.6}", model_id_in_path, iters, err);

    // reshape com ordem 'F' (coluna principal), igual ao Python.
    let (rows, cols) = model.shape;
    let mut img = vec![vec![0.0f32; cols]; rows];
    for j in 0..cols {
        for i in 0..rows {
            img[i][j] = f[j * rows + i];
        }
    }

    Json(json!({
        "message": format!("reconstrucao completa para {}", model_id_in_path),
        "image": img,
        "iters": iters,
        "final_error": err,
        "reconstruction_time": elapsed,
        "start_time": start_dt,
        "end_time": end_dt,
    })).into_response()
}

#[handler]
async fn health(Data(state): Data<&Arc<AppState>>) -> Json<serde_json::Value> {
    Json(json!({"status": "ok", "models": state.models.keys().collect::<Vec<_>>()}))
}

/// Monitor de memoria: imprime uso/disponibilidade do sistema a cada 1s.
fn spawn_memory_monitor() {
    tokio::spawn(async move {
        use sysinfo::{System, Pid, ProcessesToUpdate};
        let mut sys = System::new_all();
        let pid_self = Pid::from_u32(std::process::id());
        const GB: f64 = 1024.0 * 1024.0 * 1024.0;
        loop {
            sys.refresh_memory();
            sys.refresh_processes(ProcessesToUpdate::Some(&[pid_self]), true);
            let rss_gb = sys.process(pid_self).map(|p| p.memory()).unwrap_or(0) as f64 / GB;
            let total = sys.total_memory() as f64 / GB;
            let used = sys.used_memory() as f64 / GB;
            let avail = sys.available_memory() as f64 / GB;
            let pct = (used / total) * 100.0;
            let now = chrono::Local::now().format("%H:%M:%S");
            println!(
                "[memoria {}] sistema usada={:.2}GB disponivel={:.2}GB / total={:.2}GB ({:.1}%) | uso da app ={:.2}GB",
                now, used, avail, total, pct, rss_gb
            );
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        }
    });
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), std::io::Error> {
    let mut models = HashMap::new();
    for (name, cfg) in model_configs() {
        match load_model(name, &cfg) {
            Ok(m) => { models.insert(name.to_string(), Arc::new(m)); }
            Err(e) => eprintln!("failed to load {}: {}", name, e),
        }
    }

    let max_inflight = (num_cpus::get() * 2).max(2);
    let state = Arc::new(AppState {
        models,
        client_signals: Mutex::new(HashMap::new()),
        inflight_sem: Arc::new(Semaphore::new(max_inflight)),
    });

    spawn_memory_monitor();

    let app = Route::new()
        .at("/reconstruct/:model_id", post(reconstruct))
        .at("/health", poem::get(health))
        .data(state);

    println!(
        "server on http://0.0.0.0:8000 (max_inflight={}, max_iter={}, tol={})",
        max_inflight, MAX_ITER, TOL
    );
    Server::new(TcpListener::bind("0.0.0.0:8000")).run(app).await
}
