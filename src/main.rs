use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;

use bytes::Bytes;
use ndarray::{Array1, ArrayView1};
use ndarray_npy::NpzReader;
use zip::ZipArchive;
use poem::{
    handler,
    http::StatusCode,
    listener::TcpListener,
    post,
    web::{Data, Json, Path},
    EndpointExt, IntoResponse, Response, Route, Server,
};
use serde::Deserialize;
use serde_json::json;
use sprs::CsMat;

type F = f32;

struct Model {
    h: CsMat<F>,
    ht: CsMat<F>, // precomputed CSR transpose for fast matvec
    s: usize,
    n: usize,
    shape: (usize, usize),
}

struct AppState {
    models: HashMap<String, Arc<Model>>,
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
/// The npy data section is the 3 raw ASCII bytes right after the header.
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
/// scipy saves the arrays: format, shape, data, indices, indptr.
fn load_scipy_sparse_npz(path: &str) -> Result<CsMat<F>, Box<dyn std::error::Error>> {
    use std::io::{Read, Seek, SeekFrom};

    // Read the format entry as raw bytes (dtype is |S3, not readable by ndarray-npy).
    let file = File::open(path)?;
    let mut archive = ZipArchive::new(file)?;
    let fmt = {
        let mut entry = archive.by_name("format.npy")?;
        let mut buf = Vec::new();
        entry.read_to_end(&mut buf)?;
        parse_npy_s3_string(&buf)?
    };

    // Reclaim the file handle, seek back to start, then use NpzReader for typed arrays.
    let mut file = archive.into_inner();
    file.seek(SeekFrom::Start(0))?;
    let mut npz = NpzReader::new(file)?;

    let shape_arr: Array1<i64> = npz.by_name("shape.npy")?;
    let data_arr: Array1<f64> = npz.by_name("data.npy")?;
    let indices_arr: Array1<i32> = npz.by_name("indices.npy")?;
    let indptr_arr: Array1<i32> = npz.by_name("indptr.npy")?;

    let nrows = shape_arr[0] as usize;
    let ncols = shape_arr[1] as usize;

    let data: Vec<F> = data_arr.iter().map(|&x| x as F).collect();
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

    let ht = h.transpose_view().to_other_storage(); // materialize transpose as CSR
    println!("loaded {}: nnz={}, transpose nnz={}", name, h.nnz(), ht.nnz());

    Ok(Model { h, ht, s: cfg.s, n: cfg.n, shape: cfg.shape })
}

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
        // p = z + beta * p  ->  p *= beta; p += z
        p *= beta;
        p += &z;
        norm_z_sq = norm_z_new_sq;
    }

    (f, max_iter, r.dot(&r).sqrt())
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

fn run_reconstruction(model: &Model, g: Array1<F>) -> Array1<F> {
    let (mut f, _iters, _err) = cgnr(model, g.view(), 10, 1e-5);
    minmax_normalize(f.as_slice_mut().unwrap());
    f
}

// ---------- JSON endpoint (compat) ----------

#[derive(Deserialize)]
struct Sinal {
    g: Vec<F>,
}

#[handler]
async fn reconstruct_json(
    Path(model_id): Path<String>,
    Data(state): Data<&Arc<AppState>>,
    Json(sinal): Json<Sinal>,
) -> Response {
    let model = match state.models.get(&model_id) {
        Some(m) => m.clone(),
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({"error": "model not found", "model_id": model_id})),
            )
                .into_response();
        }
    };

    if sinal.g.len() != model.s {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "signal length mismatch", "expected": model.s, "got": sinal.g.len()})),
        )
            .into_response();
    }

    let f = tokio::task::spawn_blocking(move || {
        let g = Array1::from(sinal.g);
        run_reconstruction(&model, g)
    })
    .await;

    match f {
        Ok(arr) => {
            // reshape into 2D row-by-row, matching Python's order='F'
            // means out[i][j] = arr[j*rows + i]. We materialize as Vec<Vec<F>>.
            let model = state.models.get(&model_id).unwrap();
            let (rows, cols) = model.shape;
            let mut img = vec![vec![0.0f32; cols]; rows];
            for j in 0..cols {
                for i in 0..rows {
                    img[i][j] = arr[j * rows + i];
                }
            }
            Json(json!({"message": "reconstruction complete", "image": img})).into_response()
        }
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "compute task failed"})),
        )
            .into_response(),
    }
}

// ---------- Binary endpoint (fast path) ----------
// Request body: raw f32 little-endian bytes, length S*4.
// Response body: raw f32 little-endian bytes, length N*4, in the same flat order as `f`.

#[handler]
async fn reconstruct_binary(
    Path(model_id): Path<String>,
    Data(state): Data<&Arc<AppState>>,
    body: Bytes,
) -> Response {
    let model = match state.models.get(&model_id) {
        Some(m) => m.clone(),
        None => return (StatusCode::NOT_FOUND, "model not found").into_response(),
    };

    let expected = model.s * std::mem::size_of::<F>();
    if body.len() != expected {
        return (
            StatusCode::BAD_REQUEST,
            format!("expected {} bytes, got {}", expected, body.len()),
        )
            .into_response();
    }

    // Safe parse, vectorizes well in release mode.
    let g_vec: Vec<F> = body
        .chunks_exact(4)
        .map(|c| F::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let result = tokio::task::spawn_blocking(move || {
        let g = Array1::from(g_vec);
        run_reconstruction(&model, g)
    })
    .await;

    match result {
        Ok(arr) => {
            let slice = arr.as_slice().unwrap();
            let mut out = Vec::<u8>::with_capacity(slice.len() * 4);
            for &x in slice {
                out.extend_from_slice(&x.to_le_bytes());
            }
            Response::builder()
                .header("content-type", "application/octet-stream")
                .body(out)
        }
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "compute task failed").into_response(),
    }
}

#[handler]
async fn health(Data(state): Data<&Arc<AppState>>) -> Json<serde_json::Value> {
    Json(json!({"status": "ok", "models": state.models.keys().collect::<Vec<_>>()}))
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

    let state = Arc::new(AppState { models });
    let app = Route::new()
        .at("/reconstruct/:model_id", post(reconstruct_json))
        .at("/reconstruct_bin/:model_id", post(reconstruct_binary))
        .at("/health", poem::get(health))
        .data(state);

    println!("server on http://0.0.0.0:8000 (workers: {})", num_cpus::get());
    Server::new(TcpListener::bind("0.0.0.0:8000")).run(app).await
}
