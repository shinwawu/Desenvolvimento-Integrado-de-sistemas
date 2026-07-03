
use std::collections::HashMap;
use std::fs::File;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};

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
use tokio::sync::Notify;
use faer::sparse::{SparseColMat, SymbolicSparseColMat};
use faer::sparse::linalg::matmul::sparse_dense_matmul;
use faer::{Accum, Par};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

type F = f32;

const MAX_ITER: usize = 10;
const TOL: F = 1e-4;
// tempo maximo p reconstrucao d img

const TEMPO_CONSTRUCAO: u64 = 120;
const MEMO_MINIMA: f64 = 0.5;
// so rejeita em ultimo caso, depois de 5 min esperando 
const TEMPO_DE_ESPERA: u64 = 300;
// frequencia de re-checagem da memoria durante a espera 
const TEMPO_VERIFICACAO_MS: u64 = 500;

// classe criada para armazenar modelo H
struct Model {
    h: SparseColMat<usize, F>,
    s: usize,
    n: usize,
    shape: (usize, usize),
}

// Estado de um job assincrono. Pending enquanto a reconstrucao roda em background
// Ready quando termina (sucesso ou erro), guardando o corpo JSON ja serializado e
// o status HTTP que o /result deve devolver.
enum JobState {
    Pending,
    Ready { status_code: u16, body: String },
}

/// Limite de reconstrucoes simultaneas que SOBE e DESCE em runtime conforme a
/// carga 
struct LimiteDinamico {
    estado: Mutex<(usize, usize)>, // (maximo, em_execucao)
    notify: Notify,
}

impl LimiteDinamico {
    fn new(maximo: usize) -> Self {
        Self { estado: Mutex::new((maximo, 0)), notify: Notify::new() }
    }

    async fn acquire(self: &Arc<Self>) -> PermitDinamico {
        loop {
            // registra o waiter ANTES de checar a condicao (enable()), senao um
            // notify entre a checagem e o await seria perdido.
            let espera = self.notify.notified();
            tokio::pin!(espera);
            espera.as_mut().enable();
            {
                let mut e = self.estado.lock().unwrap();
                if e.1 < e.0 {
                    e.1 += 1;
                    return PermitDinamico { limite: self.clone() };
                }
            }
            espera.await;
        }
    }

    fn set_maximo(&self, novo: usize) {
        let aumentou = {
            let mut e = self.estado.lock().unwrap();
            let a = novo > e.0;
            e.0 = novo;
            a
        };
        // se o teto subiu, acorda os que esperam para reavaliarem a condicao
        if aumentou {
            self.notify.notify_waiters();
        }
    }

    fn em_execucao(&self) -> usize {
        self.estado.lock().unwrap().1
    }
}

/// Solta a vaga ao ser dropado (fim da reconstrucao) e acorda quem espera.
struct PermitDinamico {
    limite: Arc<LimiteDinamico>,
}

impl Drop for PermitDinamico {
    fn drop(&mut self) {
        {
            let mut e = self.limite.estado.lock().unwrap();
            e.1 -= 1;
        }
        self.limite.notify.notify_waiters();
    }
}

// estado global do worker
struct AppState {
    //modelos
    models: HashMap<String, Arc<Model>>,
    // limite dinamico de reconstrucoes simultaneas (ajustado por carga)
    request_max: Arc<LimiteDinamico>,
    // mapa de jobs, guardando o estado de cada job assincrono
    jobs: Mutex<HashMap<String, JobState>>,
    // atrobio job_id
    job_seq: AtomicU64,
    // porta deste worker, usada como prefixo do job_id.
    port: u16,
}

//confg de cada modelo
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

// funcao para extrair o campo format do npz
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

/// funcao que carrega a matriz
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

// funcao que carrega a matriz carregado sprs para o formato faer
fn sprs_to_faer(m: CsMat<F>) -> SparseColMat<usize, F> {
    let m_csc = if m.is_csr() { m.to_other_storage() } else { m };
    let (nrows, ncols) = m_csc.shape();
    let col_ptr: Vec<usize> = m_csc.indptr().raw_storage().to_vec();
    let row_idx: Vec<usize> = m_csc.indices().to_vec();
    let values: Vec<F> = m_csc.data().to_vec();
    let symbolic = SymbolicSparseColMat::new_checked(nrows, ncols, col_ptr, None, row_idx);
    SparseColMat::new(symbolic, values)
}

// funcao que chama a funcao de carregamento e conversao, validando a forma da matriz
fn load_model(name: &str, cfg: &ModelCfg) -> Result<Model, Box<dyn std::error::Error>> {
    let h_sprs = load_scipy_sparse_npz(cfg.path)?;
    if h_sprs.shape() != (cfg.s, cfg.n) {
        return Err(
            format!("{}: shape {:?}, expected ({}, {})", name, h_sprs.shape(), cfg.s, cfg.n).into(),
        );
    }
    let nnz = h_sprs.nnz();
    let h = sprs_to_faer(h_sprs);
    println!("loaded {}: nnz={} (faer SparseColMat, sem Hᵀ materializado)", name, nnz);
    Ok(Model { h, s: cfg.s, n: cfg.n, shape: cfg.shape })
}

// essa funcao compuyta o produto H @ p no CGNR . E H @ p e H @ htr no CGNE. Retornando o resultado do produto
fn par_csr_matvec(h: &SparseColMat<usize, F>, x: &ArrayView1<F>) -> Array1<F> {
    let nrows = h.nrows();
    let ncols = h.ncols();
    let x_slice = x.as_slice().expect("x must be contiguous");
    let x_mat = faer::MatRef::from_column_major_slice(x_slice, ncols, 1);

    let mut out_vec = vec![0.0_f32; nrows];
    let y_mat = faer::MatMut::from_column_major_slice_mut(&mut out_vec, nrows, 1);

    sparse_dense_matmul(
        y_mat,
        Accum::Replace,
        h.as_ref(),
        x_mat,
        1.0,
        Par::Seq,
    );
    Array1::from(out_vec)
}

//serve para calcular o produto da Ht @ r 
fn ht_matvec(h: &SparseColMat<usize, F>, r: &ArrayView1<F>) -> Array1<F> {
    let (sym, vals) = h.as_ref().parts();
    let col_ptr = sym.col_ptr();
    let row_idx = sym.row_idx();
    let ncols = h.ncols();
    let r_slice = r.as_slice().expect("r must be contiguous");
    let mut out = vec![0.0_f32; ncols];
    for j in 0..ncols {
        let mut acc = 0.0_f32;
        for k in col_ptr[j]..col_ptr[j + 1] {
            acc += vals[k] * r_slice[row_idx[k]];
        }
        out[j] = acc;
    }
    Array1::from(out)
}


fn cgnr(model: &Model, g: ArrayView1<F>, max_iter: usize, tol: F) -> (Array1<F>, usize, F) {
    let n = model.n;
    let tol_sq = tol * tol;
    let mut f = Array1::<F>::zeros(n);
    let mut r = g.to_owned();
    let mut z: Array1<F> = ht_matvec(&model.h, &r.view());
    let mut p = z.clone();
    let mut norm_z_sq = z.dot(&z);

    for k in 0..max_iter {
        let w: Array1<F> = par_csr_matvec(&model.h, &p.view());
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
        z = ht_matvec(&model.h, &r.view());
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

fn cgne(model: &Model, g: ArrayView1<F>, max_iter: usize, tol: F) -> (Array1<F>, usize, F) {
    let n = model.n;
    let tol_sq = tol * tol;
    let mut f = Array1::<F>::zeros(n);
    let mut r = g.to_owned();              // r = g - H*0 = g
    let mut p: Array1<F> = ht_matvec(&model.h, &r.view());
    let mut rtr = r.dot(&r);

    for k in 0..max_iter {
        let hp: Array1<F> = par_csr_matvec(&model.h, &p.view());
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
        let htr: Array1<F> = ht_matvec(&model.h, &r.view());
        p *= beta;
        p += &htr;
        rtr = new_rtr;
    }
    (f, max_iter, rtr.sqrt())
}

/// normaliza para que o range de valores fique [0,1], p evitar que o brilho do sinal afete a escala de cinza
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

/// roda a reconstrucao com o algoritmo escolhido
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
    // normaliza para o absoluto antes da escala de cinza para evitar que o
    // brilho do sinal afete a escala de cinza da imagem reconstruida 
    f.mapv_inplace(F::abs);
    minmax_normalize(f.as_slice_mut().unwrap());
    Ok((f, iters, err))
}


// deserializa a request de reconstrucao, validando campos da escolha da requisicao
#[derive(Deserialize)]
struct ReconstructQuery {
    cliente_id: String,
    algorithm: String,
    #[serde(default, deserialize_with = "deserialize_bool_loose")]
    complete: bool,
}


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

// memoria disponivel do sistema 
fn memoria_disponivel_gb() -> f64 {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_memory();
    disponivel_gb(&sys)
}

/// memoria disponivel a partir de um system ja com refresh_memory feito
fn disponivel_gb(sys: &sysinfo::System) -> f64 {
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    let bytes = match sys.available_memory() {
        0 => sys.total_memory().saturating_sub(sys.used_memory()),
        avail => avail,
    };
    bytes as f64 / GB
}

// funcaq assincrono: o POST valida o request, cria um job, dispara a reconstrucao
// em background e responde na hora com {job_id}. O cliente consulta o
// resultado depois em GET /result/{job_id}.
#[handler]
async fn reconstruct(
    Path(model_id_in_path): Path<String>,
    Query(params): Query<ReconstructQuery>,
    Data(state): Data<&Arc<AppState>>,
    Json(sinal): Json<Sinal>,
) -> Response {

    let g_arr = Array1::from(sinal.g);
    let algo_choice = params.algorithm.clone();

    let model = match state.models.get(&model_id_in_path) {
        Some(m) => m.clone(),
        None => {
            return Json(json!({
                "status": "error",
                "error": format!("modelo '{}' nao encontrado. modelos disponiveis: {:?}",
                    model_id_in_path, state.models.keys().collect::<Vec<_>>())
            })).into_response();
        }
    };

    if g_arr.len() != model.s {
        return Json(json!({
            "status": "error",
            "error": format!("tamanho do sinal g={} diferente do esperado {} para o modelo {}",
                g_arr.len(), model.s, model_id_in_path)
        })).into_response();
    }

    // cria o job e dispara o processamento em background.
    let seq = state.job_seq.fetch_add(1, Ordering::Relaxed);
    let job_id = format!("{}-{}", state.port, seq);
    state.jobs.lock().unwrap().insert(job_id.clone(), JobState::Pending);

    let state_bg = state.clone();
    let model_id = model_id_in_path.clone();
    let job_id_bg = job_id.clone();
    tokio::spawn(async move {
        let resultado = processar_reconstrucao(&state_bg, &model, &model_id, &algo_choice, g_arr).await;
        state_bg.jobs.lock().unwrap().insert(job_id_bg, resultado);
    });

    // responde imediatamente a conexao do POST nao fica presa ate a reconstrucao.
    (
        StatusCode::ACCEPTED,
        Json(json!({ "status": "pending", "job_id": job_id })),
    ).into_response()
}

// Executa a reconstrucao de um job em background e devolve o JobState::Ready final
async fn processar_reconstrucao(
    state: &Arc<AppState>,
    model: &Arc<Model>,
    model_id: &str,
    algo_choice: &str,
    g_arr: Array1<F>,
) -> JobState {
    // verifica se ha memoria disponivel, se nao aguarda por 5 minutos a memoria ser liberada
    if memoria_disponivel_gb() < MEMO_MINIMA {
        let deadline = std::time::Instant::now()
            + std::time::Duration::from_secs(TEMPO_DE_ESPERA);
        while memoria_disponivel_gb() < MEMO_MINIMA {
            if std::time::Instant::now() > deadline {
                return JobState::Ready {
                    status_code: StatusCode::SERVICE_UNAVAILABLE.as_u16(),
                    body: json!({
                        "status": "error",
                        "error": format!(
                            "memoria abaixo de {}GB por mais de {}s; rejeitando para evitar deadlock",
                            MEMO_MINIMA, TEMPO_DE_ESPERA
                        )
                    }).to_string(),
                };
            }
            tokio::time::sleep(std::time::Duration::from_millis(TEMPO_VERIFICACAO_MS)).await;
        }
    }

    let _permit = state.request_max.acquire().await;

    let start_dt = iso_now();
    let t0 = std::time::Instant::now();

    let model_for_task = model.clone();
    let algo = algo_choice.to_string();
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(TEMPO_CONSTRUCAO),
        tokio::task::spawn_blocking(move || run_reconstruction(&model_for_task, &algo, g_arr)),
    ).await;

    let elapsed = t0.elapsed().as_secs_f64();
    let end_dt = iso_now();

    let (f, iters, err) = match result {
        Ok(Ok(Ok(triple))) => triple,
        Ok(Ok(Err(msg))) => {
            return JobState::Ready {
                status_code: StatusCode::INTERNAL_SERVER_ERROR.as_u16(),
                body: json!({ "status": "error", "error": format!("erro na reconstrucao: {}", msg) }).to_string(),
            };
        }
        Ok(Err(_)) => {
            return JobState::Ready {
                status_code: StatusCode::INTERNAL_SERVER_ERROR.as_u16(),
                body: json!({ "status": "error", "error": "task panicked" }).to_string(),
            };
        }
        Err(_) => {
            return JobState::Ready {
                status_code: StatusCode::GATEWAY_TIMEOUT.as_u16(),
                body: json!({ "status": "error", "error": format!("timeout > {}s na reconstrucao", TEMPO_CONSTRUCAO) }).to_string(),
            };
        }
    };

    println!("reconstrucao {} completa: iters={}, erro final={:.6}", model_id, iters, err);

    // serializa a resposta manualmente sem passar por serde_json::Value nem alocar
    // Vec<Vec<f32>>. ryu/itoa sao formatadores ASCII otimos (mais rapidos que dtoa
    // do serde_json). image fica reshapado em ordem 'F' (coluna principal) inline
    // direto na escrita do JSON, evitando o staging intermediario.
    let (rows, cols) = model.shape;
    let mut buf = String::with_capacity(rows * cols * 12 + 256);
    let mut ryu_buf = ryu::Buffer::new();
    let mut itoa_buf = itoa::Buffer::new();

    buf.push_str(r#"{"status":"done","message":"reconstrucao completa para "#);
    buf.push_str(model_id);
    buf.push_str(r#"","image":["#);
    for i in 0..rows {
        if i > 0 { buf.push(','); }
        buf.push('[');
        for j in 0..cols {
            if j > 0 { buf.push(','); }
            buf.push_str(ryu_buf.format(f[j * rows + i]));
        }
        buf.push(']');
    }
    buf.push_str(r#"],"iters":"#);
    buf.push_str(itoa_buf.format(iters));
    buf.push_str(r#","erro_final":"#);
    buf.push_str(ryu_buf.format(err));
    buf.push_str(r#","tempo_reconstrucao":"#);
    buf.push_str(ryu_buf.format(elapsed));
    buf.push_str(r#","tempo_inicio":""#);
    buf.push_str(&start_dt);
    buf.push_str(r#"","tempo_fim":""#);
    buf.push_str(&end_dt);
    buf.push_str("\"}");

    JobState::Ready { status_code: StatusCode::OK.as_u16(), body: buf }
}

// GET /result/{job_id}: devolve o estado do job. se tiver pendente, ele devolve status pendente, se tiver pronto, ele retorna a imagem, se for 
// inexistente, ele da erro
#[handler]
async fn job_result(Path(job_id): Path<String>, Data(state): Data<&Arc<AppState>>) -> Response {
    let mut jobs = state.jobs.lock().unwrap();
    match jobs.get(&job_id) {
        None => {
            drop(jobs);
            return (
                StatusCode::NOT_FOUND,
                Json(json!({ "status": "not_found", "error": "job nao encontrado" })),
            ).into_response();
        }
        Some(JobState::Pending) => {
            drop(jobs);
            return Json(json!({ "status": "pending" })).into_response();
        }
        Some(JobState::Ready { .. }) => {}
    }
    // terminal: remove e devolve o corpo ja serializado.
    let Some(JobState::Ready { status_code, body }) = jobs.remove(&job_id) else {
        unreachable!("job verificado como Ready acima");
    };
    drop(jobs);
    // retorna a mensagem 
    Response::builder()
        .status(StatusCode::from_u16(status_code).unwrap_or(StatusCode::OK))
        .header("content-type", "application/json")
        .body(body)
}

#[handler]
async fn health(Data(state): Data<&Arc<AppState>>) -> Json<serde_json::Value> {
    Json(json!({"status": "ok", "models": state.models.keys().collect::<Vec<_>>()}))
}

/// Uma amostra de uso de recursos num instante (1 por segundo).
#[derive(Clone, Copy)]
struct Amostra {
    t_s: f64,
    cpu_app: f32,      // % somado da arvore (pode passar de 100 com varios cores)
    rss_app_gb: f64,
    cpu_sys: f32,      // % medio do sistema (0-100)
    mem_avail_gb: f64,
}

/// Monitor de recursos 
fn spawn_resource_monitor(app_pids: Vec<u32>, samples: Arc<Mutex<Vec<Amostra>>>) {
    tokio::spawn(async move {
        use sysinfo::{System, Pid, ProcessesToUpdate};
        use std::io::Write;
        const GB: f64 = 1024.0 * 1024.0 * 1024.0;
        let mut sys = System::new_all();
        let pids: Vec<Pid> = app_pids.iter().map(|&p| Pid::from_u32(p)).collect();

        let mut csv = match std::fs::File::create("recursos_rust_server.csv") {
            Ok(f) => std::io::BufWriter::new(f),
            Err(e) => { eprintln!("[recursos] nao consegui criar CSV: {e}"); return; }
        };
        let _ = writeln!(csv, "t_s,cpu_app_pct,rss_app_gb,cpu_sys_pct,mem_used_gb,mem_avail_gb,n_procs");

        // 1a passada so fixa baseline de CPU (sysinfo precisa de 2 refreshes p/ %)
        sys.refresh_cpu_all();
        sys.refresh_processes(ProcessesToUpdate::Some(&pids), true);
        let t0 = std::time::Instant::now();
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            sys.refresh_memory();
            sys.refresh_cpu_all();
            sys.refresh_processes(ProcessesToUpdate::Some(&pids), true);

            let rss_total: u64 = pids.iter()
                .filter_map(|p| sys.process(*p).map(|pr| pr.memory()))
                .sum();
            let n_procs = pids.iter().filter(|p| sys.process(**p).is_some()).count();
            let cpu_app: f32 = pids.iter()
                .filter_map(|p| sys.process(*p).map(|pr| pr.cpu_usage()))
                .sum();
            let cpu_sys = sys.global_cpu_usage();
            let used = sys.used_memory() as f64 / GB;
            // available_memory() (com fallback total-used); ver disponivel_gb
            let avail = disponivel_gb(&sys);
            let rss_app_gb = rss_total as f64 / GB;
            let t_s = t0.elapsed().as_secs_f64();

            let _ = writeln!(csv, "{:.1},{:.1},{:.3},{:.1},{:.3},{:.3},{}",
                t_s, cpu_app, rss_app_gb, cpu_sys, used, avail, n_procs);
            let _ = csv.flush();

            let now = chrono::Local::now().format("%H:%M:%S");
            println!("[recursos {}] cpu_app={:.0}% rss_app={:.2}GB cpu_sys={:.0}% disponivel={:.2}GB",
                now, cpu_app, rss_app_gb, cpu_sys, avail);

            samples.lock().unwrap().push(Amostra {
                t_s, cpu_app, rss_app_gb, cpu_sys, mem_avail_gb: avail,
            });
        }
    });
}

/// ajusta o limite de reconstrucoes simultaneas (concurrency) dinamicamente de acordo com a memoria disponivel do sistema.
fn spawn_monitor_concorrencia(limite: Arc<LimiteDinamico>, max_request: usize) {
    tokio::spawn(async move {
        use sysinfo::System;
        const RAM_FOLGA_GB: f64 = 1.5;    // acima disso: teto cheio
        const RAM_CRITICA_GB: f64 = 0.5;  // abaixo disso: minimo
        let mut sys = System::new();
        let mut anterior: Option<usize> = None;
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            sys.refresh_memory();
            let disp = disponivel_gb(&sys);
            let novo = if disp >= RAM_FOLGA_GB {
                max_request
            } else if disp <= RAM_CRITICA_GB {
                1
            } else {
                let r = (disp - RAM_CRITICA_GB) / (RAM_FOLGA_GB - RAM_CRITICA_GB);
                1 + (r * (max_request - 1) as f64) as usize
            };
            let novo = novo.clamp(1, max_request);
            if Some(novo) != anterior {
                limite.set_maximo(novo);
                println!(
                    "[concorrencia] disponivel={disp:.2}GB -> limite={novo}/{max_request} (em_execucao={})",
                    limite.em_execucao()
                );
                anterior = Some(novo);
            }
        }
    });
}

/// aguarda SIGINT (Ctrl-C) ou SIGTERM (como o comparativo.py mata o server).
async fn aguardar_sinal_termino() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut sigterm = signal(SignalKind::terminate()).expect("sigterm");
        let mut sigint = signal(SignalKind::interrupt()).expect("sigint");
        tokio::select! {
            _ = sigterm.recv() => {}
            _ = sigint.recv() => {}
        }
    }
    #[cfg(not(unix))]
    { let _ = tokio::signal::ctrl_c().await; }
}

/// Desenha um painel (linha temporal) no SVG: eixos, grid, ticks Y e as series.
fn desenhar_painel(
    svg: &mut String, ml: f64, w: f64, mr: f64, y0: f64, y1: f64,
    titulo: &str, ts: &[f64], t_max: f64, series: &[(&str, &str, Vec<f64>)],
) {
    let plot_w = w - ml - mr;
    let x_of = |t: f64| ml + (t / t_max) * plot_w;
    let mut ymax = 0.0_f64;
    for (_, _, vs) in series { for &v in vs { ymax = ymax.max(v); } }
    if ymax <= 0.0 { ymax = 1.0; }
    ymax *= 1.1;
    let y_of = |v: f64| y1 - (v / ymax) * (y1 - y0);
    // precisao dos rotulos do eixo Y: escalas pequenas (GB) precisam de decimais;
    // escalas grandes (CPU%) ficam melhores como inteiro.
    let casas = if ymax < 10.0 { 2 } else { 0 };

    // eixos
    svg.push_str(&format!(r#"<line x1="{ml}" y1="{y0}" x2="{ml}" y2="{y1}" stroke="black"/>"#));
    svg.push_str(&format!(r#"<line x1="{ml}" y1="{y1}" x2="{:.0}" y2="{y1}" stroke="black"/>"#, w - mr));
    // grid + ticks Y
    for i in 0..=5 {
        let v = ymax * (i as f64) / 5.0;
        let y = y_of(v);
        svg.push_str(&format!(r##"<line x1="{ml}" y1="{y:.1}" x2="{:.0}" y2="{y:.1}" stroke="#dddddd"/>"##, w - mr));
        svg.push_str(&format!(r#"<text x="{:.0}" y="{:.1}" text-anchor="end">{:.*}</text>"#, ml - 6.0, y + 4.0, casas, v));
    }
    svg.push_str(&format!(r#"<text x="{ml}" y="{:.0}" font-weight="bold">{titulo}</text>"#, y0 - 8.0));

    // series + legenda
    let mut lx = w - mr - (series.len() as f64) * 130.0;
    for (nome, cor, vs) in series {
        let mut pts = String::with_capacity(vs.len() * 14);
        for (i, &v) in vs.iter().enumerate() {
            pts.push_str(&format!("{:.1},{:.1} ", x_of(ts[i]), y_of(v)));
        }
        svg.push_str(&format!(r#"<polyline points="{pts}" fill="none" stroke="{cor}" stroke-width="1.5"/>"#));
        svg.push_str(&format!(r#"<rect x="{lx:.0}" y="{:.0}" width="11" height="11" fill="{cor}"/>"#, y0 - 12.0));
        svg.push_str(&format!(r#"<text x="{:.0}" y="{:.0}">{nome}</text>"#, lx + 15.0, y0 - 3.0));
        lx += 130.0;
    }
}

/// Gera um grafico SVG para o uso de recursos (CPU, memoria) do server Rust. O grafico eh salvo em path.
fn gerar_grafico_recursos(amostras: &[Amostra], path: &str) {
    use std::io::Write;
    if amostras.is_empty() {
        eprintln!("[recursos] sem amostras, grafico nao gerado");
        return;
    }
    let (w, h, ml, mr) = (1000.0_f64, 720.0_f64, 70.0_f64, 30.0_f64);
    let t_max = amostras.last().unwrap().t_s.max(1.0);
    let ts: Vec<f64> = amostras.iter().map(|a| a.t_s).collect();

    let mut svg = String::with_capacity(amostras.len() * 64 + 4096);
    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="monospace" font-size="12">"#
    ));
    svg.push_str(&format!(r#"<rect width="{w}" height="{h}" fill="white"/>"#));
    svg.push_str(&format!(r#"<text x="{:.0}" y="22" font-weight="bold" font-size="15">Uso de recursos - server Rust</text>"#, ml));

    // painel cpu (cima)
    let n_cores = num_cpus::get() as f64;
    desenhar_painel(&mut svg, ml, w, mr, 70.0, 340.0, "CPU (%) ", &ts, t_max, &[
        ("CPU app (%)", "#1f77b4", amostras.iter().map(|a| a.cpu_app as f64).collect()),
        ("CPU sistema (%)", "#ff7f0e", amostras.iter().map(|a| a.cpu_sys as f64 * n_cores).collect()),
    ]);
    // painel memoria (baixo)
    desenhar_painel(&mut svg, ml, w, mr, 410.0, 680.0, "Memoria (GB)", &ts, t_max, &[
        ("RSS app (GB)", "#d62728", amostras.iter().map(|a| a.rss_app_gb).collect()),
        ("RAM disponivel (GB)", "#2ca02c", amostras.iter().map(|a| a.mem_avail_gb).collect()),
    ]);
    svg.push_str(&format!(r#"<text x="{:.0}" y="705" text-anchor="middle">tempo (s) — 0 a {t_max:.0}</text>"#, w / 2.0));
    svg.push_str("</svg>");

    match std::fs::File::create(path) {
        Ok(mut f) => {
            let _ = f.write_all(svg.as_bytes());
            println!("[recursos] salvo: recursos_rust_server.csv + {path}");
        }
        Err(e) => eprintln!("[recursos] nao consegui salvar grafico: {e}"),
    }
}


// Cada worker carrega os modelos H-1.npz/H-2.npz e expoe /reconstruct/{model_id}
// e /health. 

async fn run_worker(port: u16, max_request: usize) -> Result<(), std::io::Error> {
    let mut models = HashMap::new();
    for (name, cfg) in model_configs() {
        // carrega os odelos
        match load_model(name, &cfg) {
            Ok(m) => { models.insert(name.to_string(), Arc::new(m)); }
            Err(e) => eprintln!("failed to load {}: {}", name, e),
        }
    }

    #[allow(non_snake_case)]
    let MAX_REQUEST = max_request;

    //config do worker
    let state = Arc::new(AppState {
        models,
        request_max: Arc::new(LimiteDinamico::new(MAX_REQUEST)),
        jobs: Mutex::new(HashMap::new()),
        job_seq: AtomicU64::new(0),
        port,
    });

    // ajusta o limite de reconstrucoes simultaneas em runtime conforme a carga
    spawn_monitor_concorrencia(state.request_max.clone(), MAX_REQUEST);

    // o monitor de memoria roda no proxy (1 linha agregada por segundo).

    let app = Route::new()
        .at("/reconstruct/:model_id", post(reconstruct))
        .at("/result/:job_id", poem::get(job_result))
        .at("/health", poem::get(health))
        .data(state);

    println!(
        "[worker:{port}] server on http://0.0.0.0:{port} (MAX_REQUEST={MAX_REQUEST}, max_iter={MAX_ITER}, tol={TOL})"
    );
    Server::new(TcpListener::bind(format!("0.0.0.0:{port}"))).run(app).await
}

// implementamos o load balancer
// realizamos o round robin para distribuir a carga entre os workers
// e utilizamos um job_id com prefixo na porta do worker para indicar qual worker está atribuído a ele

// o proxy roteia as requisicoes de reconstrucao para os workers 
struct ProxyState {
    worker_ports: Vec<u16>,
    rr_counter: std::sync::atomic::AtomicUsize,
    http: reqwest::Client,
}

#[handler]
async fn proxy_reconstruct(
    Path(model_id_in_path): Path<String>,
    Query(params): Query<ReconstructQuery>,
    Data(state): Data<&Arc<ProxyState>>,
    body: bytes::Bytes,
) -> Response {
    // round-robin: contador atomico mod N. Distribuicao uniforme entre workers,

    let idx = state.rr_counter
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        % state.worker_ports.len();
    let port = state.worker_ports[idx];
    let url = format!(
        "http://127.0.0.1:{port}/reconstruct/{model_id_in_path}\
         ?cliente_id={cid}&algorithm={algo}&model_id={mid}&complete={cmp}",
        cid = urlencode(&params.cliente_id),
        algo = urlencode(&params.algorithm),
        mid = urlencode(&model_id_in_path),
        cmp = params.complete,
    );
    match state.http.post(&url)
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
    {
        Ok(r) => {
            let status = StatusCode::from_u16(r.status().as_u16()).unwrap_or(StatusCode::OK);
            let body = r.bytes().await.unwrap_or_default();
            Response::builder()
                .status(status)
                .header("content-type", "application/json")
                .body(body)
        }
        Err(e) => {
            Json(json!({"error": format!("proxy -> worker:{port} falhou: {e}")})).into_response()
        }
    }
}

// GET /result/{job_id}: o job_id e prefixado com a porta do worker dono (`port-seq`),
// entao roteamos o polling de volta para o mesmo worker que criou o job.
#[handler]
async fn proxy_result(
    Path(job_id): Path<String>,
    Data(state): Data<&Arc<ProxyState>>,
) -> Response {
    let port = match job_id.split('-').next().and_then(|p| p.parse::<u16>().ok()) {
        Some(p) if state.worker_ports.contains(&p) => p,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "status": "error", "error": format!("job_id invalido: {job_id}") })),
            ).into_response();
        }
    };
    let url = format!("http://127.0.0.1:{port}/result/{}", urlencode(&job_id));
    match state.http.get(&url).send().await {
        Ok(r) => {
            let status = StatusCode::from_u16(r.status().as_u16()).unwrap_or(StatusCode::OK);
            let body = r.bytes().await.unwrap_or_default();
            Response::builder()
                .status(status)
                .header("content-type", "application/json")
                .body(body)
        }
        Err(e) => {
            Json(json!({ "status": "error", "error": format!("proxy -> worker:{port} falhou: {e}") })).into_response()
        }
    }
}

#[handler]
async fn proxy_health(Data(state): Data<&Arc<ProxyState>>) -> Json<serde_json::Value> {
    Json(json!({"status": "ok", "mode": "proxy", "workers": state.worker_ports}))
}

/// Url-encoding minimal: percent-escape so caracteres problematicos.
fn urlencode(s: &str) -> String {
    s.bytes().map(|b| match b {
        b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
            (b as char).to_string()
        }
        _ => format!("%{:02X}", b),
    }).collect()
}

async fn wait_for_health(port: u16, timeout_s: u64) -> bool {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(1))
        .build()
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_s);
    while std::time::Instant::now() < deadline {
        if let Ok(r) = client.get(format!("http://127.0.0.1:{port}/health")).send().await {
            if r.status().is_success() {
                return true;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    false
}

/// Roda como proxy + spawna N workers como child processes.
async fn run_proxy(proxy_port: u16, n_workers: usize) -> Result<(), std::io::Error> {
    use std::process::Command;

    let worker_ports: Vec<u16> = (1..=n_workers as u16)
        .map(|i| proxy_port + i)
        .collect();

    let exe = std::env::current_exe()?;
    let mut children: Vec<std::process::Child> = Vec::with_capacity(n_workers);


    let per_worker_max =
        (((num_cpus::get() as f32 * 1.5) / n_workers as f32).round() as usize).max(1);
    println!(
        "[proxy] iniciando {n_workers} workers nas portas {worker_ports:?} (max_request/worker={per_worker_max})"
    );
    for &p in &worker_ports {
        let mut cmd = Command::new(&exe);
        cmd.arg("--worker")
            .arg("--port").arg(p.to_string())
            .arg("--max-request").arg(per_worker_max.to_string());

        let child = cmd.spawn().expect("falha ao spawnar worker");
        children.push(child);
    }

    // espera todos os workers responderem /health 
    for &p in &worker_ports {
        if !wait_for_health(p, 60).await {
            eprintln!("[proxy] worker {p} nao subiu em 60s");
        }
    }
    println!("[proxy] {n_workers} workers prontos");

    // monitor de recursos (CPU+memoria) da arvore toda, gravando CSV a cada 1s
    let mut app_pids: Vec<u32> = children.iter().map(|c| c.id()).collect();
    app_pids.push(std::process::id());
    let samples: Arc<Mutex<Vec<Amostra>>> = Arc::new(Mutex::new(Vec::new()));
    spawn_resource_monitor(app_pids.clone(), samples.clone());

    // ao receber sinal de termino: gera o grafico ANTES de matar os workers
    let samples_sig = samples.clone();
    let mut kids_for_signal: Vec<u32> = children.iter().map(|c| c.id()).collect();
    tokio::spawn(async move {
        aguardar_sinal_termino().await;
        eprintln!("\n[proxy] sinal de termino recebido, gerando grafico de recursos...");
        gerar_grafico_recursos(&samples_sig.lock().unwrap(), "recursos_rust_server.svg");
        eprintln!("[proxy] encerrando workers...");
        for pid in kids_for_signal.drain(..) {
            kill_pid(pid);
        }
        std::process::exit(0);
    });

    // configuramos o request com pool de conexoes
    let http = reqwest::Client::builder()
        .pool_max_idle_per_host(128)
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .expect("build reqwest client");
    // estado do proxy
    let state = Arc::new(ProxyState {
        worker_ports: worker_ports.clone(),
        rr_counter: std::sync::atomic::AtomicUsize::new(0),
        http,
    });
    // roteamento do proxy, o proxy recebe as requisicoes e distribui para osworkers.
    // com o id do job prefixado com a porta do worker, o cliente sempre consulta o resultado no mesmo worker
    let app = Route::new()
        .at("/reconstruct/:model_id", post(proxy_reconstruct))
        .at("/result/:job_id", poem::get(proxy_result))
        .at("/health", poem::get(proxy_health))
        .data(state);

    println!("[proxy] escutando em http://0.0.0.0:{proxy_port} (sticky routing por cliente_id)");
    let result = Server::new(TcpListener::bind(format!("0.0.0.0:{proxy_port}"))).run(app).await;

    // saida normal (erro no server): gera o grafico antes de matar os filhos
    gerar_grafico_recursos(&samples.lock().unwrap(), "recursos_rust_server.svg");
    for mut c in children {
        let _ = c.kill();
        let _ = c.wait();
    }
    result
}

#[cfg(windows)]
fn kill_pid(pid: u32) {
    let _ = std::process::Command::new("taskkill")
        .args(["/F", "/PID", &pid.to_string()])
        .output();
}

#[cfg(not(windows))]
fn kill_pid(pid: u32) {
    let _ = std::process::Command::new("kill")
        .args(["-9", &pid.to_string()])
        .output();
}



fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    args.iter().position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

/// Calcula o numero de workers escalando com os DOIS recursos da maquina:
/// workers = min(teto_cpu, teto_ram). 
fn calcular_workers() -> usize {
    use sysinfo::System;
    const MIN_WORKERS: usize = 2;
    const RAM_POR_WORKER_GB: f64 = 0.9;
    const MARGEM_GB: f64 = 1.0;
    let mut sys = System::new();
    sys.refresh_memory();
    let disp_gb = disponivel_gb(&sys);
    let cpu = num_cpus::get();

    let teto_cpu = (cpu / 2).max(2);
    let teto_ram = (((disp_gb - MARGEM_GB) / RAM_POR_WORKER_GB) as i64).max(1) as usize;
    let n = teto_cpu.min(teto_ram).max(MIN_WORKERS);
    println!(
        "[proxy] topologia: cpu={cpu} disponivel={disp_gb:.1}GB -> workers={n} (teto_cpu={teto_cpu}, teto_ram={teto_ram})"
    );
    n
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), std::io::Error> {
    // inicializa os argumentos e passa eles para rodar como proxy ou worker.
    // proxy cria workers e distribui as requsicoes entre eles
    // enquanto worker carrega modleso e processa as requsicioes de construcao
    let args: Vec<String> = std::env::args().collect();
    let is_worker = args.iter().any(|a| a == "--worker");
    let port: u16 = parse_arg(&args, "--port").unwrap_or(8000);
    let max_request: usize = parse_arg(&args, "--max-request")
        .unwrap_or((num_cpus::get() * 2).max(2));

    if is_worker {
        run_worker(port, max_request).await
    } else {
        // sem --workers: calcula dinamicamente com base em CPU + RAM disponivel
        // (so no proxy; os workers nao recalculam)
        let n_workers: usize = parse_arg(&args, "--workers").unwrap_or_else(calcular_workers);
        run_proxy(port, n_workers).await
    }
}