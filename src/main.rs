use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};
use ndarray::{Array1, Array2, Axis};
use ndarray_npy::NpzReader;
use std::fs::File;
use poem::{get, handler, post, Route, Server, EndpointExt, web::{Json, Path}};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize)]
struct Sinal {
    g: Vec<f64>, 
}


struct Model {
    s: usize,
    n: usize,
    shape: (usize, usize),     
    path: String,
    matrix: Option<Arc<Array2<f64>>>, 
    prioridade: String,
    in_use: bool,
}


static MODELS_H: LazyLock<HashMap<String, Mutex<Model>>> = LazyLock::new(|| {
    let mut map = HashMap::new();

    map.insert(
        String::from("60x60"),
        Mutex::new(Model {    
            s: 50816,
            n: 3600,
            shape: (60, 60),
            path: String::from("../data/H-1.npz"),
            matrix: None,
            prioridade: String::from("baixa"),
            in_use: false,
        }),
    );

    map.insert(
        String::from("30x30"),
        Mutex::new(Model {
            s: 27904,
            n: 900,
            shape: (30, 30),
            path: String::from("data/H-2-dense.npz"),
            matrix: None,
            prioridade: String::from("alta"),
            in_use: false,
        }),
    );
    map
});

fn verificar_matrix(model_id: &str) -> Option<Arc<Array2<f64>>> {
    // Verifica se há uma instância do modelo em memória para usar sua matriz sem recarregar.
    if let Some(model_mutex) = MODELS_H.get(model_id) {
        let model = model_mutex.lock().unwrap();
        model.matrix.as_ref().cloned()
    } else {
        println!("Modelo {} não encontrado!", model_id);
        None
    }
}

fn carregar_matrix(model_id: &str) -> Option<Arc<Array2<f64>>> {
    if let Some(matrix) = verificar_matrix(model_id) {
        println!("Matriz do modelo {} já carregada em memória.", model_id);
        return Some(matrix);
    }
    if let Some(model_mutex) = MODELS_H.get(model_id) {
        let mut model = model_mutex.lock().unwrap();
        if model.matrix.is_none() {
            println!("Abrindo arquivo: {}", model.path);
            let file = match File::open(&model.path) {
                Ok(f) => f,
                Err(e) => {
                    println!("Erro ao abrir arquivo: {:?}", e);
                    return None;
                }
            };
            let mut npz = match NpzReader::new(file) {
                Ok(n) => n,
                Err(e) => {
                    println!("Erro ao abrir NPZ: {:?}", e);
                    return None;
                }
            };
            println!("Listando arrays no arquivo NPZ:");
            match npz.names() {
                Ok(names) => {
                    for name in names {
                        println!(" - {}", name);
                    }
                }
                Err(e) => {
                    println!("Erro ao listar arrays: {:?}", e);
                    return None;
                }
            }
            let mat: Array2<f64> = npz.by_name("arr_0.npy").ok()?;
            println!("Array lido com shape {:?}", mat.shape());
            model.matrix = Some(Arc::new(mat));
        }
        model.matrix.as_ref().cloned()
    } else {
        println!("Modelo {} não encontrado!", model_id);
        None
    }

}

fn cgnr_function(h_matrix: &Arc<Array2<f64>>, g: Vec<f64>, max_iter: usize, tol: f64) -> (Vec<f64>, usize, f64) {
    let g_arr = Array1::from(g);
    let mut f = Array1::<f64>::zeros(h_matrix.shape()[1]);
    let mut r = g_arr.clone();
    let mut z = h_matrix.t().dot(&r);
    let mut p = z.clone();
    let mut norm_z_sq = z.dot(&z);
    let mut error = r.dot(&r).sqrt();
    let mut iter = 0;

    while iter < max_iter && error > tol {
        let w = h_matrix.dot(&p);
        let norm_w_sq = w.dot(&w);
        if norm_w_sq == 0.0 {
            break;
        }
        let alpha = norm_z_sq / norm_w_sq;
        f = f + &(alpha * &p);
        r = r - &(alpha * &w);

        error = r.dot(&r).sqrt();
        if error < tol {
            break;
        }
        let z_new = h_matrix.t().dot(&r);
        let norm_z_new_sq = z_new.dot(&z_new);
        if norm_z_sq == 0.0 {
            break;
        }
        let beta = norm_z_new_sq / norm_z_sq;
        p = z_new.clone() + &(beta * &p);

        norm_z_sq = norm_z_new_sq;
        iter += 1;
    }

    (f.to_vec(), iter+1, error)
}   

fn construir_img(f: Vec<f64>, shape: (usize, usize)) -> Vec<Vec<f64>> {
    let mut img = vec![vec![0.0; shape.1]; shape.0];
    let img_array = Array2::from_shape_vec(shape, f).unwrap();
    let img_norm: f64 = img_array.iter().map(|&x| x * x).sum::<f64>().sqrt();

    if img_norm > 0.0 {
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                img[i][j] = img_array[[i, j]] / img_norm;
            }
        }
    } else {
        println!("Norma da imagem é zero, retornando imagem de zeros.");
    }
    img
}

fn run_cgnr(model_id: &str, g: Vec<f64>) -> Option<Vec<Vec<f64>>> {
    let model = MODELS_H.get(model_id)?.lock().unwrap();
    let s = model.s;
    let n = model.n;
    let img_shape = model.shape;
    drop(model); // Libere o lock antes de carregar a matriz

    if g.len() != s {
        println!("Erro: tamanho de g ({}) diferente do esperado ({})", g.len(), s);
        return None;
    }
    let h_matrix = carregar_matrix(model_id)?;
    println!("Matriz carregada com shape {:?}", h_matrix.shape());
    let (f, iter, erro_atual) = cgnr_function(&h_matrix, g, 10, 1e-6);
    println!("Resultado: Iterações={}, Erro={}", iter, erro_atual);

    Some(construir_img(f, img_shape))
}

#[handler]
async fn hello() -> &'static str {
    "Servidor de Reconstrução Online"
}

// AJUSTE: Extratores do Poem (Path e Json) configurados corretamente
#[handler]
async fn reconstruct(Path(model_id): Path<String>, Json(sinal): Json<Sinal>) -> Json<Value> {
    if !MODELS_H.contains_key(&model_id) {
        // AJUSTE: Utilizando serde_json::json! para construir a resposta padronizada
        return Json(json!({
            "error": "modelo não encontrado",
            "model_id": model_id
        }));
    }
    
    let img = run_cgnr(&model_id, sinal.g);

    if let Some(imagem) = img {
        Json(json!({
            "success": "imagem reconstruída",
            "model_id": model_id,
            "image": imagem
        }))
    } else {
        Json(json!({
            "error": "falha na reconstrução",
            "model_id": model_id
        }))
    }
}

#[tokio::main]
async fn main() -> Result<(), std::io::Error> {
    let app = Route::new()
        .at("/", get(hello))
        .at("/reconstruct/:model_id", post(reconstruct));
        
    println!("Servidor rodando em http://127.0.0.1:8000");
    Server::new(poem::listener::TcpListener::bind("127.0.0.1:8000"))
        .run(app)
        .await
}