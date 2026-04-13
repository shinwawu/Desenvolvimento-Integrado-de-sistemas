use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};
use polars::prelude::*;
// estrutura para representar um modelo
struct Model {
    s: i32,
    n: i32,
    shape: (usize, usize),     
    path: String,
    matrix: Option<Arc<Vec<usize>>>, 
    prioridade: String,
    in_use: bool,
}
// construcao do modelo em um hashmap para acesso global e realizar threading com mutex
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
            path: String::from("../data/H-2.npz"),
            matrix: None,
            prioridade: String::from("alta"),
            in_use: false,
        }),
    );
    map
});

fn verificar_matrix(model_id: &str) -> Option<Arc<Vec<usize>>> {
    //tentamos buscar o modelo no hashmap
    let model_mutex = MODELS_H.get(model_id)?; // 

    // trava o mutex para acessar os dados do modelo
    let model = model_mutex.lock().unwrap();

    // verifica se a matriz já foi carregada, se sim, retorna a matriz 
    if model.matrix.is_some() {
        
        return model.matrix.clone();
    }
    else {
        model.matrix = polars::prelude::LazyFrame::scan_npz(&model.path)
            .ok()?
            .collect()
            .ok()?
            .column("matrix")?
            .utf8()?
            .get(0)
            .and_then(|s| s.parse::<Vec<usize>>().ok())
            .map(Arc::new);
    };
    return model.matrix.clone();
    
};
fn main() {
    if let Some(matriz_compartilhada) = verificar_matrix("60x60") {
        println!("matriz obtida de outro modelo, tamanho: {}", matriz_compartilhada.len());
    } else {
        println!("a matriz ainda nao foi carregada no modelo");
    }
}