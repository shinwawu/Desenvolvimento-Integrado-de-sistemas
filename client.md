Cliente

Na inicializacao de cada client.py
é criado 300 threads, p cada thread ocorre um processo assincrono 
ele chama a inicializacao do cliente e escolhe as configuracoes aleatorias
como modelo da imagem, algoritmo e o ganho.

Então na parte da request do envio de sinal, ocorre um sleep aleatorio entre 0.1s e 1.5s 
e cria uma sessao, e uma funcao do request, essa sessão é a conexão do client e server.
Além disso, temos uma funcao para fazer retry para erros.
No fim, faz envio do sinal dispara uma thread que realiza uma requisicao post
E recebe um job_id e aguarda o resultado
(Por que job_id + polling? Para nao manter a conexao do POST aberta durante a
reconstrucao, liberando o worker e a conexao em vez de segura-los esperando.)

Enquanto espera o resultado, realiza polling para verificar como está o processo
polling é a consulta periódica do resultado da tarefa ate obter resultado ou atingir time out

terminando, ele adiciona informacoes na imagem e salvar as métricas.
Acabando todos os processos, ele gera um relatorio dos resultados dos clientes com as métricas
e os resultados em geral

Server python
Construido com Fastapi
realizando validacoes de entrada com pydantic
Iniciamos a aplicacao com 2 workers e 2 proxy servindo a aplicacao principal
modelo load balancer com worker e proxy
E construimos um orquestrador que inicia os workers e o proxy
worker processo que lida com o processamento, cada worker carrega os modelos e recebe as solicitacoes de reconstrucao e retorna resultados
o proxy é responsavel por receber solicitacoes dos clientes e distribuilas entre os workers

então de inicio verificamos se as portas estao disponiveis e em seguida verificamos se os workers subiram.
logo em seguida criamos uma thread para servir como monitoramento de memoria e recurso.

Carregamos os modelos em memoria na inicializacao, e também precomputamos a matriz H transposta.
Temos monitoramento live direto do terminal da memoria.
E também eliminacao de jobs que ja foram concluidos, mas que estao na memoria ainda .

Então por meio do endpoint reconstruct model id, o usuario envia sinal e a configuracao de forma aleatoria e realizamos as verificacoes se o modelo esta carregado e se os dados informados estao compativeis, em seguida criamos uma task para processar a reconstrucao para um worker e retornamos o job_id.
(Por que o job_id tem prefixo porta-seq, ex 8001-42? Para o proxy saber rotear o
GET /result de volta ao mesmo worker dono do job — roteamento sticky por job.)

Então antes de processar, verificamos se temos memoria o suficiente antes, caso não, ele aguarda por liberacao de memoria por 5 min.
(Por que esperar em vez de rejeitar? Para nao derrubar requests sob pico de carga;
so rejeita em ultimo caso, depois dos 5 min.)
ai ele cria uma thread para realizar a reconstrucao e retorna

Eliminamos os jobs completos apos 60 s

Server Rust
Usamos o tokio como runtime assincrono (equivalente ao asyncio/uvicorn) e o poem
como framework HTTP (equivalente ao FastAPI). A deserializacao/validacao dos
campos do request e feita com serde (equivalente ao pydantic).

A arquitetura e o fluxo sao os mesmos da versao Python. O proxy sobe N workers
como processos filhos, espera o /health de cada um, distribui as requisicoes por
round-robin e roteia o GET /result pelo prefixo porta-seq do job_id (sticky por job).

O endpoint POST /reconstruct valida se o modelo esta carregado e se o tamanho do
sinal bate, cria o job (Pending), dispara a reconstrucao em background (tokio task)
e responde na hora com 202 e o job_id. O GET /result faz o polling: enquanto
Pending devolve pending; quando pronto, devolve o resultado e remove o job do mapa.

A politica de admissao e identica: espera por liberacao de memoria (ate 5 min antes
de rejeitar), limita reconstrucoes simultaneas com um semaforo (2*cpu) e roda o
calculo numa thread blocking com timeout de 120s. Os algoritmos CGNR/CGNE, o
max_iter=10, tol=1e-4, o abs e o minmax_normalize para [0,1] sao os mesmos.

Diferencas em relacao ao Python (otimizacoes):
- Os modelos usam faer (SparseColMat/CSC) e NAO materializam a matriz transposta:
  Hᵀ·r e calculado direto sobre H, cortando o uso de memoria pela metade.
  (Por isso o piso de memoria aqui e 0.05GB, contra 0.5GB no Python.)
- A resposta JSON com a imagem e serializada manualmente (ryu/itoa), sem montar
  estruturas intermediarias — mais rapido sob carga.
- O job e consumido no primeiro fetch terminal (removido do mapa na hora), sem a
  janela de graca / GC em background que a versao Python mantem.
- Usa o allocator mimalloc, melhor sob alta concorrencia.


script comparativo
O comparativo.py mede Python vs Rust na mesma maquina, sob a mesma carga: 3
clients.py em paralelo x 300 reqs cada (900 reconstrucoes por servidor).

A ideia central e a comparacao pareada: a cada invocacao sorteamos um seed base
(logado no resumo) e dele derivamos um seed por instancia. As mesmas seeds sao
passadas aos clients nas duas rodadas (Python e Rust), entao os dois servidores
recebem EXATAMENTE o mesmo workload — a mesma sequencia de img, algoritmo, ganho
e intervalos. Assim a diferenca medida e do servidor, nao do sorteio.

Para cada servidor o fluxo e: espera a porta 8000 livre, sobe o servidor e espera
o /health responder, dispara os 3 clients em paralelo (com stagger de 2s entre os
starts para nao criar um pico de 900 conexoes no mesmo instante), espera todos
terminarem e analisa os CSVs gerados pelos clients.

Da analise tiramos as metricas: numero de reconstrucoes, convergencia, divisao
por modelo (30x30/60x60), latencia (p50, p99, media, max) e throughput (req/s).

Entre rodar o Python e o Rust ha dois cuidados que garantem uma comparacao justa:
- Mata os workers fantasma: tanto o uvicorn (Python) quanto o binario Rust
  spawnam processos filhos que podem continuar segurando a porta 8000 depois do
  parent morrer; o script os encerra antes do proximo servidor subir.
- Espera o SO reclamar a memoria: cada worker segura ~600MB de modelos; se o Rust
  subir antes do SO liberar a memoria do Python, ele inicia sob pressao e o
  controle de admissao o estrangula — o que falsearia a comparacao por ordem de
  execucao. Por isso esperamos a memoria liberar/estabilizar antes de seguir.

No fim gera um relatorio unico (relatorio_comparativo.csv) com uma linha por
metrica e as colunas Python e Rust lado a lado. Os artefatos de cada rodada
(relatorios e imagens) sao renomeados com sufixo _python / _rust para um run nao
sobrescrever o outro.
