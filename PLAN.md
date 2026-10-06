# 🛡️ P0 — LLM Gateway (Python, construido desde cero)

> **Plan de proyecto, v2.** **Estado:** ✅ Plan completo · 🟡 Setup M0 hecho · **v2 (2026-10-06): absorbe por completo el "High-Performance LLM Edge Gateway" (Go) del CV** y lo supera en robustez. Reemplaza a ese proyecto en P1, P2, P3 y P4.
> Stack: Python 3.12 · FastAPI + uvicorn (uvloop, httptools) · httpx (async, HTTP/2) · Pydantic v2 · orjson · Redis 8 (budget, rate limit, caché exacta y **semántica obligatoria** con vector search HNSW) · ONNX Runtime (embeddings y guardrails en CPU) · Ollama · Groq · Google AI Studio (Gemma 4) · Anthropic · Prometheus + OpenTelemetry (GenAI semconv) · Langfuse · Presidio · Toxiproxy · k6 · Docker Compose · Cloud Run (solo en la prueba final de P3)
> Gasto: **$0** en desarrollo (`mock`, `ollama` y los free tiers de Groq y Google AI Studio). Anthropic (Haiku) solo en la prueba final de P3, con tope duro de presupuesto.

## Módulos del plan
| # | Sección | Estado |
|---|---|---|
| 0 | Herencia del gateway en Go: qué se conserva, qué se corrige y qué se agrega | ✅ v2 |
| 1 | Visión, problema y frase del CV | ✅ v2 |
| 2 | Arquitectura macro, ADRs y flujo de una request | ✅ v2 |
| 3 | Componentes | ✅ v2 |
| 4 | Resiliencia: timeouts, reintentos, circuit breaker, fallback, bulkheads, hedging, concurrencia adaptativa, load shedding | ✅ v2 |
| 5 | Control de costos: precios, presupuesto atómico, prompt caching del proveedor | ✅ v2 |
| 6 | **Caché semántica (obligatoria)**: diseño en capas contra falsos hits | ✅ v2 |
| 7 | Streaming, cancelación y salida estructurada | ✅ |
| 8 | Observabilidad, seguridad y guardrails | ✅ v2 |
| 9 | UI de chat, ejemplos de clientes y experiencia de desarrollo | ✅ v2 |
| 10 | Evaluación: overhead, caché, caos, disponibilidad, costo y contratos | ✅ v2 |
| 11 | Requisitos EARS (contrato de aceptación) | ✅ v2 |
| 12 | Recursos, ejecución y despliegue | ✅ |
| 13 | Estructura del repo y del README | ✅ v2 |
| 14 | Hitos de implementación | ✅ v2 |
| 15 | Riesgos y pendientes | ✅ v2 |

## Regla del README
README progresivo: **contexto teórico, conceptual y macro primero**; en cada componente, el detalle técnico al final. Incluye cómo funciona, los pasos para ejecutarlo y las alternativas de ejecución o despliegue (local primero).

---

## 0. Herencia del gateway en Go (`Leito2/llm-edge-gateway`)

El repo en Go queda **archivado aparte** como antecedente. P0 debe contener **todo** lo que ese proyecto demuestra, como componente real (no como idea), y corregir sus límites. Inventario hecho sobre el código (Go 1.22, Fiber v2, `sony/gobreaker`, Redis Stack 7.4 con RediSearch, Ollama, Groq, 66 tests).

### 0.1 Qué se conserva (paridad funcional obligatoria)
| Capacidad en Go | Implementación en Go | Equivalente en P0 | Sección |
|---|---|---|---|
| Caché semántica | `nomic-embed-text` (768d) vía Ollama, `FT.SEARCH` KNN 1 sobre HNSW COSINE (M=6), umbral 0.85, TTL 168 h, contador de hits, *write-back* asíncrono | Caché semántica en Redis 8 (HNSW COSINE) con **dos embedders intercambiables**: ONNX e5-small (CPU, por defecto) y `nomic-embed-text` vía Ollama (el original, como baseline medido) | §6 |
| Circuit breaker de 3 estados | `gobreaker`: abre tras 3 fallas consecutivas, `OpenTimeout` 30 s, 1 request en semiabierto | Breaker propio por proveedor con **tres disparadores** (fallas consecutivas, tasa de error y tasa de llamadas lentas) | §4.3 |
| Fallback local | Gemma 3 1B en CPU vía Ollama, timeout 120 s | Último eslabón de toda cadena: `ollama:gemma3:1b` (CPU), más cadenas por alias | §2, §4 |
| Upstream Groq | `llama-3.3-70b-versatile`, compatible con OpenAI, free tier | `GroqProvider` (free tier) + `OpenAICompatProvider` genérico (OpenRouter, Together, Fireworks, vLLM, llama.cpp) | §3 |
| Streaming SSE compatible con OpenAI | `data: {...}` + `[DONE]`, *write-back* de la caché al terminar el stream | Igual, más cancelación de punta a punta, detección de stalls y errores tipados | §7 |
| Auth Bearer | Comparación en tiempo constante (`subtle.ConstantTimeCompare`), *fail-closed* | Keys **por cliente** guardadas con hash SHA-256 + `hmac.compare_digest` (tiempo constante), *fail-closed* | §8.3 |
| Headers de diagnóstico | `X-Cache-Status` (HIT/MISS/MISS-FALLBACK), `X-Provider`, `X-Cache-Similarity`, `X-Latency-Ms`, `X-Accel-Buffering: no` | Los mismos más `X-Alias`, `X-Fallback-Depth`, `X-Budget-Remaining`, `X-Request-Id`, `X-Cache-Layer` | §2 ADR-14 |
| UI de chat integrada | HTML único embebido con `//go:embed` en `GET /`, selector de modelo, *pill* de estado de caché, streaming | UI de un solo archivo servida por FastAPI en `GET /`, sin build; **agrega** selector de alias, panel de presupuesto, estado de breakers y trazas | §9 |
| Health y stats | `/health` (estado del breaker), `/stats` (caché, breaker, contadores atómicos) | `/healthz`, `/readyz`, `/stats` y métricas Prometheus reales | §8.1 |
| Apagado ordenado | Captura `SIGINT`/`SIGTERM`, 5 s de *graceful shutdown* | *Drain* de streams activos con deadline y liquidación del presupuesto de los streams cortados | §4.9 |
| Config 12-factor | `.env` autocargado | `pydantic-settings` (`GW_*`) + YAML validados al arrancar; recarga en caliente de rutas y keys | §3 |
| Ejemplos de clientes | curl (stream/no stream), Python, Node (fetch y SDK), Go | curl, OpenAI SDK (Python y Node), LangChain/LangGraph, httpx async | §9.2 |
| Tests | 66 tests en 8 paquetes | Unitarios, contratos, integración y caos; cobertura ≥ 85% en `routing`, `resilience`, `budget` y `cache` | §10, §11 |

### 0.2 Qué se corrige (límites encontrados al leer el código)
| Límite del gateway en Go | Riesgo | Corrección en P0 |
|---|---|---|
| La clave semántica es **solo el último mensaje del usuario**; ignora el system prompt, el modelo, la temperatura y la conversación | Respuesta cacheada de otro contexto (falso hit grave) | Namespace por `tenant × alias × hash(system prompt) × versión del prompt × versión del embedder` + elegibilidad por contexto (§6.2) |
| Umbral fijo 0.85 sin calibrar | Con 0.85, "comisión nacional" y "comisión internacional" pueden coincidir | Umbral **calibrado** sobre un set de pares etiquetados para una precisión objetivo ≥ 99%, más guardas de números y negaciones, y verificador opcional en la zona gris (§6.3) |
| Se cachea **cualquier** respuesta (incluso con `temperature > 0` o respuestas truncadas) | Se congelan respuestas creativas o incompletas | Política de elegibilidad: determinismo, `finish_reason == stop`, sin tools, sin datos por usuario (§6.2) |
| El breaker solo cuenta fallas consecutivas | Un proveedor que falla 40% de las veces nunca abre el circuito | Disparadores por tasa de error y por *slow-call rate* en ventana deslizante (§4.3) |
| Un solo upstream + un solo fallback | No hay cadenas por caso de uso | Aliases con cadenas ordenadas y routing según salud (§4.6) |
| Una sola API key global | Sin aislamiento ni límites por cliente | Keys por cliente, rate limit, presupuesto y namespaces de caché por cliente |
| Sin control de costos | Un bucle puede gastar dinero real | Presupuesto atómico con reserva y liquidación (§5.2) |
| Métricas en contadores en memoria; `cache.Stats` usa `DBSIZE` (cuenta todas las keys) | Métricas que no se pueden graficar ni alertar, tamaño de caché inflado | Prometheus + OTel; tamaño por namespace con `FT.INFO` |
| `Reset` hace `FLUSHDB` | Borra presupuesto y rate limits junto con la caché | Invalidación por namespace o por *tag*, nunca `FLUSHDB` |
| `X-Provider` fijo en el código | Diagnóstico engañoso con upstreams intercambiados | El header sale de la cadena real ejecutada |
| Bug histórico: el contexto del stream se cancelaba al retornar el handler (todas las llamadas a Groq abortaban en ~30 ms) | Regresión silenciosa | **Test de contrato** dedicado: un stream largo sobrevive al retorno del handler, y la desconexión del cliente sí lo cancela (§7) |

### 0.3 Afirmaciones del CV que P0 debe **re-medir** con método
| Afirmación del CV (Go) | Cómo se valida en P0 | Prueba |
|---|---|---|
| Hits de caché en 34–90 ms | Latencia de un hit de punta a punta (embed + KNN + respuesta), p50/p95, con los dos embedders | G9 |
| 30–40% menos costo de API | Replay de tráfico realista (Zipf de P3 + P2) con precios de referencia: costo con y sin caché | G11 |
| 99.9% de disponibilidad | Tasa de éxito sobre 10k requests con fallas inyectadas en el proveedor primario (Toxiproxy) | G10 |
| 10K+ req/s en un core | Throughput honesto del camino de caché en Python (uvloop, varios workers) frente a la cifra del binario Go; se publica la diferencia | G13 |

> **Regla de honestidad:** si Python no alcanza la cifra de Go, se publica la cifra real y la explicación (GIL, serialización JSON, costo del embedding). La comparación Go vs Python es un resultado, no un fracaso.

---

## 1. Visión, problema y frase del CV

### 1.1 El problema
Los cuatro proyectos llaman a LLMs: explicaciones de casos (P1), agente System 2 y juez (P2), generación y verificación (P3), agentes de investigación, extracción para el grafo y auditoría visual (P4). Si cada servicio llama a los proveedores por su cuenta:
- **El costo no tiene un punto de control:** no hay tope de presupuesto global, y un bucle o un bug puede gastar dinero real en minutos.
- **Cada servicio reimplementa** reintentos, timeouts, fallback y métricas, cada uno de forma distinta.
- **Cambiar de proveedor** (mock → Ollama → Groq/Gemma → Haiku) implica tocar código en cuatro repos.
- **No hay caché compartida:** la misma pregunta frecuente se paga muchas veces.
- **Los free tiers tienen límites** (requests y tokens por minuto). Sin un punto central que los respete, un servicio agota la cuota de todos.

Un **gateway** centraliza todo eso detrás de una sola API. Productos como LiteLLM o Portkey resuelven el problema, pero este proyecto lo construye **desde cero en Python**: como aprendizaje profundo, con control total del comportamiento, y como pieza de portafolio que conecta los otros cuatro proyectos.

### 1.2 Qué construimos
Un servicio FastAPI asíncrono con:
1. **API compatible con OpenAI:** `/v1/chat/completions` (con y sin streaming), `/v1/embeddings` y `/v1/models`. Cualquier SDK de OpenAI, LangChain o LangGraph lo usa sin cambios.
2. **Proveedores intercambiables:** `mock` (determinista, con inyección de latencia y fallas), `ollama` (local; Gemma 3 1B como último recurso en CPU), `groq` (free tier), `google` (Google AI Studio: **Gemma 4 31B**, *golden evaluator* y visión), `openai_compat` (cualquier endpoint compatible) y `anthropic` (traducción OpenAI ↔ Messages API, incluido el streaming).
3. **Routing por alias:** los servicios piden `fast`, `smart`, `judge`, `judge_golden`, `vision`, `extract` o `smart_paid`; un YAML define la cadena de fallback de cada alias. Selección según salud (EWMA de TTFT y tasa de error).
4. **Resiliencia:** timeouts por fase, reintentos con backoff y jitter (solo antes del primer token), **circuit breaker obligatorio** por proveedor, fallback, bulkheads, **hedged requests**, **concurrencia adaptativa**, **load shedding por prioridad**, *singleflight* y **rate limiting saliente** por proveedor (cuotas de los free tiers).
5. **Control de costos:** tabla de precios, conteo de tokens previo, contabilidad por request, **presupuesto atómico en Redis** (reservar y luego liquidar) con tope por corrida, por cliente y por día, y **prompt caching del proveedor** cuando existe.
6. **Caché exacta + caché semántica obligatoria** en Redis 8 con defensas en capas contra falsos hits (§6) e **invalidación por tags** conectada a los cambios de la KB de P3.
7. **Streaming SSE** con propagación de cancelación, detección de stalls y *replay* de hits como stream.
8. **Salida estructurada:** `response_format` con JSON Schema, validación y un reintento controlado.
9. **Guardrails en el borde:** detección de prompt injection (Llama Prompt Guard 2, 22M, ONNX en CPU), redacción de PII (Presidio) en logs y opcionalmente en el tráfico saliente, límites de tamaño.
10. **Observabilidad:** métricas de Prometheus (TTFT, tokens, costo, caché, breakers, presupuesto, cuotas), spans de OTel con la convención GenAI, Langfuse opcional y headers de diagnóstico.
11. **Seguridad:** API keys por cliente (P1–P4) guardadas con hash y comparadas en tiempo constante, rate limiting y logs sin prompts por defecto.
12. **UI de chat integrada** y ejemplos de clientes, heredados del proyecto en Go.

### 1.3 Métricas de éxito
| Tipo | Métrica | Meta |
|---|---|---|
| Overhead | p95 de latencia agregada por el gateway (con `mock`, sin caché) | < 5 ms |
| Overhead | TTFT agregado al streaming | < 3 ms |
| Caché | **Latencia de un hit semántico** de punta a punta (p95, embedder ONNX en CPU) | < 60 ms (mejor que los 34–90 ms del Go) |
| Caché | **Tasa de falsos hits** en el set de pares "parecidos pero distintos" | < 1% |
| Caché | Hit rate y costo ahorrado sobre tráfico realista (Zipf de P3 + P2) | Se reporta (el CV dice 30–40%) |
| Resiliencia | Requests exitosas con el proveedor primario fallando (G10, 10k requests) | ≥ 99.9% |
| Resiliencia | Reducción del p99 con hedging frente a sin hedging (proveedor con cola larga) | Se reporta |
| Costo | **Nunca** se supera el tope de presupuesto, ni siquiera con 50 requests concurrentes | 0 excesos |
| Cuotas | 0 errores 429 propagados al cliente cuando hay otro proveedor disponible | 0 |
| Seguridad | Recall de prompt injection en un set público, con falsos positivos medidos | Se reporta |
| Compatibilidad | Suite de contratos con los SDKs oficiales de OpenAI (Python y Node) | 100% en verde |

### 1.4 Frase del CV (plantilla)
> **LLM Gateway (Python, from scratch)** · FastAPI · httpx · Redis 8 · ONNX Runtime · OpenTelemetry · Groq · Gemma 4 · Ollama
> OpenAI-compatible async gateway serving four production-style ML systems: multi-provider fallback with per-provider circuit breakers, hedged requests and adaptive concurrency (**{a}% availability under injected failures**), **atomic Redis budget enforcement (0 overspend under concurrency)**, and a layered semantic cache with calibrated thresholds (**{h}% hit rate, {f}% false hits, {s}% cost saved, p95 hit {x} ms**); adds **p95 {o} ms** overhead and streams SSE with end-to-end cancellation.

### 1.5 Por qué desde cero y no LiteLLM
| Criterio | Construir (este proyecto) | LiteLLM |
|---|---|---|
| Aprendizaje y portafolio | Máximo: cada decisión es tuya y explicable | Bajo: se configura |
| Proveedores | 6 adaptadores (mock, ollama, groq, google, openai_compat, anthropic) | Más de 100 |
| Control del presupuesto y de la caché | Total; se diseña para P1–P4 | Configurable, más genérico |
| Mantenimiento | Tuyo | De la comunidad |

LiteLLM queda como **referencia y baseline**: el benchmark de overhead compara ambos en la misma máquina. El gateway en Go queda como **segundo baseline** (cache hit y throughput).

---

## 2. Arquitectura macro, ADRs y flujo de una request

```
 Clientes (OpenAI SDK / LangChain / LangGraph / httpx / UI de chat en GET /)
   P1 explainer · P2 System 2 + juez · P3 RAG API · P4 agentes de investigación
            │  Authorization: Bearer <api-key del cliente>      X-Priority: interactive|batch
            ▼
 ┌──────────────────────────────── llm-gateway (FastAPI, async) ────────────────────────────────┐
 │ auth → rate limit (cliente) → load shedding (prioridad) → guardrails de entrada (injection/PII)│
 │   → resolver alias → caché exacta → caché semántica (namespace → KNN → guardas → verificador) │
 │        │ hit ──────────────────────────────────────────────────────────────► respuesta/stream │
 │        │ miss → singleflight (una sola llamada por clave en vuelo)                           │
 │        ▼                                                                                     │
 │   conteo de tokens → budget.reserve(costo máximo) ──insuficiente──► 402 budget_exceeded      │
 │        ▼                                                                                     │
 │   cadena por alias ordenada por salud (EWMA TTFT, error rate), cada proveedor con:           │
 │   rate limit saliente (RPM/TPM) · bulkhead adaptativo · circuit breaker · timeouts por fase · │
 │   reintentos con jitter · hedging (solo antes del primer token)                              │
 │        ▼                                                                                     │
 │   adapter: mock │ ollama │ groq │ google │ openai_compat │ anthropic                         │
 │        ▼                                                                                     │
 │   budget.settle(real) · cache.store(+tags) · métricas · span OTel · headers X-* · SSE        │
 └──────────────────────────────────────────────────────────────────────────────────────────────┘
        │                       │                                   │
   Redis 8 (budget, rate     Prometheus / OTel Collector /     Invalidación por tags
   limits, cuotas, cachés)   Langfuse (opcional)               (eventos kb-changes de P3)
```

### Decisiones de diseño (ADRs)
- **ADR-1 · Contrato OpenAI hacia afuera, adaptadores hacia adentro.** Los clientes hablan siempre el formato de OpenAI; cada proveedor tiene un adapter que traduce. Agregar un proveedor no toca a ningún cliente.
- **ADR-2 · Alias, no modelos.** Los servicios piden un alias; `config/routes.yaml` decide qué proveedor y modelo hay detrás. Pasar de Ollama a Haiku en la prueba final de P3 es un cambio de configuración.
- **ADR-3 · Presupuesto con reserva y liquidación.** Antes de llamar, se reserva atómicamente el costo **máximo** estimado (tokens de entrada más `max_tokens`) en Redis con Lua. Al terminar, se liquida el costo real y se libera el sobrante.
- **ADR-4 · Reintentar y hacer hedging solo antes del primer token.** Una vez enviados tokens al cliente, reintentar duplicaría el texto. Desde ahí, las fallas se reportan como un evento `error` en el stream.
- **ADR-5 · Caché solo para requests elegibles.** Ver §6.2.
- **ADR-6 · Python async de punta a punta.** httpx async, sin clientes bloqueantes en el camino del streaming. La cancelación de asyncio se propaga desde el cliente hasta el proveedor. Inferencia ONNX (embeddings, guardrails) en un *thread pool* acotado para no bloquear el event loop.
- **ADR-7 · Fallar cerrado en el presupuesto.** Si Redis no responde, las llamadas a proveedores **pagos** se rechazan; los gratuitos siguen funcionando.
- **ADR-8 · Caché semántica obligatoria, con defensas en capas.** La caché semántica está **siempre activa** en los aliases elegibles (no apagada por defecto como en la v1 de este plan). La seguridad no viene de apagarla sino de: namespace estricto, umbral calibrado, guardas léxicas, verificador en la zona gris e invalidación por tags.
- **ADR-9 · Circuit breaker obligatorio con tres disparadores.** Fallas consecutivas (heredado del Go), tasa de error y tasa de llamadas lentas en ventana deslizante. Ningún proveedor se llama sin breaker.
- **ADR-10 · Un adapter genérico compatible con OpenAI.** Groq, Google AI Studio (endpoint compatible), OpenRouter, vLLM o llama.cpp comparten `OpenAICompatProvider`; solo cambian `base_url`, key, cuotas y quirks declarados en YAML. Anthropic y Ollama nativo tienen adapter propio.
- **ADR-11 · Free tiers externos como proveedores de $0, con cuotas como ciudadanos de primera.** Cada proveedor declara RPM/TPM/RPD; el gateway los respeta con token buckets salientes y salta al siguiente de la cadena antes de recibir un 429. A los free tiers solo van **datos sintéticos, públicos o del propio usuario con su consentimiento** (las notas de estudio de P4, filtradas antes por Presidio y por una lista de exclusión); sus términos permiten usar el contenido para mejorar productos.
- **ADR-12 · Hedging con presupuesto.** Se lanza una segunda request a otro proveedor solo si la primera supera el p95 histórico de TTFT del alias, solo en requests sin tokens emitidos y con un tope de hedges (p. ej., ≤ 5% del tráfico). El perdedor se cancela y se liquida lo consumido.
- **ADR-13 · Guardrails en el borde, no en cada servicio.** Detección de injection y redacción de PII viven en el gateway, con modo `monitor` (solo marca) o `enforce` (bloquea) por alias.
- **ADR-14 · Diagnóstico en headers.** Cada respuesta dice qué capa de caché respondió, qué proveedor real atendió, cuántos fallbacks hubo y cuánto presupuesto queda. Heredado y ampliado del Go.
- **ADR-15 · Requisitos en EARS como contrato.** Los requisitos de §11 están escritos en notación EARS y cada uno tiene un test que lo prueba (práctica heredada del harness SDD del proyecto multiagente del CV).

### Flujo de una request en streaming
1. Llega `POST /v1/chat/completions` con `stream: true`, alias `smart` y `X-Priority: interactive`.
2. Se autentica la key, se aplica rate limiting del cliente y, si el gateway está saturado, se descarta primero el tráfico `batch`.
3. Guardrails de entrada: score de injection y redacción de PII (según el modo del alias).
4. Se resuelve la cadena (p. ej., `google:gemma-4-31b` → `groq:llama-3.3-70b` → `ollama:qwen3:1.7b` → `ollama:gemma3:1b`) ordenada por salud.
5. Caché exacta; si no hay hit, caché semántica (§6). Si hay hit, se responde como un stream simulado y el costo es $0.
6. *Singleflight*: si otra request idéntica ya está en vuelo, se espera su resultado.
7. Conteo de tokens y `budget.reserve(costo_máximo)`. Si no alcanza, el error es `budget_exceeded`.
8. Se toma un permiso del rate limit saliente y del bulkhead del proveedor; el breaker decide si se intenta.
9. El adapter abre el stream. Si falla o excede el TTFT antes del primer token, entra el siguiente de la cadena (o un hedge si corresponde).
10. Cada chunk se traduce a `chat.completion.chunk` y se envía por SSE.
11. Al final: uso real, `budget.settle(real)`, almacenamiento en caché si es elegible, métricas, span y headers.
12. Si el cliente se desconecta, se cancela el stream upstream, se liquida lo consumido y se marca `client_disconnect`.

---

## 3. Componentes
> Patrón: **🧠 Concepto → ⚙️ Cómo funciona aquí → 🔧 Detalle técnico → 🔁 Alternativas.**

| Componente | 🧠 / ⚙️ | 🔧 | 🔁 |
|---|---|---|---|
| **API layer** (`api/`) | Endpoints compatibles con OpenAI, admin, health y la UI | FastAPI con modelos Pydantic; `EventSourceResponse` nativo para SSE; errores en el formato de OpenAI; `orjson` para serializar | Starlette pura |
| **Auth y rate limit** (`security/`) | Una key por cliente, con límites de requests y tokens por minuto | Keys con hash SHA-256 en `config/clients.yaml`, `hmac.compare_digest`; token bucket en Redis vía Lua | API gateway externo |
| **Admission control** (`admission/`) | Decide qué entra cuando hay saturación | Clases de prioridad (`interactive` > `batch`); cola acotada; *load shedding* por prioridad con `503 overloaded` + `Retry-After` | Envoy / NGINX |
| **Router** (`routing/`) | Resuelve el alias en una cadena de proveedores ordenada por salud | `routes.yaml` validado con Pydantic al arrancar y recargado en caliente; EWMA de TTFT y error rate por proveedor | Routing por costo o calidad aprendido (future work) |
| **Providers** (`providers/`) | Adapters con una interfaz común: `complete()`, `stream()`, `embed()` | Protocolo `Provider`; `MockProvider` (latencia, TPOT, tasa de fallas y cola larga configurables, determinista por semilla); `OllamaProvider`; `OpenAICompatProvider` (Groq, Google AI Studio, OpenRouter, vLLM); `AnthropicProvider` (traduce mensajes y system, mapea `content_block_delta` a chunks) | SDKs oficiales (más peso) |
| **Cuotas salientes** (`quotas/`) | Respetar RPM/TPM/RPD de cada proveedor | Token buckets en Redis por proveedor y modelo; lectura de headers `x-ratelimit-*` y `retry-after` para ajustar en vivo | Esperar al 429 (lo que hace casi todo el mundo) |
| **Resiliencia** (`resilience/`) | Timeouts, reintentos, breaker, bulkhead, hedging, concurrencia adaptativa | Breaker con ventana deslizante; backoff exponencial con jitter completo; limitador AIMD/gradiente por proveedor; hedging con presupuesto | tenacity / aiobreaker |
| **Budget** (`budget/`) | Reserva y liquidación atómicas | Lua en Redis: `reserve(run, client, max_cost)` y `settle(id, real)`; topes por run, cliente y día; `GET /admin/budget` | Contabilidad en Postgres (más lenta) |
| **Pricing y tokens** (`pricing/`) | Del uso al costo en USD, y estimación previa | `config/prices.yaml` (entrada, salida, cache read/write); conteo previo con tokenizers de HF/`tiktoken`; el `usage` del proveedor manda al liquidar | Tablas de LiteLLM |
| **Caché** (`cache/`) | Exacta + semántica obligatoria con defensas en capas | Ver §6 | GPTCache, Qdrant |
| **Embeddings** (`embeddings/`) | Vectores para la caché y endpoint `/v1/embeddings` | Protocolo `Embedder`: `OnnxEmbedder` (e5-small int8, CPU) y `OllamaEmbedder` (`nomic-embed-text`, heredado); micro-batching | FastEmbed |
| **Structured output** (`structured/`) | Garantizar JSON válido según un schema | `response_format` nativo cuando existe (Ollama `format`, tool-use en Anthropic, JSON mode en Groq/Google); validación con `jsonschema`; un reintento con el error como feedback | Outlines / Instructor del lado cliente |
| **Guardrails** (`guardrails/`) | Injection y PII en el borde | Prompt Guard 2 22M (ONNX, CPU); Presidio (analyzer + anonymizer) con reconocedores en español; modos `off`/`monitor`/`enforce` por alias | NeMo Guardrails, Llama Guard |
| **Telemetry** (`telemetry/`) | Métricas, trazas y logs | `prometheus_client`; OTel con atributos `gen_ai.*`; logs JSON sin prompts (`LOG_PROMPTS=false`); exportador opcional a Langfuse | OpenLIT |
| **UI** (`ui/`) | Demo visual sin escribir código | Un solo `index.html` (HTML + JS vanilla) servido en `GET /` | Gradio (más peso) |

---

## 4. Resiliencia

### 4.1 Timeouts por fase
| Fase | Valor por defecto | Razón |
|---|---|---|
| Conexión | 2 s | Proveedor caído → fallback rápido |
| **Primer token (TTFT)** | 10 s (`smart`), 5 s (`fast`) | Lo que el usuario siente |
| **Entre tokens (stall)** | 10 s | Un stream congelado es una falla |
| Total | 120 s | Deadline duro |
| Embedding de la caché | 300 ms | Si el embedder se atrasa, se salta la caché semántica (nunca bloquea la request) |

### 4.2 Reintentos con jitter
Backoff con jitter completo: espera $= \text{rand}(0, \min(c, b \cdot 2^{n}))$. Así se evita la manada de reintentos sincronizados. Solo se reintentan errores **reintentables** (timeouts de conexión, 429, 5xx) y **solo antes del primer token** (ADR-4). Un 429 con `retry-after` mayor que el TTFT del alias no se reintenta: se pasa al siguiente proveedor.

### 4.3 Circuit breaker por proveedor (obligatorio)
Tres disparadores sobre una ventana deslizante de las últimas $N$ llamadas (o los últimos $W$ segundos):
- **Fallas consecutivas** $\geq k$ (heredado del Go: $k=3$).
- **Tasa de error** $r = \frac{\text{fallas}}{N} > \theta_e$ con un mínimo de llamadas para decidir.
- **Slow-call rate:** fracción de llamadas con TTFT $> t_{slow}$ mayor que $\theta_s$. Un proveedor lento también se trata como degradado.

Estados: **cerrado** → **abierto** (salta directo al siguiente de la cadena) → tras un enfriamiento $T$, **semiabierto** (deja pasar $m$ requests de prueba) → cerrado si salen bien. El enfriamiento crece con backoff si vuelve a abrirse. El estado se expone en métricas, en `/stats`, en la UI y en `/healthz`.

### 4.4 Bulkheads y concurrencia adaptativa
Un límite de concurrencia por proveedor. En vez de un número fijo, un **limitador adaptativo** (AIMD o gradiente, como `concurrency-limits` de Netflix): sube el límite mientras la latencia se mantiene y lo baja cuando crece. Ollama arranca con 2 slots (la GPU de 4 GB no aguanta más). Si se excede, la request espera con un timeout o pasa al siguiente de la cadena.

### 4.5 Hedged requests
Para los aliases interactivos, si una request no recibe el primer token antes del p95 histórico del alias, se lanza una copia contra el siguiente proveedor sano. Gana el primero que emita un token y el otro se cancela. Tope de hedges por ventana (ADR-12) para no duplicar costos ni cuotas. Nunca se hace hedging contra proveedores pagos salvo configuración explícita.

### 4.6 Routing según salud
Dentro de la cadena del alias, el orden base es el del YAML, pero se **saltan** los proveedores con breaker abierto, cuota agotada o EWMA de TTFT muy por encima de su línea base. Así el gateway se adapta sin cambiar la configuración.

### 4.7 Load shedding y singleflight
- **Load shedding:** cuando la cola supera su límite, se rechaza primero el tráfico `batch` (p. ej., el juez de P2 corriendo un set completo) para proteger el `interactive` (el RAG de P3 o la UI).
- **Singleflight:** N requests idénticas en vuelo generan **una** llamada al proveedor; el resto espera ese resultado. Evita la estampida cuando una respuesta popular expira de la caché.

### 4.8 Modos degradados
- Redis caído: sin caché, rate limit local en memoria, y los proveedores pagos rechazan la request (ADR-7).
- Embedder caído o lento: solo caché exacta.
- Todos los proveedores caídos: error `all_providers_unavailable`, que los clientes traducen a su propio fallback (plantilla en P1, abstención en P3).

### 4.9 Apagado ordenado
Con `SIGTERM`: deja de aceptar requests (`/readyz` en 503), espera a que terminen los streams activos hasta un deadline, cancela los restantes con un evento `error` tipado y **liquida el presupuesto** de todo lo consumido.

---

## 5. Control de costos

### 5.1 Contabilidad
$C = (t_{in}\,p_{in} + t_{out}\,p_{out} + t_{cache\_read}\,p_{cr} + t_{cache\_write}\,p_{cw}) / 10^6$, usando los tokens reportados por el proveedor. Los precios están en `prices.yaml` y se verifican antes de la prueba final. Los free tiers cuestan $0 pero tienen **precio de referencia** (lo que costaría pagando), para reportar el ahorro de la caché en dólares comparables.

### 5.2 Presupuesto atómico (reservar y liquidar)
```
reserve: if spent + reserved + max_cost > cap → reject
         else reserved += max_cost; return reservation_id      (Lua, atómico)
settle:  reserved -= max_cost; spent += real_cost              (Lua, atómico)
expira:  las reservas huérfanas (p. ej., un proceso que murió) caducan con TTL y se liberan
```
Los topes son `RUN_BUDGET_USD` (la prueba final de P3, p. ej., US$3), el tope por cliente y el diario. El test de concurrencia dispara 50 requests contra un tope que alcanza para 10, y verifica que pasan como máximo 10 y que el gasto real nunca supera el tope.

### 5.3 Conteo previo de tokens
El `max_cost` de la reserva usa un conteo real de los tokens de entrada (tokenizer del modelo cuando está disponible, aproximación conservadora si no) más `max_tokens`. Requests que excedan el límite de entrada del alias se rechazan antes de llamar (protección contra abuso de costo).

### 5.4 Prompt caching del proveedor
Distinto de la caché del gateway: el proveedor cobra menos por los prefijos de prompt repetidos.
- **Anthropic:** el gateway marca con `cache_control` los bloques estables (system prompt largo, instrucciones del RAG) cuando el alias lo permite. Se contabilizan `cache_creation` y `cache_read` por separado.
- **Otros proveedores:** se ordenan los mensajes para maximizar prefijos estables (lo fijo primero, lo variable al final), lo que favorece el caching implícito cuando existe.
- Métrica: porcentaje de tokens de entrada servidos desde la caché del proveedor.

---

## 6. Caché semántica (obligatoria)

### 6.1 Capas
```
request ─► L1 exacta (SHA-256 de la request normalizada) ──hit──► respuesta
              │ miss
              ▼
           L2 semántica: namespace → embedding → KNN k=5 (HNSW, COSINE, filtro TAG por namespace)
              │ candidato con similitud s
              ├─ s ≥ t_alto ──► guardas léxicas ──ok──► HIT
              ├─ t_bajo ≤ s < t_alto ──► guardas + verificador (cross-encoder) ──ok──► HIT
              └─ s < t_bajo ──► MISS
```

### 6.2 Elegibilidad y namespace
- **Namespace** = `cliente × alias × hash(system prompt) × versión del prompt × modelo del embedder`. Dos sistemas con system prompts distintos nunca comparten respuestas. El namespace se filtra en la misma query vectorial (`@ns:{...}=>[KNN 5 @embedding $vec]`).
- **Texto que se embebe:** la última pregunta del usuario. En conversaciones de varios turnos, la request solo es elegible si el resto del historial coincide por hash (o si el alias declara `semantic_scope: last_turn`, como el FAQ de P3).
- **Elegible para guardar:** `temperature ≤ 0.3` o `cache: true`, sin `tools`, sin datos por usuario (el cliente marca `X-Cache-Scope: private` para excluir), `finish_reason == "stop"` y sin errores.
- **Nunca se cachea:** respuestas truncadas, respuestas de fallback de emergencia (`gemma3:1b`) cuando el alias exige calidad, ni salidas estructuradas que no pasaron la validación.

### 6.3 Defensas contra falsos hits
1. **Umbral calibrado, no adivinado.** Con un set de ~1.000 pares etiquetados ("misma pregunta" / "parecida pero distinta", en español e inglés, a partir de las FAQ de P3 y los mensajes de P2) se traza la curva precisión–cobertura y se eligen `t_alto` (precisión ≥ 99,5% sin verificador) y `t_bajo` (zona gris). Cada embedder tiene sus propios umbrales: no se reutiliza el 0.85 de `nomic-embed-text` con e5.
2. **Guardas léxicas:** si la pregunta nueva y la cacheada difieren en números, montos, fechas, monedas, códigos de producto o en negaciones ("no me llegó" vs "me llegó"), el hit se rechaza aunque la similitud sea alta.
3. **Verificador en la zona gris:** un cross-encoder pequeño (MiniLM, ONNX en CPU) confirma la equivalencia. Solo corre en la zona gris para no sumar latencia a todos los hits.
4. **Medición continua:** la tasa de falsos hits se mide en el set de pares (G6) y por muestreo en tráfico real: el juez de P2 revisa una muestra de hits.

### 6.4 Frescura e invalidación
- **TTL** por alias (heredado: 7 días en el Go; aquí corto para el RAG y largo para FAQs estáticas).
- **Invalidación por tags:** el cliente puede enviar `X-Cache-Tags: doc:123,doc:456` (P3 envía los `doc_id` citados). Cuando P3 publica un cambio de la KB, un consumidor opcional del topic `kb-changes` (o una llamada a `POST /admin/cache/invalidate`) borra las entradas con esos tags. Así **la caché nunca rompe la promesa de frescura de P3**.
- **Sin `FLUSHDB`:** borrado por namespace o por tag; el presupuesto y los rate limits viven en otro prefijo.

### 6.5 Hits en streaming
Un hit se reenvía como stream de chunks con un ritmo configurable (o de una vez), para que el cliente no distinga el camino. El header `X-Cache-Layer: exact|semantic|verified` y `X-Cache-Similarity` lo dicen igual.

### 6.6 Detalle técnico
- Índice: Redis 8 Query Engine, `FT.CREATE` sobre HASH con prefijo `semcache:`, campos `ns` (TAG), `tags` (TAG), `embedding` (VECTOR HNSW, FLOAT32, COSINE, `M=16`, `EF_CONSTRUCTION=200`), `created_at` (NUMERIC), `hits` (NUMERIC).
- Embedder por defecto: e5-small multilingüe en ONNX int8 (CPU, ~10–20 ms por consulta corta). Alternativa medida: `nomic-embed-text` en Ollama (el del Go).
- *Write-back* asíncrono y acotado (cola con límite), como en el Go, para no sumar latencia a la respuesta.
- Memoria: `maxmemory` dedicado y política LFU para las entradas de caché; tamaño por namespace en `/stats`.

---

## 7. Streaming, cancelación y salida estructurada
- **SSE de salida** con el formato `chat.completion.chunk` de OpenAI, cerrado con `data: [DONE]`, compatible con los SDKs. Headers `Cache-Control: no-cache` y `X-Accel-Buffering: no` (heredado).
- **Traducción del streaming de Anthropic:** `message_start`, `content_block_delta` (text_delta), `message_delta` (usage) y `message_stop` se convierten en chunks de OpenAI más el uso final.
- **Cancelación:** si el cliente se desconecta, se cancela la tarea, se cierra el stream httpx upstream y el proveedor deja de generar. Se liquida el costo de los tokens recibidos. Lo valida un test con un servidor real: corta en el token 5 y verifica que el upstream se cerró.
- **Regresión del Go cubierta:** un test verifica que el stream **sobrevive** al retorno del handler (el bug que abortaba todas las llamadas a Groq en ~30 ms).
- **Stall:** si no llega un chunk en `stall_timeout`, se cancela el upstream y se emite un chunk de error.
- **Salida estructurada:** `response_format: {type: json_schema}`. Si la validación falla, se hace un reintento con el error como contexto. Si vuelve a fallar, se responde con un error tipado (`invalid_structured_output`), nunca con JSON roto.
- **Multimodal:** los mensajes con partes `image_url` (base64 o URL) se aceptan en el alias `vision` y se traducen al formato de cada proveedor (Gemma 4 vía AI Studio; Ollama con un modelo con visión). Los demás aliases las rechazan con un error claro.

---

## 8. Observabilidad, seguridad y guardrails

### 8.1 Métricas (Prometheus) y endpoints
`gw_requests_total{alias,provider,status}` · `gw_ttft_seconds{alias,provider}` · `gw_request_seconds` · `gw_tokens_total{provider,direction}` · `gw_cost_usd_total{provider,client}` · `gw_cost_saved_usd_total{layer}` · `gw_cache_requests_total{layer,result}` · `gw_cache_similarity` (histograma) · `gw_breaker_state{provider}` · `gw_budget_remaining_usd{scope}` · `gw_fallbacks_total{from,to}` · `gw_hedges_total{outcome}` · `gw_quota_remaining{provider,kind}` · `gw_shed_total{priority}` · `gw_concurrency_limit{provider}` · `gw_guardrail_events_total{type,action}` · `gw_stream_outcomes_total{outcome}`. Solo labels acotados: nunca prompts ni IDs de usuario.

Endpoints: `/healthz` (vivo), `/readyz` (Redis y al menos un proveedor disponible), `/stats` (JSON heredado del Go: caché, breakers, presupuesto, cuotas), `/metrics`.

### 8.2 Trazas
Un span por request con atributos `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.*` y spans hijos por capa de caché, por guardrail y por intento de proveedor (así se ven los fallbacks y los hedges). Exportación OTLP hacia el collector, Tempo o Langfuse (todo opcional). El `trace_id` vuelve en `X-Request-Id`.

### 8.3 Seguridad
- Keys por cliente con hash SHA-256 y comparación en tiempo constante; rotación vía archivo y recarga sin downtime. *Fail-closed* si falta la configuración.
- Secrets de proveedores solo por variables de entorno (Secret Manager en Cloud Run).
- `LOG_PROMPTS=false` por defecto; si se activa, pasan por la redacción de Presidio.
- Límites de tamaño de request (tokens de entrada, imágenes) para evitar abusos de costo.
- CORS deshabilitado salvo para la UI servida por el mismo origen.

### 8.4 Guardrails
| Guardrail | Cómo | Modo por defecto | Quién lo usa más |
|---|---|---|---|
| Prompt injection / jailbreak | Llama Prompt Guard 2 22M en ONNX (CPU), sobre mensajes `user` y sobre contenido marcado como externo (`X-Untrusted-Content`) | `monitor` (marca y mide) | **P4**: el contenido web (SearXNG + Crawl4AI) es la superficie de ataque principal |
| PII | Presidio (emails, teléfonos, tarjetas, documentos de identidad; reconocedores en español) | Redacción en logs y trazas; opcional en el tráfico hacia free tiers | P1–P4 |
| Tamaño y forma | Límites por alias | `enforce` | Todos |

Los falsos positivos se miden (G14) antes de pasar cualquier guardrail a `enforce`.

---

## 9. UI de chat, ejemplos y experiencia de desarrollo

### 9.1 UI integrada (heredada del Go y ampliada)
Un único `index.html` (HTML, CSS y JS vanilla, sin build, sin dependencias externas) servido en `GET /`:
- Campo de API key (guardada en `localStorage`, enmascarada), **selector de alias** y de modelo.
- Streaming con cursor y errores visibles (si llega un chunk de error, se muestra en rojo en vez de quedar en blanco).
- *Pills* con los headers reales: capa de caché, similitud, proveedor, profundidad de fallback, latencia y presupuesto restante.
- Panel lateral con el estado de los breakers y las cuotas (leído de `/stats`).
- Atajos: `Enter` envía, `Shift+Enter` agrega una línea; botón `Clear`.

### 9.2 Ejemplos de clientes (`examples/`)
`curl_nonstream.sh`, `curl_stream.sh`, `python_openai_sdk.py`, `python_httpx_async.py`, `langgraph_client.py`, `node_fetch.mjs` y `node_openai_sdk.mjs`, todos contra el mismo endpoint.

### 9.3 DX
`make up` (gateway + Redis), `make demo-breaker` (mata el primario y muestra la conmutación en la UI), `make demo-cache` (misma pregunta dos veces y una "parecida pero distinta"), troubleshooting en el README (heredado: key que no persiste entre terminales, Redis caído, embedder caído, buffering de proxies).

---

## 10. Evaluación
| ID | Prueba | Cómo | Resultado esperado |
|---|---|---|---|
| G1 | **Overhead** | Carga open-loop con k6 contra `mock` con latencia 0, directo vs a través del gateway; p50/p95/p99 | p95 agregado < 5 ms |
| G2 | Overhead vs LiteLLM | Mismo test con LiteLLM proxy | Comparación honesta |
| G3 | **Budget bajo concurrencia** | 50 requests concurrentes con un tope que alcanza para 10 | 0 excesos |
| G4 | Fallback | `mock` primario con 100% de fallas → secundario | ≥ 99% de éxito; breaker abierto en métricas |
| G5 | Cancelación | El cliente corta en el token 5; además, test de regresión del bug de contexto del Go | Upstream cancelado; tokens liquidados ≈ consumidos; stream largo sobrevive |
| G6 | **Caché semántica: calidad** | Set de ~1.000 pares etiquetados (ES/EN) | Curva precisión–cobertura por embedder; **tasa de falsos hits < 1%** en el umbral elegido |
| G7 | Contratos | Suite con los SDKs oficiales de OpenAI (streaming, no streaming, `response_format`, embeddings, errores) | 100% en verde |
| G8 | Caos | Redis caído, embedder caído, Ollama caído y lento (Toxiproxy) | Comportamiento según §4.8 |
| G9 | **Latencia de un hit** | p50/p95 de un hit exacto y semántico, con e5 ONNX y con `nomic-embed-text` | p95 semántico < 60 ms; comparado con los 34–90 ms del Go |
| G10 | **Disponibilidad** | 10k requests con fallas intermitentes, latencia de cola y 429 inyectados en el primario | ≥ 99.9% de éxito |
| G11 | **Costo ahorrado** | Replay de tráfico Zipf (FAQ de P3 + mensajes de P2) con precios de referencia | Hit rate y % de costo ahorrado (el CV dice 30–40%) |
| G12 | Hedging | Proveedor `mock` con cola larga (p99 alto) | p99 con hedging vs sin hedging; % de hedges y costo extra |
| G13 | Throughput | Camino de caché exacta con 1, 2 y 4 workers (uvloop) | req/s sostenidos en este hardware, comparados con la cifra del Go |
| G14 | Guardrails | Set público de prompt injection + prompts benignos en español | Recall y tasa de falsos positivos por umbral |
| G15 | Cuotas | Free tier simulado con RPM bajo | 0 errores 429 propagados mientras haya otro proveedor |

---

## 11. Requisitos EARS (contrato de aceptación)
Cada requisito tiene un test con su ID (`test_req_XX_*`). El README incluye la tabla de trazabilidad requisito → test.

| ID | Requisito |
|---|---|
| R01 | WHEN un cliente envía una request con una key válida THEN el gateway SHALL responder en el formato de OpenAI con los headers `X-Provider`, `X-Alias` y `X-Request-Id`. |
| R02 | IF la key falta o no coincide THEN el gateway SHALL responder 401 sin llamar a ningún proveedor. |
| R03 | WHEN el proveedor primario de un alias falla antes del primer token THEN el gateway SHALL intentar el siguiente proveedor de la cadena. |
| R04 | WHILE el breaker de un proveedor está abierto, el gateway SHALL saltarse ese proveedor sin llamarlo. |
| R05 | IF la reserva de presupuesto excede el tope THEN el gateway SHALL responder `budget_exceeded` sin llamar al proveedor. |
| R06 | El gateway SHALL garantizar que el gasto liquidado nunca supere el tope, con cualquier nivel de concurrencia. |
| R07 | WHEN una request elegible coincide con una entrada semántica sobre el umbral calibrado y pasa las guardas THEN el gateway SHALL responder desde la caché con `X-Cache-Layer` y `X-Cache-Similarity`. |
| R08 | IF la request candidata difiere de la cacheada en números, montos, fechas o negaciones THEN el gateway SHALL tratarla como miss. |
| R09 | WHEN llega una invalidación para un tag THEN el gateway SHALL dejar de servir toda entrada con ese tag. |
| R10 | WHEN el cliente se desconecta durante un stream THEN el gateway SHALL cancelar la llamada upstream y liquidar los tokens consumidos. |
| R11 | IF ya se emitieron tokens al cliente THEN el gateway SHALL NOT reintentar ni hacer hedging; SHALL emitir un evento `error` tipado. |
| R12 | IF Redis no está disponible THEN el gateway SHALL rechazar los proveedores pagos y SHALL seguir sirviendo los gratuitos sin caché. |
| R13 | WHILE la cuota de un proveedor está agotada, el gateway SHALL enrutar al siguiente proveedor en lugar de devolver 429. |
| R14 | WHEN la cola de admisión está llena THEN el gateway SHALL descartar primero las requests `batch`. |
| R15 | IF se pide `response_format` con JSON Schema y la salida no valida tras un reintento THEN el gateway SHALL responder `invalid_structured_output`. |
| R16 | El gateway SHALL NOT escribir prompts ni respuestas en logs mientras `LOG_PROMPTS=false`. |

---

## 12. Recursos, ejecución y despliegue
- **RAM:** el gateway ~200 MB, el embedder ONNX ~150 MB, Prompt Guard ~100 MB, el verificador ~100 MB. Redis se comparte con el proyecto que corra en ese momento. Ollama corre nativo en Windows (`host.docker.internal:11434`).
- **Modos de ejecución:** (1) **contenedor en el Compose de cada proyecto** (el perfil que necesite LLM incluye el servicio `llm-gateway` desde su imagen); (2) **standalone** con `docker compose up` en este repo, para desarrollo, demos con la UI y benchmarks; (3) **Cloud Run** en la prueba final de P3 (1 instancia como máximo, ingress interno, keys en Secret Manager).
- **Imagen:** multi-stage con Python 3.12 slim y uv; los modelos ONNX se descargan en el build (capa cacheada) o al primer arranque (documentado).
- **Keys externas ($0):** `GW_GROQ_API_KEY` y `GW_GOOGLE_API_KEY` (free tiers). Sin ellas, las cadenas siguen funcionando con Ollama y `mock`.

---

## 13. Estructura del repo y del README
```
llm-gateway/
├── README.md · PLAN.md · LICENSE · Makefile · docker-compose.yml · Dockerfile · .env.example · pyproject.toml
├── src/llm_gateway/
│   ├── main.py · settings.py
│   ├── api/ (chat.py, embeddings.py, models.py, admin.py, health.py, schemas.py)
│   ├── providers/ (base.py, mock.py, ollama.py, openai_compat.py, anthropic.py)
│   ├── routing/ · admission/ · quotas/ · resilience/ · budget/ (lua/) · pricing/
│   ├── cache/ (exact.py, semantic.py, guards.py, verifier.py, invalidation.py)
│   ├── embeddings/ · structured/ · guardrails/ · security/ · telemetry/ · ui/ (index.html)
├── config/ (routes.yaml, prices.yaml, providers.yaml, clients.example.yaml)
├── examples/ (curl, Python, Node, LangGraph)
├── tests/ (unit/, contract/, integration/, chaos/) — cada requisito EARS con su test
├── bench/ (overhead.py, k6/, cache_pairs/, traffic_replay.py)
└── docs/ (adr/, results/, images/)
```
**Esqueleto del README (en inglés):** Part I (problem; core concepts: gateways, circuit breakers, budgets, semantic caching and false hits, hedging, SSE), Part II (components), Part III (resilience, cost control and the semantic cache in depth), Part IV (proof: G1–G15, EARS traceability, comparison with the Go predecessor and LiteLLM), Part V (run it: UI demo, standalone, inside P1–P4, Cloud Run), Part VI (reflection: lessons, limitations, Go vs Python, why not LiteLLM).

---

## 14. Hitos de implementación
| Hito | Objetivo | Criterio de aceptación |
|---|---|---|
| **M0 · Bootstrap** ✅ | Repo, estructura, configs, CI, README esqueleto | Hecho en el setup inicial |
| **M1 · Núcleo + mock + UI** | `/v1/chat/completions` (normal y streaming) con `MockProvider`, schemas OpenAI, health, `/stats`, headers `X-*`, **UI de chat** | El SDK oficial de OpenAI funciona contra el gateway (G7 parcial); la UI muestra streaming y headers |
| **M2 · Proveedores + routing** | `OllamaProvider` (incl. fallback `gemma3:1b`), `OpenAICompatProvider` (Groq, Google AI Studio), `routes.yaml`, aliases, cuotas salientes | P1, P2 y P4 pueden apuntar al gateway con `fast` y `smart`; G15 |
| **M3 · Resiliencia** | Timeouts por fase, reintentos, **breaker con 3 disparadores**, bulkhead adaptativo, fallback, singleflight, load shedding, apagado ordenado | G4, G8 parcial; R03, R04, R13, R14 |
| **M4 · Caché semántica** | Exacta + semántica obligatoria (namespace, umbral calibrado, guardas, verificador, tags), `/v1/embeddings`, los dos embedders | **G6, G9 y G11 medidos**; R07–R09 |
| **M5 · Budget + pricing** | Lua de reserva y liquidación, conteo previo, topes, `/admin/budget`, prompt caching | **G3 en verde**; R05, R06 |
| **M6 · Streaming robusto + hedging** | Cancelación, stalls, `[DONE]`, errores tipados, regresión del Go, hedging con presupuesto | G5, G12; R10, R11 |
| **M7 · Structured + Anthropic + visión** | `response_format`, validación, `AnthropicProvider` con streaming, partes de imagen en `vision` | Probado **solo contra mocks grabados** para Anthropic ($0); visión probada con Gemma 4 en AI Studio |
| **M8 · Observabilidad, seguridad y guardrails** | OTel GenAI, dashboards, keys por cliente, rate limit, Prompt Guard, Presidio | Dashboard del gateway en Grafana; G14; R16 |
| **M9 · Benchmarks + publicación** | G1, G2, G10, G13, README completo con trazabilidad EARS, `v1.0` | Frase del CV con números; comparación con el Go publicada |

**Orden respecto de los otros proyectos:** M1–M2 antes del M7 de P1 (explainer), del M5 de P2 (System 2) y del M2 de P4 (agentes). M3–M7 antes de la prueba final de P3. M4 (caché semántica) antes de la medición de costo de P3 (M6 de P3).

---

## 15. Riesgos y pendientes
| Riesgo | Mitigación |
|---|---|
| Subestimar la traducción del streaming de Anthropic | Fixtures grabados de streams reales (cuando se haga la prueba final) más tests de contrato; empezar por el modo no streaming |
| **Falsos hits de la caché semántica** (respuestas financieras incorrectas en P3) | Defensas en capas de §6.3, invalidación por tags de §6.4, G6 como compuerta en la CI: si la tasa de falsos hits supera 1%, la CI falla |
| La búsqueda vectorial de Redis 8 no está en la imagen o versión usada | Verificar en M4; plan B: Redis Stack (como el Go) o Qdrant (ya está en P3 y P4) |
| Cambios en los free tiers (Groq, Google AI Studio): límites, modelos o términos | Cuotas en YAML, cadenas con fallback local; verificar al implementar y antes de cada demo |
| Python no alcanza el throughput del Go | Se publica la cifra real (G13) y el análisis; el valor del proyecto está en la robustez, no en los req/s |
| Hedging duplica costos o agota cuotas | Tope de hedges por ventana, nunca contra proveedores pagos por defecto, métricas de costo extra |
| Guardrails con falsos positivos en español | Modo `monitor` por defecto; G14 antes de `enforce` |
| Precios desactualizados | `prices.yaml` versionado; verificarlos antes de la prueba final |
| Scope creep (es tentador construir un LiteLLM entero) | Solo 6 adapters; las features más allá de §1.2 van a "future work" |

**Por verificar al implementar:** el endpoint OpenAI-compatible de Ollama (streaming, `format` y visión), el endpoint compatible con OpenAI de Google AI Studio para Gemma 4 (streaming, JSON mode, imágenes) y sus cuotas del free tier, los modelos y cuotas vigentes de Groq, la forma actual de los eventos de streaming de Anthropic y de `cache_control`, las APIs de vector search de Redis 8, Prompt Guard 2 exportable a ONNX, y FastAPI SSE nativo con `POST` (ya verificado en 0.142).

**Future work:** routing aprendido por calidad y costo (un clasificador que elige el modelo más barato suficiente, idea que P2 explora a nivel de aplicación), gateway MCP (exponer herramientas además de modelos), WebSockets, multi-región.
