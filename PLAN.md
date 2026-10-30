# 🛡️ P0 — LLM Gateway (Python, construido desde cero)

> **Plan de proyecto, v2.** **Estado:** ✅ Plan completo · 🟡 Setup M0 hecho · **v2 (2026-10-06): absorbe por completo el "High-Performance LLM Edge Gateway" (Go) del CV** y lo supera en robustez. Reemplaza a ese proyecto en P1, P2, P3 y P4. **v2.1 (2026-10-29):** decisiones de la sesión de revisión (*grilling*) aplicadas en todo el plan; alcance core/stretch en §1.6; términos fijos en [`GLOSSARY.md`](GLOSSARY.md); ADR-0003 a ADR-0006 en `docs/adr/`.
> Stack: Python 3.12 · FastAPI + uvicorn (uvloop, httptools) · httpx (async, HTTP/2) · Pydantic v2 · orjson · Redis 8 (budget, rate limit, caché exacta y **semántica obligatoria** con vector search HNSW) · ONNX Runtime (embeddings y guardrails en CPU) · Ollama · Groq · Google AI Studio (Gemma 4) · Anthropic (Claude Haiku 5.5) · Prometheus + OpenTelemetry (GenAI semconv) · Langfuse · Presidio · Toxiproxy · k6 · Docker Compose · Cloud Run + Redis Cloud (solo en la prueba final de P3)
> Gasto: **$0** en desarrollo (`mock`, `ollama` y los free tiers de Groq y Google AI Studio). Anthropic (`claude-haiku-5-5`) solo en la prueba final de P3, con tope duro de presupuesto en el gateway **y** límite de gasto en la consola de Anthropic.

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
| Upstream Groq | `llama-3.3-70b-versatile`, compatible con OpenAI, free tier | `OpenAICompatProvider` genérico configurado para Groq (free tier); el mismo adapter sirve a Ollama, Google AI Studio, OpenRouter, vLLM y llama.cpp | §3 |
| Streaming SSE compatible con OpenAI | `data: {...}` + `[DONE]`, *write-back* de la caché al terminar el stream | Igual, más cancelación de punta a punta, detección de stalls y errores tipados | §7 |
| Auth Bearer | Comparación en tiempo constante (`subtle.ConstantTimeCompare`), *fail-closed* | Keys **por cliente** guardadas con hash SHA-256 + `hmac.compare_digest` (tiempo constante), *fail-closed* | §8.3 |
| Headers de diagnóstico | `X-Cache-Status` (HIT/MISS/MISS-FALLBACK), `X-Provider`, `X-Cache-Similarity`, `X-Latency-Ms`, `X-Accel-Buffering: no` | Los mismos (con nombres `X-Gateway-*`) más `X-Alias`, `X-Gateway-Fallback-Depth`, `X-Request-Id`, `X-Cache-Layer` y los avisos de ajuste (`X-Max-Tokens-Clamped`, `X-Params-Dropped`, `X-Budget-Skipped`); el presupuesto se consulta en `/admin/budget` | §8.5 |
| UI de chat integrada | HTML único embebido con `//go:embed` en `GET /`, selector de modelo, *pill* de estado de caché, streaming | UI de un solo archivo servida por FastAPI en `GET /`, sin build; **agrega** selector de alias, panel de presupuesto, estado de breakers y trazas | §9 |
| Health y stats | `/health` (estado del breaker), `/stats` (caché, breaker, contadores atómicos) | `/healthz`, `/readyz`, `/stats` y métricas Prometheus reales | §8.1 |
| Apagado ordenado | Captura `SIGINT`/`SIGTERM`, 5 s de *graceful shutdown* | *Drain* de streams activos con deadline y liquidación del presupuesto de los streams cortados | §4.9 |
| Config 12-factor | `.env` autocargado | `pydantic-settings` (`GW_*`) + YAML validados al arrancar; recarga en caliente de rutas y keys (stretch) | §3 |
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
3. **Routing por alias:** los servicios piden `fast`, `smart`, `judge`, `judge_golden`, `vision`, `extract` o `smart_paid`; un YAML define la cadena de fallback de cada alias. Se saltan los proveedores con breaker abierto o cuota agotada; la EWMA de TTFT se mide (reordenar por ella es stretch).
4. **Resiliencia:** timeouts por fase, reintentos con backoff y jitter (solo antes del primer token), **circuit breaker obligatorio** por proveedor, fallback, bulkheads fijos, **load shedding por prioridad**, *singleflight*, **rate limiting saliente** por proveedor (cuotas de los free tiers) y apagado ordenado. *Stretch:* **hedged requests** y **concurrencia adaptativa**.
5. **Control de costos:** tabla de precios, conteo de tokens previo, contabilidad por request, **presupuesto atómico en Redis** (reservar y luego liquidar, en micro-dólares enteros) con tope por corrida, por cliente y por día, y **prompt caching de Anthropic** (`cache_control`) desde el core.
6. **Caché exacta + caché semántica obligatoria** en Redis 8 con defensas en capas contra falsos hits (§6) e **invalidación por tags** conectada a los cambios de la KB de P3.
7. **Streaming SSE** con propagación de cancelación, detección de stalls y *replay* de hits como stream.
8. **Salida estructurada:** `response_format` con JSON Schema, validación y un reintento controlado.
9. **Guardrails en el borde:** límites de tamaño (core). *Stretch:* detección de prompt injection (Llama Prompt Guard 2, 22M, ONNX en CPU) y redacción de PII (Presidio) en logs y opcionalmente en el tráfico saliente.
10. **Observabilidad:** métricas de Prometheus (TTFT, tokens, costo, caché, breakers, presupuesto, cuotas), spans mínimos de OTel con la convención GenAI (opcionales), un log JSON por request y headers de diagnóstico. *Stretch:* Langfuse.
11. **Seguridad:** API keys por cliente (P1–P4) guardadas con hash y comparadas en tiempo constante, rate limiting y logs sin prompts por defecto.
12. **UI de chat integrada** y ejemplos de clientes, heredados del proyecto en Go.

### 1.3 Métricas de éxito
| Tipo | Métrica | Meta |
|---|---|---|
| Overhead | p95 de latencia agregada por el gateway (con `mock`, sin caché) | < 5 ms |
| Overhead | TTFT agregado al streaming | < 3 ms |
| Overhead | Latencia agregada en un **miss** con la caché semántica activa (embedding + KNN) | Se reporta junto al número sin caché (G1) |
| Caché | **Latencia de un hit semántico** de punta a punta (p95, embedder ONNX en CPU) | < 60 ms (mejor que los 34–90 ms del Go) |
| Caché | **Tasa de falsos hits** en el set de pares "parecidos pero distintos" | < 1% |
| Caché | Hit rate y costo ahorrado sobre tráfico realista (Zipf de P3 + P2) | Se reporta (el CV dice 30–40%) |
| Resiliencia | Requests exitosas con el proveedor primario fallando (G10, 10k requests) | ≥ 99.9% |
| Resiliencia | Reducción del p99 con hedging frente a sin hedging (proveedor con cola larga) | Se reporta (stretch) |
| Costo | **Nunca** se supera el tope de presupuesto, ni siquiera con 50 requests concurrentes | 0 excesos |
| Cuotas | 0 errores 429 propagados al cliente cuando hay otro proveedor disponible | 0 |
| Seguridad | Recall de prompt injection en un set público, con falsos positivos medidos | Se reporta (stretch) |
| Compatibilidad | Suite de contratos con los SDKs oficiales de OpenAI (Python y Node) | 100% en verde |

### 1.4 Frase del CV (plantilla)
> **LLM Gateway (Python, from scratch)** · FastAPI · httpx · Redis 8 · ONNX Runtime · Prometheus · Groq · Gemma 4 · Ollama · Claude Haiku 5.5
> OpenAI-compatible async gateway serving four production-style ML systems: multi-provider fallback with per-provider circuit breakers (**{a}% availability under injected failures**), **atomic Redis budget enforcement (0 overspend under concurrency)**, and a layered semantic cache with calibrated thresholds (**{h}% hit rate, {f}% false hits, {s}% cost saved, p95 hit {x} ms**); adds **p95 {o} ms** overhead and streams SSE with end-to-end cancellation.

La frase usa **solo lo que entra en el core** (§1.6). Si un stretch se completa y se mide, se agrega su número (p. ej., "hedged requests cut p99 by {p}%").

### 1.5 Por qué desde cero y no LiteLLM
| Criterio | Construir (este proyecto) | LiteLLM |
|---|---|---|
| Aprendizaje y portafolio | Máximo: cada decisión es tuya y explicable | Bajo: se configura |
| Proveedores | 6 adaptadores (mock, ollama, groq, google, openai_compat, anthropic) | Más de 100 |
| Control del presupuesto y de la caché | Total; se diseña para P1–P4 | Configurable, más genérico |
| Mantenimiento | Tuyo | De la comunidad |

LiteLLM queda como **referencia y baseline opcional** (G2): el benchmark de overhead compara ambos en la misma máquina. El gateway en Go queda como **segundo baseline** (cache hit y throughput).

### 1.6 Alcance: core y stretch
| Core (entra en la frase del CV) | Stretch (solo si sobra tiempo; no bloquea la v1.0) |
|---|---|
| M1–M5 completos: contrato OpenAI, proveedores, routing, resiliencia (sin hedging ni AIMD), caché exacta y semántica, budget y pricing | Hedging (ADR-12, G12) |
| Cancelación de streams SSE de punta a punta (M6 sin hedging) | Concurrencia adaptativa AIMD/gradiente (§4.4) |
| Adapter nativo de Anthropic con **prompt caching** (ADR-0005) | Reordenar la cadena por EWMA de TTFT (§4.6) |
| Métricas de Prometheus, log JSON por request y trazas OTel mínimas opcionales | Prompt Guard 2 y Presidio (§8.4, G14); Langfuse |
| Keys por cliente, `allowed_aliases`, rate limit y límites de tamaño | Visión (`vision`), recarga en caliente de rutas y keys |

---

## 2. Arquitectura macro, ADRs y flujo de una request

```
 Clientes (OpenAI SDK / LangChain / LangGraph / httpx / UI de chat en GET /)
   P1 explainer · P2 System 2 + juez · P3 RAG API · P4 agentes de investigación
            │  Authorization: Bearer <api-key del cliente>      X-Priority: interactive|batch
            ▼
 ┌──────────────────────────────── llm-gateway (FastAPI, async) ────────────────────────────────┐
 │ auth → rate limit (cliente) → load shedding (prioridad) → límites de tamaño → guardrails*    │
 │   → resolver alias (404 / 403) → caché exacta → caché semántica (namespace → KNN → guardas → │
 │     verificador)  │ hit ─────────────────────────────────────────────────► respuesta/stream  │
 │                   │ miss → singleflight (una sola llamada por clave en vuelo)                │
 │                   ▼                                                                          │
 │   cadena del alias; se saltan los proveedores con breaker abierto o cuota agotada.           │
 │   Si el proveedor es pago: count_tokens → budget.reserve(costo máximo)                       │
 │        ──no alcanza──► siguiente proveedor (402 budget_exceeded si no queda ninguno)         │
 │   Por proveedor: rate limit saliente (RPM/TPM) · bulkhead · circuit breaker ·                │
 │   timeouts por fase · reintentos con jitter · hedging* (solo antes del primer token)         │
 │        ▼                                                                                     │
 │   adapter: mock │ openai_compat (ollama, groq, google) │ anthropic                           │
 │        ▼                                                                                     │
 │   budget.settle(real) · cache.store(+tags) · métricas · span OTel · headers X-* · SSE        │
 │   * stretch (§1.6)                                                                           │
 └──────────────────────────────────────────────────────────────────────────────────────────────┘
        │                       │                                   │
   Redis 8 (budget, rate     Prometheus / OTel Collector /     Invalidación por tags
   limits, cuotas, cachés)   Langfuse (opcional)               (eventos kb-changes de P3)
```

### Decisiones de diseño (ADRs)
- **ADR-1 · Contrato OpenAI hacia afuera, adaptadores hacia adentro.** Los clientes hablan siempre el formato de OpenAI; cada proveedor tiene un adapter que traduce. Agregar un proveedor no toca a ningún cliente.
- **ADR-2 · Alias, no modelos.** Los servicios piden un alias; `config/routes.yaml` decide qué proveedor y modelo hay detrás. Pasar de Ollama a Haiku 5.5 en la prueba final de P3 es un cambio de configuración.
- **ADR-3 · Presupuesto con reserva y liquidación.** Antes de llamar a un proveedor **pago**, se reserva atómicamente el costo **máximo** estimado (tokens de entrada más `max_tokens`) en Redis con Lua, en micro-dólares enteros. Al terminar, se liquida el costo real y se libera el sobrante. Detalle en [ADR-0004](docs/adr/0004-budget-accounting.md).
- **ADR-4 · Reintentar y hacer hedging solo antes del primer token.** Una vez enviados tokens al cliente, reintentar duplicaría el texto. Desde ahí, las fallas se reportan como un evento `error` en el stream.
- **ADR-5 · Caché solo para requests elegibles.** Ver §6.2.
- **ADR-6 · Python async de punta a punta.** httpx async, sin clientes bloqueantes en el camino del streaming. La cancelación de asyncio se propaga desde el cliente hasta el proveedor. Inferencia ONNX (embeddings, guardrails) en un *thread pool* acotado para no bloquear el event loop.
- **ADR-7 · Fallar cerrado en el presupuesto.** Si Redis no responde, las llamadas a proveedores **pagos** se rechazan; los gratuitos siguen funcionando, sin caché y con un rate limit aproximado en memoria del proceso.
- **ADR-8 · Caché semántica obligatoria, con defensas en capas.** La caché semántica está **siempre activa** en los aliases elegibles (no apagada por defecto como en la v1 de este plan). La seguridad no viene de apagarla sino de: namespace estricto, umbral calibrado, guardas léxicas, verificador en la zona gris e invalidación por tags.
- **ADR-9 · Circuit breaker obligatorio con tres disparadores.** Fallas consecutivas (heredado del Go), tasa de error y tasa de llamadas lentas en ventana deslizante. Ningún proveedor se llama sin breaker.
- **ADR-10 · Un adapter genérico compatible con OpenAI.** Groq, Google AI Studio (endpoint compatible), OpenRouter, vLLM o llama.cpp comparten `OpenAICompatProvider`; solo cambian `base_url`, key, cuotas y quirks declarados en YAML. **Ollama también usa este adapter** (endpoint `/v1`); solo tendrá adapter nativo si en M2 aparece un hueco real (streaming, `usage` o `response_format`). Anthropic tiene adapter propio ([ADR-0005](docs/adr/0005-anthropic-adapter-haiku-5-5.md)).
- **ADR-11 · Free tiers externos como proveedores de $0, con cuotas como ciudadanos de primera.** Cada proveedor declara RPM/TPM/RPD; el gateway los respeta con token buckets salientes y salta al siguiente de la cadena antes de recibir un 429. A los free tiers solo van **datos sintéticos, públicos o del propio usuario con su consentimiento** (las notas de estudio de P4, filtradas antes por Presidio y por una lista de exclusión); sus términos permiten usar el contenido para mejorar productos.
- **ADR-12 · Hedging con presupuesto (stretch).** Se lanza una segunda request a otro proveedor solo si la primera supera el p95 histórico de TTFT del alias, solo en requests sin tokens emitidos y con un tope de hedges (p. ej., ≤ 5% del tráfico). El perdedor se cancela y se liquida lo consumido.
- **ADR-13 · Guardrails en el borde, no en cada servicio.** Detección de injection y redacción de PII viven en el gateway, con modo `monitor` (solo marca) o `enforce` (bloquea) por alias. Los límites de tamaño son core; injection y PII son stretch.
- **ADR-14 · Diagnóstico en headers.** Cada respuesta dice qué capa de caché respondió, qué proveedor real atendió, cuántos fallbacks hubo y qué ajustó el gateway (clamp, parámetros quitados, proveedor pago saltado). El presupuesto restante **no** va en headers: se consulta en `/admin/budget`. Lista completa en §8.5.
- **ADR-15 · Requisitos en EARS como contrato.** Los requisitos de §11 están escritos en notación EARS y cada uno tiene un test que lo prueba (práctica heredada del harness SDD del proyecto multiagente del CV).

**ADRs en archivo** (`docs/adr/`): [0002](docs/adr/0002-semantic-cache-own-implementation-not-langcache.md) caché semántica propia sobre Redis 8 · [0003](docs/adr/0003-single-worker-in-process-state.md) un worker y estado de resiliencia en memoria del proceso · [0004](docs/adr/0004-budget-accounting.md) contabilidad del presupuesto · [0005](docs/adr/0005-anthropic-adapter-haiku-5-5.md) adapter de Anthropic para Haiku 5.5 · [0006](docs/adr/0006-p3-final-deployment-cloud-run-redis-cloud.md) despliegue de la prueba final de P3.

### Flujo de una request en streaming
1. Llega `POST /v1/chat/completions` con `stream: true`, alias `smart` y `X-Priority: interactive`.
2. Se autentica la key, se aplica rate limiting del cliente y, si el gateway está saturado, se descarta primero el tráfico `batch`. Se aplican los límites de tamaño y el clamp de `max_tokens`.
3. Guardrails de entrada (stretch): score de injection y redacción de PII (según el modo del alias).
4. Se resuelve el alias (404 si no existe, 403 si el cliente no lo tiene permitido) y su cadena (p. ej., `google:gemma-4-31b` → `groq:llama-3.3-70b` → `ollama:qwen3:1.7b` → `ollama:gemma3:1b`), saltando proveedores con breaker abierto o cuota agotada.
5. Caché exacta; si no hay hit, caché semántica (§6). Si hay hit, se responde como un stream simulado y el costo es $0.
6. *Singleflight*: si otra request idéntica y cacheable ya está en vuelo, se espera su resultado.
7. Solo si el proveedor es pago: conteo de tokens y `budget.reserve(costo_máximo)`. Si no alcanza, se salta al siguiente proveedor de la cadena; `budget_exceeded` solo si no queda ninguno.
8. Se toma un permiso del rate limit saliente y del bulkhead del proveedor; el breaker decide si se intenta.
9. El adapter abre el stream. Si falla o excede el TTFT antes del primer token, entra el siguiente de la cadena (o un hedge, stretch).
10. Cada chunk se traduce a `chat.completion.chunk` y se envía por SSE.
11. Al final: uso real, `budget.settle(real)`, almacenamiento en caché si es elegible, métricas, span y headers.
12. Si el cliente se desconecta, se cancela el stream upstream, se liquida lo consumido y se marca `client_disconnect`.

---

## 3. Componentes
> Patrón: **🧠 Concepto → ⚙️ Cómo funciona aquí → 🔧 Detalle técnico → 🔁 Alternativas.**

| Componente | 🧠 / ⚙️ | 🔧 | 🔁 |
|---|---|---|---|
| **API layer** (`api/`) | Endpoints compatibles con OpenAI, admin, health y la UI | FastAPI con modelos Pydantic; `EventSourceResponse` nativo para SSE; errores en el formato de OpenAI; `orjson` para serializar | Starlette pura |
| **Auth y rate limit** (`security/`) | Una key por cliente, con límites de requests y tokens por minuto y lista de aliases permitidos | Keys `gw_<cliente>_<32 aleatorios>` creadas con `scripts/new_key.py` (muestra la key una vez y guarda solo su SHA-256 en `config/clients.yaml`); `hmac.compare_digest`; `allowed_aliases` *fail-closed* (alias no permitido → 403); token bucket en Redis vía Lua y, si Redis cae, limitador en memoria del proceso con el mismo `rpm` | API gateway externo |
| **Admission control** (`admission/`) | Decide qué entra cuando hay saturación | Límite global de requests en proceso (64, configurable); `batch` se rechaza desde el 75%, `interactive` solo al 100%, con `503 overloaded` + `Retry-After`; `X-Priority` solo puede **bajar** la prioridad del alias | Envoy / NGINX |
| **Router** (`routing/`) | Resuelve el alias en su cadena de proveedores | `routes.yaml` validado con Pydantic al arrancar (recarga en caliente: stretch); alias desconocido → `404 model_not_found` con la lista de aliases válidos; `max_tokens` del cliente sobre el tope del alias se recorta (`X-Max-Tokens-Clamped`) y, si falta, se usa el del alias; se saltan proveedores con breaker abierto o cuota agotada; EWMA de TTFT solo como métrica | Routing por costo o calidad aprendido (future work) |
| **Providers** (`providers/`) | Adapters con una interfaz común: `complete()`, `stream()`, `embed()` | Protocolo `Provider`; `MockProvider` (latencia, TPOT, tasa de fallas y cola larga configurables, determinista por semilla; **solo con `GW_ENV` = `dev` o `test`**); `OpenAICompatProvider` (Ollama vía `/v1`, Groq, Google AI Studio, OpenRouter, vLLM; siempre envía `stream_options: {include_usage: true}`); `AnthropicProvider` (ADR-0005). Errores tipados en §3.1 | SDKs oficiales (más peso) |
| **Cuotas salientes** (`quotas/`) | Respetar RPM/TPM/RPD de cada proveedor | Token buckets en Redis por proveedor y modelo; lectura de headers `x-ratelimit-*` y `retry-after` para ajustar en vivo | Esperar al 429 (lo que hace casi todo el mundo) |
| **Resiliencia** (`resilience/`) | Timeouts, reintentos, breaker, bulkhead (core); hedging y concurrencia adaptativa (stretch) | Breaker con ventana deslizante en memoria del proceso (ADR-0003); backoff exponencial con jitter completo; bulkhead fijo por proveedor; limitador AIMD/gradiente y hedging con presupuesto como stretch | tenacity / aiobreaker |
| **Budget** (`budget/`) | Reserva y liquidación atómicas | Lua en Redis: `reserve(run, client, max_cost)` revisa los 3 topes (corrida, cliente/día, global/día; día UTC) en una sola operación, y `settle(id, real)`; montos en micro-dólares enteros; reservas con TTL; `GET /admin/budget` (ADR-0004) | Contabilidad en Postgres (más lenta) |
| **Pricing y tokens** (`pricing/`) | Del uso al costo en USD, y estimación previa | `config/prices.yaml` (entrada, salida, cache read/write); conteo previo con el endpoint `count_tokens` de Anthropic y, si falla o no existe, `ceil(caracteres / 3)`; el `usage` del proveedor manda al liquidar; si falta `usage`, estimación conservadora (redondeo hacia arriba) y métrica `gw_usage_estimated_total` | Tablas de LiteLLM |
| **Caché** (`cache/`) | Exacta + semántica obligatoria con defensas en capas | Ver §6 | GPTCache, Qdrant |
| **Embeddings** (`embeddings/`) | Vectores para la caché y endpoint `/v1/embeddings` | Protocolo `Embedder`: `OnnxEmbedder` (multilingual-e5-small int8, CPU, por defecto) y `OllamaEmbedder` (`nomic-embed-text`, heredado); EmbeddingGemma-300M como candidato medido en G6 (§6.6); micro-batching | FastEmbed |
| **Structured output** (`structured/`) | Garantizar JSON válido según un schema | `response_format` nativo cuando existe (Ollama `format`, tool-use en Anthropic, JSON mode en Groq/Google); validación con `jsonschema`; un reintento con el error como feedback | Outlines / Instructor del lado cliente |
| **Guardrails** (`guardrails/`) | Injection y PII en el borde (stretch) | Prompt Guard 2 22M (ONNX, CPU); Presidio (analyzer + anonymizer) con reconocedores en español; modos `off`/`monitor`/`enforce` por alias | NeMo Guardrails, Llama Guard |
| **Telemetry** (`telemetry/`) | Métricas, trazas y logs | `prometheus_client`; OTel mínimo con atributos `gen_ai.*`, activo solo con `GW_OTEL_ENDPOINT`; un log JSON por request en stdout, sin prompts (`LOG_PROMPTS=false`); Langfuse como stretch | OpenLIT |
| **UI** (`ui/`) | Demo visual sin escribir código | Un solo `index.html` (HTML + JS vanilla) servido en `GET /`, sin ninguna key embebida | Gradio (más peso) |

### 3.1 Errores de proveedor
`ProviderError` lleva un `kind` con 3 valores. La capa de resiliencia decide con él:

| `kind` | Ejemplos | ¿Reintenta? | ¿Fallback? | ¿Cuenta para el breaker? |
|---|---|---|---|---|
| `transient` | Timeout, error de conexión, `429`, `5xx` | Sí, antes del primer token (reglas de §4.2; `429` no se reintenta) | Sí | Sí |
| `provider` | `401`, `404` de modelo, configuración rota | No | Sí | Sí |
| `client` | `400` por una request inválida | No | **No**: se devuelve al cliente | No |

**Excepción:** "contexto demasiado largo" llega como `400`, pero se trata como `provider` (fallback), porque otro modelo de la cadena puede tener más contexto.
**Rechazo del modelo** (`stop_reason: "refusal"` en Anthropic): se responde con `finish_reason: "content_filter"`, sin fallback, sin contar para el breaker y sin guardar en caché.

### 3.2 Contrato hacia el cliente
- **Modelo desconocido:** `404 model_not_found` (formato OpenAI) con la lista de aliases válidos.
- **Alias no permitido para la key:** `403 alias_not_allowed`. Solo P3 puede usar `smart_paid`.
- **`max_tokens`:** si supera el tope del alias, se recorta y se avisa con `X-Max-Tokens-Clamped`; si falta, se usa el valor del alias.
- **`mock`:** solo existe con `GW_ENV` = `dev` o `test`. En otro entorno, los eslabones `mock` de las cadenas se ignoran.
- **`allow_degraded: false`** (en `judge_golden`, `extract` y `vision`): el alias no cae a proveedores marcados `degraded: true`; si no queda otro, responde `503 all_providers_unavailable`. Así un juicio o una extracción nunca salen de un modelo de baja calidad sin avisar.

---

## 4. Resiliencia

### 4.1 Timeouts por fase
| Fase | Valor por defecto | Razón |
|---|---|---|
| Conexión | 2 s | Proveedor caído → fallback rápido |
| **Primer token (TTFT)** | 10 s (`smart`), 5 s (`fast`) | Lo que el usuario siente |
| **Entre tokens (stall)** | 10 s | Un stream congelado es una falla. Los eventos de *thinking* de Anthropic (que no se reenvían) cuentan como señal de vida |
| Total | 120 s | Deadline duro; siempre gana sobre reintentos y fallbacks |
| Embedding de la caché | 300 ms | Si el embedder se atrasa, se salta la caché semántica (nunca bloquea la request) |

### 4.2 Reintentos con jitter
Backoff con jitter completo: espera $= \text{rand}(0, \min(c, b \cdot 2^{n}))$, con $c = 500$ ms. Así se evita la manada de reintentos sincronizados. Reglas (solo **antes del primer token**, ADR-4):
- **Como máximo 1 reintento por proveedor**, solo para errores de conexión y `5xx` (`kind = transient`, §3.1).
- **`429` no se reintenta:** se pasa al siguiente proveedor. Las cuotas de §3 ya deberían evitarlo; si llega, la cuota real es menor que la configurada.
- **Como máximo 5 intentos por request**, sumando toda la cadena (incluye la reparación de JSON de §7).
- El `total_timeout` del alias siempre gana: al vencer, se responde con error.

En una cadena, pasar rápido al siguiente proveedor es mejor que insistir con el mismo.

### 4.3 Circuit breaker por proveedor (obligatorio)
Tres disparadores sobre una ventana deslizante de las últimas $N$ llamadas:
- **Fallas consecutivas** $\geq k$ (heredado del Go: $k=3$).
- **Tasa de error** $r = \frac{\text{fallas}}{N} > \theta_e$ con un mínimo de llamadas para decidir.
- **Slow-call rate:** fracción de llamadas con TTFT $> t_{slow}$ mayor que $\theta_s$. Un proveedor lento también se trata como degradado.

| Parámetro | Valor inicial |
|---|---|
| $k$ (fallas consecutivas para abrir) | 3 |
| Ventana $N$ / mínimo de llamadas | 20 / 10 |
| $\theta_e$ (tasa de error) | > 50% |
| $t_{slow}$ (llamada lenta) | TTFT > 80% del `ttft_timeout` del alias |
| $\theta_s$ (tasa de llamadas lentas) | > 60% |
| Enfriamiento $T$ | 30 s; se duplica en cada nueva apertura, hasta 5 min |
| Requests de prueba en semiabierto $m$ | 1 |

Estados: **cerrado** → **abierto** (salta directo al siguiente de la cadena) → tras el enfriamiento $T$, **semiabierto** (deja pasar $m$ requests de prueba) → cerrado si salen bien. Los valores se ajustan después de medir G4 y G10. Solo cuentan los errores `transient` y `provider` (§3.1); los `client` y los rechazos no cuentan. El estado vive en la **memoria del proceso** y el gateway corre con 1 worker ([ADR-0003](docs/adr/0003-single-worker-in-process-state.md)). Se expone en métricas, en `/stats`, en la UI y en `/healthz`.

### 4.4 Bulkheads y concurrencia adaptativa
**Core:** un límite **fijo** de concurrencia por proveedor (`concurrency` en `providers.yaml`). Ollama arranca con 2 slots (la GPU de 4 GB no aguanta más). Si se excede, la request espera con un timeout o pasa al siguiente de la cadena.
**Stretch:** reemplazar el número fijo por un **limitador adaptativo** (AIMD o gradiente, como `concurrency-limits` de Netflix), que sube el límite mientras la latencia se mantiene y lo baja cuando crece.

### 4.5 Hedged requests (stretch)
Para los aliases interactivos, si una request no recibe el primer token antes del p95 histórico del alias, se lanza una copia contra el siguiente proveedor sano. Gana el primero que emita un token y el otro se cancela. Tope de hedges por ventana (ADR-12) para no duplicar costos ni cuotas. Nunca se hace hedging contra proveedores pagos salvo configuración explícita. El flag `hedge` de `routes.yaml` no tiene efecto hasta implementar este stretch.

### 4.6 Routing según salud
**Core:** el orden es el del YAML, y solo se **saltan** los proveedores con breaker abierto o cuota agotada. La EWMA de TTFT por proveedor se calcula y se expone como métrica.
**Stretch:** saltar o reordenar también los proveedores con EWMA de TTFT muy por encima de su línea base.

### 4.7 Load shedding y singleflight
- **Load shedding:** límite global de 64 requests en proceso (configurable). Las `batch` se rechazan con `503` + `Retry-After` desde el 75% (48); las `interactive`, solo al 100%. Así el juez de P2 corriendo un set completo no frena al RAG de P3 ni a la UI. La prioridad sale del alias; el cliente puede **bajarla** con `X-Priority: batch`, pero no subirla.
- **Singleflight:** N requests idénticas **y cacheables** en vuelo generan **una** llamada al proveedor; el resto espera ese resultado. Si la request es streaming, las que esperan reciben la respuesta completa como stream simulado (mismo mecanismo que un hit, §6.5). Vive en memoria del proceso (ADR-0003). Evita la estampida cuando una respuesta popular expira de la caché.

### 4.8 Modos degradados
- Redis caído: sin caché, rate limit aproximado en memoria del proceso, y los proveedores pagos rechazan la request (ADR-7).
- Embedder caído o lento: solo caché exacta.
- Solo quedan proveedores `degraded: true` y el alias tiene `allow_degraded: false`: `503 all_providers_unavailable` (§3.2).
- Todos los proveedores caídos: error `all_providers_unavailable`, que los clientes traducen a su propio fallback (plantilla en P1, abstención en P3).

### 4.9 Apagado ordenado
Cloud Run da 10 s entre `SIGTERM` y el cierre forzado. Con `SIGTERM`:
1. Deja de aceptar requests (`/readyz` en 503).
2. Espera hasta **8 s** a que terminen los streams activos.
3. Cancela los restantes con un evento `error` tipado y **liquida el presupuesto** de todo lo consumido.
4. Usa los **2 s** finales para cerrar las conexiones a Redis.

Si el drenaje excediera los 10 s, el proceso moriría sin liquidar, y eso rompería R06.

---

## 5. Control de costos

### 5.1 Contabilidad
$C = (t_{in}\,p_{in} + t_{out}\,p_{out} + t_{cache\_read}\,p_{cr} + t_{cache\_write}\,p_{cw}) / 10^6$, usando los tokens reportados por el proveedor. Los precios están en `prices.yaml` y se verifican antes de la prueba final. Los free tiers cuestan $0 pero tienen **precio de referencia** (lo que costaría pagando), para reportar el ahorro de la caché en dólares comparables. Todos los montos se guardan en **micro-dólares enteros** (1 USD = 1 000 000): Lua en Redis acumula errores de redondeo con decimales, y una request a Haiku 5.5 puede costar US$0.00005. La conversión a dólares se hace solo al mostrar.

### 5.2 Presupuesto atómico (reservar y liquidar)
```
reserve: if spent + reserved + max_cost > cap (para cada uno de los 3 topes) → reject
         else reserved += max_cost; return reservation_id      (Lua, atómico)
settle:  reserved -= max_cost; spent += real_cost              (Lua, atómico)
expira:  las reservas huérfanas (p. ej., un proceso que murió) caducan con TTL y se liberan
```
- **Topes:** `RUN_BUDGET_USD` (la prueba final de P3, p. ej., US$3), el tope por cliente por día y el tope global por día. El día empieza a las **00:00 UTC**. Un solo script revisa los 3 topes: si uno no alcanza, no se reserva nada (así no queda dinero bloqueado en un tope cuando otro falla).
- **Solo proveedores pagos reservan.** Los gratuitos no tocan el presupuesto; su costo de referencia va solo a métricas. Si Redis cae, los gratuitos siguen y los pagos se bloquean (ADR-7).
- **Si la reserva no alcanza:** se salta el proveedor pago y se sigue con el siguiente de la cadena, con `X-Budget-Skipped: <proveedor>`. `402 budget_exceeded` solo si no queda ningún proveedor (R05).
- **TTL de la reserva:** `total_timeout` del alias + 30 s. Si la liquidación llega después del vencimiento, el costo real **se suma igual** a `spent` y se cuenta en `gw_budget_late_settle_total`. El gasto registrado nunca queda por debajo del real.
- **Stream cancelado o cortado:** se liquida la entrada completa más los tokens de salida emitidos; si falta `usage`, se usa la estimación conservadora de §3.

El test de concurrencia dispara 50 requests contra un tope que alcanza para 10, y verifica que pasan como máximo 10 y que el gasto real nunca supera el tope. Un segundo test llena la caché hasta `maxmemory` y verifica que las keys del presupuesto sobreviven (`volatile-lfu`: solo se desalojan keys con TTL, y el presupuesto no tiene). Detalle y razones en [ADR-0004](docs/adr/0004-budget-accounting.md).

### 5.3 Conteo previo de tokens
`max_cost` = tokens de entrada × precio de entrada + `max_tokens` (ya recortado, §3.2) × precio de salida.
- **Entrada:** endpoint `count_tokens` de Anthropic; si falla, `ceil(caracteres / 3)` (conservador). Que `count_tokens` sea gratis se verifica en M7.
- **Salida:** en Haiku 5.5 el *thinking* adaptativo se cobra como salida y está **dentro** de `max_tokens`, así que la reserva ya lo cubre.
- **Con prompt caching activo:** la entrada se reserva al precio `cache_write` (el peor caso); la liquidación cobra el costo real.
- Requests que excedan `max_input_tokens` del alias se rechazan con `400` antes de llamar (protección contra abuso de costo). En `smart_paid` el tope es 100 000 tokens, para quedar siempre en el tramo barato de Haiku 5.5.

### 5.4 Prompt caching del proveedor
Distinto de la caché del gateway: el proveedor cobra menos por los prefijos de prompt repetidos.
- **Anthropic (core):** si el alias declara `prompt_cache: true`, el gateway marca con `cache_control` el system prompt y las tools cuando su tamaño alcanza el mínimo del modelo (512 tokens en Haiku 5.5). El cliente puede desactivarlo con `X-Prompt-Cache: off`. Se contabilizan `cache_creation` y `cache_read` por separado. Detalle en [ADR-0005](docs/adr/0005-anthropic-adapter-haiku-5-5.md).
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
- **Texto que se embebe:** la última pregunta del usuario. La caché **semántica** solo se usa si la request tiene un único mensaje del usuario, o si el alias declara `semantic_scope: last_turn` (cada pregunta es independiente, como el FAQ de P3). En el resto de las conversaciones de varios turnos solo aplica la caché exacta, que compara la conversación completa.
- **El alias decide la elegibilidad:** `cache` (exacta) y `semantic_cache` en `routes.yaml`. La `temperature` del cliente **se ignora** para decidir. El cliente excluye su request con `X-Cache-Scope: private` (no se lee ni se escribe la caché).
- **Elegible para guardar:** sin `tools`, sin `X-Cache-Scope: private`, `finish_reason == "stop"` y sin errores.
- **Nunca se cachea:** respuestas truncadas, rechazos del modelo (`content_filter`), respuestas de proveedores marcados `degraded: true` en la cadena (p. ej., `mock` o `gemma3:1b`), ni salidas estructuradas que no pasaron la validación.

### 6.3 Defensas contra falsos hits
1. **Umbral calibrado, no adivinado.** Con un set de ~1.000 pares etiquetados ("misma pregunta" / "parecida pero distinta", en español e inglés, a partir de las FAQ de P3 y los mensajes de P2) se traza la curva precisión–cobertura y se eligen `t_alto` (precisión ≥ 99,5% sin verificador) y `t_bajo` (zona gris). Cada embedder tiene sus propios umbrales: no se reutiliza el 0.85 de `nomic-embed-text` con e5.
2. **Guardas léxicas:** si la pregunta nueva y la cacheada difieren en números, montos, fechas, monedas, códigos de producto o en negaciones ("no me llegó" vs "me llegó"), el hit se rechaza aunque la similitud sea alta.
3. **Verificador en la zona gris:** un cross-encoder pequeño (MiniLM, ONNX en CPU) confirma la equivalencia. Solo corre en la zona gris para no sumar latencia a todos los hits.
4. **Medición continua:** la tasa de falsos hits se mide en el set de pares (G6) y por muestreo en tráfico real: el juez de P2 revisa una muestra de hits.
5. **Modo observación hasta calibrar:** mientras un embedder no tenga umbrales calibrados con G6, la caché semántica busca y registra el hit que *habría* servido (similitud, candidato), pero **no** lo sirve: llama igual al proveedor. La caché exacta funciona desde el día 1.

### 6.4 Frescura e invalidación
- **TTL** por alias (`cache_ttl_s`; heredado: 7 días en el Go): 7 días para FAQs estáticas, 1 hora para el RAG de P3 (más invalidación por tags) y 24 horas por defecto.
- **Invalidación por tags:** el cliente puede enviar `X-Cache-Tags: doc:123,doc:456` (P3 envía los `doc_id` citados). Cuando P3 publica un cambio de la KB, un consumidor opcional del topic `kb-changes` (o una llamada a `POST /admin/cache/invalidate`) borra las entradas con esos tags. Así **la caché nunca rompe la promesa de frescura de P3**.
- **Sin `FLUSHDB`:** borrado por namespace o por tag; el presupuesto y los rate limits viven en otro prefijo.

### 6.5 Hits en streaming
Un hit se reenvía como stream de chunks con un ritmo configurable (o de una vez), para que el cliente no distinga el camino. El header `X-Cache-Layer: exact|semantic|verified` y `X-Cache-Similarity` lo dicen igual.

### 6.6 Detalle técnico
- Índice: Redis 8 Query Engine, `FT.CREATE` sobre HASH con prefijo `semcache:`, campos `ns` (TAG), `tags` (TAG), `embedding` (VECTOR HNSW, FLOAT32, COSINE, `M=16`, `EF_CONSTRUCTION=200`), `created_at` (NUMERIC), `hits` (NUMERIC).
- Embedder por defecto: multilingual-e5-small en ONNX int8 (~118M parámetros, 384 dims, CPU, ~10–20 ms por consulta corta, licencia MIT). Corre dentro del gateway, sin Ollama ni internet.
- **Elección medida, no por ranking:** en M4, G6 compara con los mismos pares (ES/EN) a `e5-small`, **EmbeddingGemma-300M** (más calidad, ~2–3 veces más pesado) y `nomic-embed-text` en Ollama (el del Go). Gana el que da más cobertura con 99,5% de precisión, siempre que su p95 de hit sea < 60 ms (G9) dentro del límite de memoria del contenedor. Modelos de ~600M (Qwen3-Embedding-0.6B, BGE-M3) se descartan por memoria y latencia en CPU; las APIs de embeddings, por el modo offline y la política de datos. Los rankings MTEB miden sobre todo recuperación de documentos largos, no paráfrasis de preguntas cortas.
- Memoria de Redis: `maxmemory-policy volatile-lfu`. Las entradas de caché tienen TTL y el presupuesto no, así que solo se desaloja caché. AOF activo (`appendfsync everysec`) para que el presupuesto sobreviva a un reinicio.
- *Write-back* asíncrono y acotado (cola con límite), como en el Go, para no sumar latencia a la respuesta.
- Memoria: `maxmemory` dedicado; tamaño por namespace en `/stats`.
- Almacenamiento y búsqueda detrás de una interfaz `SemanticCacheBackend` (implementación única: `redis_local`). **Redis LangCache** (servicio gestionado, en preview) se descartó como caché principal y queda como baseline opcional en M9: ver [ADR-0002](docs/adr/0002-semantic-cache-own-implementation-not-langcache.md).

---

## 7. Streaming, cancelación y salida estructurada
- **SSE de salida** con el formato `chat.completion.chunk` de OpenAI, cerrado con `data: [DONE]`, compatible con los SDKs. Headers `Cache-Control: no-cache` y `X-Accel-Buffering: no` (heredado).
- **Traducción del streaming de Anthropic:** `message_start`, `content_block_delta` (text_delta), `message_delta` (usage) y `message_stop` se convierten en chunks de OpenAI más el uso final. Los bloques de *thinking* no se reenvían, pero cuentan como señal de vida para el `stall_timeout`.
- **Uso en streaming:** el gateway siempre pide `stream_options: {include_usage: true}` a los proveedores compatibles con OpenAI.
- **Keep-alive:** el gateway envía un comentario SSE `: ping` cada 10 s sin datos. Los clientes lo ignoran, y evita que los proxies cierren una conexión silenciosa (p. ej., mientras Haiku 5.5 piensa).
- **Error a mitad del stream:** chunk final `data: {"error": {"type": ..., "code": ...}}` en formato OpenAI y cierre **sin** `[DONE]`. El SDK de OpenAI lo convierte en excepción, así el cliente no confunde "cortado" con "terminado".
- **Cancelación:** si el cliente se desconecta, se cancela la tarea, se cierra el stream httpx upstream y el proveedor deja de generar. Se liquida el costo de los tokens recibidos. Lo valida un test con un servidor real: corta en el token 5 y verifica que el upstream se cerró.
- **Regresión del Go cubierta:** un test verifica que el stream **sobrevive** al retorno del handler (el bug que abortaba todas las llamadas a Groq en ~30 ms).
- **Stall:** si no llega un chunk en `stall_timeout`, se cancela el upstream y se emite un chunk de error.
- **Salida estructurada:** `response_format: {type: json_schema}`. Si la validación falla, se hace 1 reintento con el mismo proveedor y el error como contexto; si vuelve a fallar, se pasa al siguiente proveedor de la cadena. Todo cuenta dentro del límite de 5 intentos (§4.2). Al final se responde con un error tipado (`invalid_structured_output`), nunca con JSON roto.
- **Salida estructurada con `stream: true`:** el gateway recibe la respuesta completa, la valida y recién entonces la envía como stream simulado (como un hit, §6.5). Se pierde TTFT, pero el cliente nunca recibe JSON roto, y como no se emitió nada todavía, se puede reintentar o cambiar de proveedor sin romper ADR-4.
- **Rechazo del modelo:** `finish_reason: "content_filter"` (§3.1).
- **Multimodal (stretch):** los mensajes con partes `image_url` (base64 o URL) se aceptan en el alias `vision` y se traducen al formato de cada proveedor (Gemma 4 vía AI Studio; Ollama con un modelo con visión). Los demás aliases las rechazan con un error claro.

---

## 8. Observabilidad, seguridad y guardrails

### 8.1 Métricas (Prometheus) y endpoints
`gw_requests_total{alias,provider,status}` · `gw_ttft_seconds{alias,provider}` · `gw_ttft_ewma_seconds{provider}` · `gw_request_seconds` · `gw_tokens_total{provider,direction}` · `gw_cost_usd_total{provider,client}` · `gw_cost_saved_usd_total{layer}` · `gw_cache_requests_total{layer,result}` · `gw_cache_similarity` (histograma) · `gw_cache_shadow_hits_total` (modo observación) · `gw_breaker_state{provider}` · `gw_budget_remaining_usd{scope}` · `gw_budget_late_settle_total` · `gw_usage_estimated_total{provider}` · `gw_fallbacks_total{from,to}` · `gw_quota_remaining{provider,kind}` · `gw_shed_total{priority}` · `gw_stream_outcomes_total{outcome}` · stretch: `gw_hedges_total{outcome}`, `gw_concurrency_limit{provider}`, `gw_guardrail_events_total{type,action}`. Solo labels acotados: nunca prompts ni IDs de usuario.

Endpoints:
- **Públicos:** `/healthz` (vivo) y `/readyz` (Redis y al menos un proveedor disponible), que Cloud Run necesita.
- **Con key de admin** (`GW_ADMIN_KEY_SHA256`, separada de las keys de cliente): `/stats` (JSON heredado del Go: caché, breakers, presupuesto, cuotas), `/metrics` (Prometheus envía la key al leer) y `/admin/*`.

### 8.2 Trazas y logs
- **Trazas (core mínimo, opcionales):** un span por request con atributos `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.*` y un span hijo por intento de proveedor (así se ven los fallbacks). Spans por capa de caché y por guardrail cuando existan. Solo se activan si existe `GW_OTEL_ENDPOINT`; sin esa variable no hacen nada. Exportación OTLP a Jaeger (perfil `obs` de docker compose). Langfuse es stretch.
- **Correlación:** el gateway acepta `X-Request-ID` del cliente o crea uno, y lo devuelve. También acepta `traceparent` (W3C), así la traza de P3 continúa dentro del gateway.
- **Log:** una línea JSON por request en stdout: id, cliente, alias, proveedor, intentos, capa de caché, TTFT, tokens, costo y estado. **Nunca** el texto de la pregunta ni de la respuesta (R16). Cloud Run lee ese JSON directamente.

### 8.3 Seguridad
- Keys por cliente con formato `gw_<cliente>_<32 aleatorios>`, creadas con `scripts/new_key.py` (muestra la key una sola vez y guarda solo su SHA-256), comparadas en tiempo constante. El prefijo permite reconocer de quién es una key filtrada. SHA-256 alcanza (sin bcrypt) porque la key es aleatoria y larga. *Fail-closed* si falta la configuración. Rotación vía archivo (recarga sin downtime: stretch).
- `allowed_aliases` por cliente, *fail-closed*: alias no permitido → 403.
- Secrets de proveedores solo por variables de entorno (Secret Manager en Cloud Run).
- `LOG_PROMPTS=false` por defecto. Si se activa y Presidio no está instalado, **el gateway no arranca** y muestra un error claro: no hay forma de escribir prompts sin redactar.
- Límites de tamaño de request (tokens de entrada, imágenes) para evitar abusos de costo.
- CORS deshabilitado salvo para la UI servida por el mismo origen. La UI no trae keys embebidas (§9.1).

### 8.4 Guardrails
| Guardrail | Cómo | Modo por defecto | Quién lo usa más |
|---|---|---|---|
| Prompt injection / jailbreak | Llama Prompt Guard 2 22M en ONNX (CPU), sobre mensajes `user` y sobre contenido marcado como externo (`X-Untrusted-Content`) | `monitor` (marca y mide) | **P4**: el contenido web (SearXNG + Crawl4AI) es la superficie de ataque principal |
| PII | Presidio (emails, teléfonos, tarjetas, documentos de identidad; reconocedores en español) | Redacción en logs y trazas; opcional en el tráfico hacia free tiers | P1–P4 |
| Tamaño y forma | Límites por alias | `enforce` | Todos |

Los falsos positivos se miden (G14) antes de pasar cualquier guardrail a `enforce`. **Core:** solo "Tamaño y forma". Prompt injection y PII son stretch (§1.6).

### 8.5 Headers
**Respuesta** (todos documentados en una sola tabla del README; la UI los muestra como *pills*):

| Header | Significado |
|---|---|
| `X-Request-Id` | Id de seguimiento (el del cliente o uno nuevo) |
| `X-Alias` | Alias resuelto |
| `X-Gateway-Provider` | Proveedor y modelo que respondieron de verdad |
| `X-Gateway-Fallback-Depth` | Cuántos proveedores se probaron |
| `X-Cache-Layer` | `exact`, `semantic`, `verified` o ausente |
| `X-Cache-Similarity` | Similitud del hit semántico |
| `X-Max-Tokens-Clamped` | `max_tokens` recortado al tope del alias |
| `X-Params-Dropped` | Parámetros que el proveedor no acepta y se quitaron (p. ej., `temperature` en Haiku 5.5) |
| `X-Budget-Skipped` | Proveedor pago saltado porque no alcanzó el presupuesto |

**Request:** `Authorization: Bearer`, `X-Priority` (solo baja), `X-Cache-Scope: private`, `X-Cache-Tags`, `X-Prompt-Cache: off`, `X-Request-ID`, `traceparent`, `X-Untrusted-Content` (stretch).

---

## 9. UI de chat, ejemplos y experiencia de desarrollo

### 9.1 UI integrada (heredada del Go y ampliada)
Un único `index.html` (HTML, CSS y JS vanilla, sin build, sin dependencias externas) servido en `GET /`:
- Campo de API key (el HTML **no** trae ninguna key; la que pega el usuario se guarda en `sessionStorage`, enmascarada, y se borra al cerrar la pestaña), **selector de alias** y de modelo.
- Streaming con cursor y errores visibles (si llega un chunk de error, se muestra en rojo en vez de quedar en blanco).
- *Pills* con los headers reales de §8.5: capa de caché, similitud, proveedor, profundidad de fallback, latencia y ajustes del gateway.
- Panel lateral con el estado de los breakers, las cuotas y el presupuesto (leído de `/stats`; solo con key de admin).
- Para mostrar el proyecto sin exponer nada: un GIF o video en el README, grabado en local. Cliente `demo` en `clients.example.yaml` (solo alias `fast`, `rpm` bajo, presupuesto 0) para quien lo corra en su computador.
- Atajos: `Enter` envía, `Shift+Enter` agrega una línea; botón `Clear`.

### 9.2 Ejemplos de clientes (`examples/`)
`curl_nonstream.sh`, `curl_stream.sh`, `python_openai_sdk.py`, `python_httpx_async.py`, `langgraph_client.py`, `node_fetch.mjs` y `node_openai_sdk.mjs`, todos contra el mismo endpoint.

### 9.3 DX
`make up` (gateway + Redis), `make up-obs` (además Prometheus, Grafana con el dashboard versionado y Jaeger; ~300–500 MB extra), `make eval` (evaluaciones completas), `make demo-breaker` (mata el primario y muestra la conmutación en la UI), `make demo-cache` (misma pregunta dos veces y una "parecida pero distinta"), troubleshooting en el README (heredado: key que no persiste entre terminales, Redis caído, embedder caído, buffering de proxies).

---

## 10. Evaluación
| ID | Alcance | Prueba | Cómo | Resultado esperado | Dónde corre |
|---|---|---|---|---|---|
| G1 | Core | **Overhead** | Carga open-loop con k6 contra `mock` con latencia 0, directo vs a través del gateway; p50/p95/p99. **Dos números:** sin caché, y en un miss con la caché semántica activa (embedding + KNN) | p95 sin caché < 5 ms; el costo del miss se publica | Local |
| G2 | Opcional (M9) | Overhead vs LiteLLM | Mismo test con LiteLLM proxy | Comparación honesta | Local |
| G3 | Core | **Budget bajo concurrencia** | 50 requests concurrentes con un tope que alcanza para 10 | 0 excesos | Cada push |
| G4 | Core | Fallback | `mock` primario con 100% de fallas → secundario | ≥ 99% de éxito; breaker abierto en métricas | Cada push |
| G5 | Core | Cancelación | El cliente corta en el token 5; además, test de regresión del bug de contexto del Go | Upstream cancelado; tokens liquidados ≈ consumidos; stream largo sobrevive | Cada push |
| G6 | Core | **Caché semántica: calidad** | Set de ~1.000 pares etiquetados (ES/EN); embedders `e5-small`, EmbeddingGemma-300M y `nomic-embed-text` | Curva precisión–cobertura por embedder; **tasa de falsos hits < 1%** en el umbral elegido | 100 pares difíciles en cada push; completo cada noche |
| G7 | Core | Contratos | Suite con los SDKs oficiales de OpenAI (streaming, no streaming, `response_format`, embeddings, errores) | 100% en verde | Cada push |
| G8 | Core | Caos | Redis caído, embedder caído, Ollama caído y lento (Toxiproxy) | Comportamiento según §4.8 | Cada push (con `mock`); Ollama en local |
| G9 | Core | **Latencia de un hit** | p50/p95 de un hit exacto y semántico, con cada embedder de G6 | p95 semántico < 60 ms; comparado con los 34–90 ms del Go | Cada noche (tendencia); número final en local |
| G10 | Core | **Disponibilidad** | 10k requests con fallas intermitentes, latencia de cola y 429 inyectados en el primario | ≥ 99.9% de éxito | Cada noche |
| G11 | Core | **Costo ahorrado** | Replay de tráfico Zipf (FAQ de P3 + mensajes de P2) con precios de referencia | Hit rate y % de costo ahorrado (el CV dice 30–40%) | Cada noche |
| G12 | Stretch | Hedging | Proveedor `mock` con cola larga (p99 alto) | p99 con hedging vs sin hedging; % de hedges y costo extra | Local |
| G13 | Core | Throughput | Camino de caché exacta con 1, 2 y 4 workers (uvloop) | req/s sostenidos en este hardware, comparados con la cifra del Go | Local |
| G14 | Stretch | Guardrails | Set público de prompt injection + prompts benignos en español | Recall y tasa de falsos positivos por umbral | Cada noche |
| G15 | Core | Cuotas | Free tier simulado con RPM bajo | 0 errores 429 propagados mientras haya otro proveedor | Cada push |

**Dónde corre cada prueba.** La regla: en la CI corre lo que da el mismo resultado en cualquier máquina.
- **Cada push** (`.github/workflows/ci.yml`, minutos): tests normales + pruebas con `mock` y Redis como servicio del job. Detectan si un cambio rompe el contrato.
- **Cada noche o con botón** (`.github/workflows/eval-nightly.yml`, `schedule` + `workflow_dispatch`): pruebas lentas o que descargan modelos; los modelos ONNX se guardan con `actions/cache`. Es lo mismo que `make eval`, que se corre además antes de cada versión y antes de la prueba final de P3.
- **Solo local:** los números de velocidad (G1, G13, G2, G12), porque los runners de GitHub son compartidos y su velocidad varía.

> **Baseline opcional de caché:** si en M9 Redis LangCache sigue gratis y accesible, G6, G9 y G11 se corren también contra él con los mismos datos y se publica la comparación (ADR-0002).

---

## 11. Requisitos EARS (contrato de aceptación)
Cada requisito tiene un test con su ID (`test_req_XX_*`). El README incluye la tabla de trazabilidad requisito → test.

| ID | Requisito |
|---|---|
| R01 | WHEN un cliente envía una request con una key válida THEN el gateway SHALL responder en el formato de OpenAI con los headers `X-Gateway-Provider`, `X-Alias` y `X-Request-Id`. |
| R02 | IF la key falta o no coincide THEN el gateway SHALL responder 401 sin llamar a ningún proveedor. |
| R03 | WHEN el proveedor primario de un alias falla antes del primer token THEN el gateway SHALL intentar el siguiente proveedor de la cadena. |
| R04 | WHILE el breaker de un proveedor está abierto, el gateway SHALL saltarse ese proveedor sin llamarlo. |
| R05 | El gateway SHALL NOT llamar a un proveedor pago sin una reserva de presupuesto aceptada. IF la reserva excede un tope THEN el gateway SHALL saltar ese proveedor y seguir con la cadena; IF no queda ningún proveedor THEN SHALL responder `budget_exceeded`. |
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
| R17 | IF el `model` pedido no es un alias conocido THEN el gateway SHALL responder `404 model_not_found` con la lista de aliases válidos. |
| R18 | IF la key del cliente no tiene permitido el alias THEN el gateway SHALL responder 403 sin llamar a ningún proveedor. |
| R19 | IF `max_tokens` supera el tope del alias THEN el gateway SHALL recortarlo al tope y SHALL incluir `X-Max-Tokens-Clamped`. |
| R20 | WHEN el proveedor rechaza la request (`refusal`) THEN el gateway SHALL responder con `finish_reason: "content_filter"`, SHALL NOT hacer fallback y SHALL NOT guardar la respuesta en caché. |
| R21 | WHILE la request lleva `X-Cache-Scope: private`, el gateway SHALL NOT leer ni escribir ninguna capa de caché. |
| R22 | IF `LOG_PROMPTS=true` y Presidio no está disponible THEN el gateway SHALL NOT arrancar. |
| R23 | WHEN llega `SIGTERM` THEN el gateway SHALL dejar de aceptar requests, SHALL esperar hasta 8 s a los streams activos y SHALL liquidar el presupuesto de los que cancele. |

---

## 12. Recursos, ejecución y despliegue
- **RAM:** el gateway ~200 MB, el embedder ONNX ~150 MB, el verificador ~100 MB, Prompt Guard ~100 MB (stretch). Redis se comparte con el proyecto que corra en ese momento. Ollama corre nativo en Windows (`host.docker.internal:11434`). El perfil `obs` (Prometheus, Grafana, Jaeger) suma ~300–500 MB y solo se levanta con `make up-obs`.
- **Proceso:** uvicorn con **1 worker** por defecto (async: 1 worker atiende muchas requests a la vez). Breaker, singleflight y el rate limit de respaldo viven en memoria del proceso (ADR-0003); cuotas y presupuesto, en Redis.
- **Redis local:** `--appendonly yes --appendfsync everysec --maxmemory-policy volatile-lfu` (§6.6).
- **Modos de ejecución:** (1) **contenedor en el Compose de cada proyecto** (el perfil que necesite LLM incluye el servicio `llm-gateway` desde su imagen); (2) **standalone** con `docker compose up` en este repo, para desarrollo, demos con la UI y benchmarks; (3) **Cloud Run** en la prueba final de P3.
- **Prueba final de P3 ([ADR-0006](docs/adr/0006-p3-final-deployment-cloud-run-redis-cloud.md)):** Cloud Run con 1 instancia como máximo, **ingress interno** (solo P3 lo llama), keys en Secret Manager. Redis en el **plan gratis de Redis Cloud** (30 MB, 30 conexiones), en la misma región: un Redis dentro del servicio perdería el presupuesto cada vez que Cloud Run escala a cero. Doble barrera de gasto: el tope del gateway **y** un límite de gasto en la consola de Anthropic (p. ej., US$5). Además, una alerta de presupuesto en Google Cloud. La prueba solo usa datos sintéticos o públicos, y al terminar se compara el gasto registrado con la consola de Anthropic.
- **Imagen:** multi-stage con Python 3.12 slim y uv; los modelos ONNX se descargan en el build (capa cacheada) o al primer arranque (documentado).
- **Keys externas ($0):** `GW_GROQ_API_KEY` y `GW_GOOGLE_API_KEY` (free tiers). Sin ellas, las cadenas siguen funcionando con Ollama y `mock`.

---

## 13. Estructura del repo y del README
```
llm-gateway/
├── README.md · PLAN.md · GLOSSARY.md · CLAUDE.md · LICENSE · Makefile · docker-compose.yml · Dockerfile · .env.example · pyproject.toml
├── .github/workflows/ (ci.yml: cada push · eval-nightly.yml: evaluaciones lentas)
├── scripts/ (doctor.py, new_key.py)
├── src/llm_gateway/
│   ├── main.py · settings.py
│   ├── api/ (chat.py, embeddings.py, models.py, admin.py, health.py, schemas.py)
│   ├── providers/ (base.py, mock.py, openai_compat.py, anthropic.py; ollama.py solo si M2 lo justifica)
│   ├── routing/ · admission/ · quotas/ · resilience/ · budget/ (lua/) · pricing/
│   ├── cache/ (exact.py, semantic.py, guards.py, verifier.py, invalidation.py)
│   ├── embeddings/ · structured/ · guardrails/ · security/ · telemetry/ · ui/ (index.html)
├── config/ (routes.yaml, prices.yaml, providers.yaml, clients.example.yaml)
├── observability/ (prometheus.yml, grafana/ con el dashboard versionado)
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
| **M2 · Proveedores + routing** | `OpenAICompatProvider` (Groq, Google AI Studio y Ollama vía `/v1`, incl. fallback `gemma3:1b`), `ProviderError` con 3 tipos, `routes.yaml`, aliases, `allowed_aliases`, clamp de `max_tokens`, `allow_degraded`, cuotas salientes. Verificar streaming, `usage` y `response_format` de Ollama `/v1` | P1, P2 y P4 pueden apuntar al gateway con `fast` y `smart`; G15; R17–R19 |
| **M3 · Resiliencia** | Timeouts por fase, reintentos, **breaker con 3 disparadores**, bulkhead fijo, fallback, singleflight, load shedding, apagado ordenado | G4, G8 parcial; R03, R04, R13, R14, R23 |
| **M4 · Caché semántica** | Exacta + semántica obligatoria (namespace, umbral calibrado, guardas, verificador, tags, modo observación), `/v1/embeddings`, embedders `e5-small` (por defecto), EmbeddingGemma-300M y `nomic-embed-text` | **G6, G9 y G11 medidos**; R07–R09, R21 |
| **M5 · Budget + pricing** | Lua de reserva y liquidación (micro-dólares, 3 topes, TTL), conteo previo, `/admin/budget`, salto de proveedor pago, prompt caching (marcado automático y reserva a precio `cache_write`) | **G3 en verde**; test de `volatile-lfu`; R05, R06 |
| **M6 · Streaming robusto** | Cancelación, stalls, `: ping`, `[DONE]`, chunk de error, regresión del Go | G5; R10, R11 |
| **M7 · Structured + Anthropic** | `response_format` (con stream: validar antes de enviar), reparación, `AnthropicProvider` para Haiku 5.5 con streaming, prompt caching, `effort`, parámetros quitados y rechazos (ADR-0005) | Probado con fixtures escritos desde los eventos documentados ($0); antes de la prueba final de P3, grabar unos pocos streams reales con `GW_RUN_BUDGET_USD=0.10`; verificar si `count_tokens` es gratis; R15, R20 |
| **M8 · Observabilidad y seguridad** | Métricas, log JSON, OTel mínimo + Jaeger, perfil `obs` con dashboard de Grafana, keys por cliente con `new_key.py`, key de admin, rate limit con respaldo en memoria | Dashboard del gateway en Grafana; R16, R22 |
| **M9 · Benchmarks + publicación** | G1 (dos números), G10, G13, workflow nocturno, README completo con trazabilidad EARS y tabla de headers, GIF de la UI, `v1.0`. Opcionales: G2 y baseline LangCache | Frase del CV con números; comparación con el Go publicada |
| **Stretch** | Hedging (G12), AIMD, reordenar por EWMA, Prompt Guard + Presidio (G14), Langfuse, visión, recarga en caliente | Solo después de `v1.0`; no bloquea la frase del CV |

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
| Scope creep (es tentador construir un LiteLLM entero) | Solo 6 adapters; división core/stretch de §1.6; las features más allá de §1.2 van a "future work" |
| Cloud Run escala a cero y un Redis dentro del servicio pierde el presupuesto | Redis Cloud gratis para la prueba final + límite de gasto en la consola de Anthropic (ADR-0006) |
| El plan gratis de Redis Cloud no incluye búsqueda vectorial o cambia sus límites | Verificar en M9; si falta, el presupuesto va en Redis Cloud y la caché semántica se apaga en esa prueba |
| Haiku 5.5 rechaza `temperature`/`top_p`/`top_k` no por defecto (todos los modelos desde 2026-10-21, según la doc) | El adapter los quita y avisa con `X-Params-Dropped` (ADR-0005) |
| Un reinicio de Redis local borra el gasto registrado | AOF `everysec`; paso manual de comparar con la consola del proveedor antes de la prueba final |

**Por verificar al implementar:** el endpoint OpenAI-compatible de Ollama (streaming, `usage`, `response_format` y visión), si `count_tokens` de Anthropic es gratis (M7), que el plan gratis de Redis Cloud incluya el Query Engine (M9), los precios de Haiku 5.5 antes de la prueba final, el endpoint compatible con OpenAI de Google AI Studio para Gemma 4 (streaming, JSON mode, imágenes) y sus cuotas del free tier, los modelos y cuotas vigentes de Groq, la forma actual de los eventos de streaming de Anthropic y de `cache_control`, las APIs de vector search de Redis 8, Prompt Guard 2 exportable a ONNX, y FastAPI SSE nativo con `POST` (ya verificado en 0.142).

**Future work:** routing aprendido por calidad y costo (un clasificador que elige el modelo más barato suficiente, idea que P2 explora a nivel de aplicación), gateway MCP (exponer herramientas además de modelos), WebSockets, multi-región.
