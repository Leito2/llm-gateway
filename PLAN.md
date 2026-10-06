# 🛡️ P0 — LLM Gateway (Python, construido desde cero)

> **Plan de proyecto.** **Estado:** ✅ Plan completo · 🟡 Setup M0 hecho · Reemplaza al "LLM Edge Gateway (Go)" en P1, P2 y P3.
> Stack: Python 3.12 · FastAPI + uvicorn · httpx (async) · Pydantic v2 · Redis 8 (budget, caché exacta y semántica con vector search) · ONNX Runtime (embeddings para la caché semántica) · Prometheus + OpenTelemetry (GenAI semconv) · Docker Compose · Cloud Run (solo en la prueba final de P3)
> Gasto: **$0** en desarrollo (proveedores `mock` y `ollama`). Anthropic (Haiku) solo en la prueba final de P3, con tope duro de presupuesto.

## Módulos del plan
| # | Sección | Estado |
|---|---|---|
| 1 | Visión, problema y frase del CV | ✅ |
| 2 | Arquitectura macro y flujo de una request | ✅ |
| 3 | Componentes | ✅ |
| 4 | Resiliencia: timeouts, reintentos, circuit breaker, fallback, bulkheads | ✅ |
| 5 | Control de costos: precios, presupuesto atómico, cachés | ✅ |
| 6 | Streaming, cancelación y salida estructurada | ✅ |
| 7 | Observabilidad y seguridad | ✅ |
| 8 | Evaluación: overhead, caché, caos y contratos | ✅ |
| 9 | Recursos, ejecución y despliegue | ✅ |
| 10 | Estructura del repo y del README | ✅ |
| 11 | Hitos de implementación | ✅ |
| 12 | Riesgos y pendientes | ✅ |

## Regla del README
README progresivo: **contexto teórico, conceptual y macro primero**; en cada componente, el detalle técnico al final. Incluye cómo funciona, los pasos para ejecutarlo y las alternativas de ejecución o despliegue (local primero).

---

## 1. Visión, problema y frase del CV

### 1.1 El problema
Los tres proyectos (P1 fraude, P2 router, P3 RAG) llaman a LLMs: explicaciones de casos (P1), agente System 2 y juez (P2), generación y verificación (P3). Si cada servicio llama a los proveedores por su cuenta:
- **El costo no tiene un punto de control:** no hay tope de presupuesto global, y un bucle o un bug puede gastar dinero real en minutos.
- **Cada servicio reimplementa** reintentos, timeouts, fallback y métricas, cada uno de forma distinta.
- **Cambiar de proveedor** (mock → Ollama → Haiku) implica tocar código en tres repos.
- **No hay caché compartida:** la misma pregunta frecuente se paga muchas veces.

Un **gateway** centraliza todo eso detrás de una sola API. Productos como LiteLLM o Portkey resuelven el problema, pero este proyecto lo construye **desde cero en Python**: como aprendizaje profundo, con control total del comportamiento, y como pieza de portafolio que conecta los otros tres proyectos.

### 1.2 Qué construimos
Un servicio FastAPI asíncrono con:
1. **API compatible con OpenAI** (`/v1/chat/completions`, con y sin streaming; `/v1/models`): cualquier SDK de OpenAI, LangChain o LangGraph lo usa sin cambios.
2. **Proveedores intercambiables:** `mock` (determinista, con inyección de latencia y fallas), `ollama` (local) y `anthropic` (traducción OpenAI ↔ Messages API, incluido el streaming).
3. **Routing por alias:** los servicios piden `fast`, `smart` o `judge`; un YAML define la cadena de fallback de cada alias.
4. **Resiliencia:** timeouts por fase (conexión, primer token, entre tokens, total), reintentos con backoff y jitter (solo antes del primer token), circuit breaker por proveedor, fallback y límites de concurrencia por proveedor.
5. **Control de costos:** tabla de precios, contabilidad por request y **presupuesto atómico en Redis** (reservar y luego liquidar), con tope por corrida, por cliente y por día.
6. **Caché exacta y semántica** en Redis 8 (embeddings ONNX en CPU), acotada a requests deterministas.
7. **Streaming SSE** con propagación de cancelación y detección de stalls.
8. **Salida estructurada:** `response_format` con JSON Schema, validación y un reintento controlado.
9. **Observabilidad:** métricas de Prometheus (TTFT, tokens, costo, caché, estado del breaker, presupuesto), spans de OTel con la convención GenAI y Langfuse opcional.
10. **Seguridad:** API keys por cliente (P1, P2, P3) guardadas con hash, rate limiting y logs sin prompts por defecto.

### 1.3 Métricas de éxito
| Tipo | Métrica | Meta |
|---|---|---|
| Overhead | p95 de latencia agregada por el gateway (con `mock`, sin caché) | < 5 ms |
| Overhead | TTFT agregado al streaming | < 3 ms |
| Resiliencia | Requests exitosas con el proveedor primario caído (fallback) | ≥ 99% |
| Costo | **Nunca** se supera el tope de presupuesto, ni siquiera con 50 requests concurrentes | 0 excesos (test de concurrencia) |
| Caché | Hit rate y costo ahorrado sobre un tráfico FAQ de P3 (distribución Zipf) | Se reporta |
| Compatibilidad | Suite de contratos con el SDK oficial de OpenAI como cliente | 100% en verde |

### 1.4 Frase del CV (plantilla)
> **LLM Gateway (Python, from scratch)** · FastAPI · httpx · Redis 8 · ONNX Runtime · OpenTelemetry
> OpenAI-compatible async gateway with multi-provider fallback, per-provider circuit breakers, **atomic Redis budget enforcement (0 overspend under concurrency)** and an exact + semantic cache (**{h}% hit rate, {s}% cost saved**); adds **p95 {x} ms** overhead and streams SSE with end-to-end cancellation. Serves three production-style ML systems.

### 1.5 Por qué desde cero y no LiteLLM
| Criterio | Construir (este proyecto) | LiteLLM |
|---|---|---|
| Aprendizaje y portafolio | Máximo: cada decisión es tuya y explicable | Bajo: se configura |
| Proveedores | Solo 3 (mock, ollama, anthropic) | Más de 100 |
| Control del presupuesto y de la caché | Total; se diseña para P1, P2 y P3 | Configurable, más genérico |
| Mantenimiento | Tuyo | De la comunidad |

LiteLLM queda como **referencia y baseline**: el benchmark de overhead compara ambos en la misma máquina.

---

## 2. Arquitectura macro y flujo de una request

```
 Clientes (OpenAI SDK / LangChain / httpx)
   P1 explainer · P2 System 2 + juez · P3 RAG API
            │  Authorization: Bearer <api-key del cliente>
            ▼
 ┌──────────────────────────── llm-gateway (FastAPI, async) ────────────────────────────┐
 │ auth → rate limit → resolver alias → caché (exacta → semántica) ──hit──► respuesta    │
 │                                   │ miss                                             │
 │                                   ▼                                                  │
 │                        budget.reserve(estimado)  ──insuficiente──► 402 budget_exceeded│
 │                                   │ ok                                               │
 │                                   ▼                                                  │
 │            cadena de fallback: [proveedor A → B → C], cada uno con:                  │
 │            bulkhead (semáforo) · circuit breaker · timeouts por fase · reintentos     │
 │                                   │                                                  │
 │            adapter: mock │ ollama (OpenAI-compat) │ anthropic (Messages API)          │
 │                                   │ stream / respuesta                               │
 │                                   ▼                                                  │
 │   budget.settle(costo real) · cache.store · métricas · span OTel · evento SSE         │
 └──────────────────────────────────────────────────────────────────────────────────────┘
            │                                  │
            ▼                                  ▼
      Redis 8 (budget, caché, rate limit)   Prometheus / OTel Collector / Langfuse (opcional)
```

### Decisiones de diseño (ADRs)
- **ADR-1 · Contrato OpenAI hacia afuera, adaptadores hacia adentro.** Los clientes hablan siempre el formato de OpenAI; cada proveedor tiene un adapter que traduce. Agregar un proveedor no toca a ningún cliente.
- **ADR-2 · Alias, no modelos.** Los servicios piden `fast`, `smart` o `judge`; `config/routes.yaml` decide qué proveedor y modelo hay detrás de cada alias. Pasar de Ollama a Haiku en la prueba final de P3 es un cambio de configuración.
- **ADR-3 · Presupuesto con reserva y liquidación.** Antes de llamar, se reserva atómicamente el costo **máximo** estimado (tokens de entrada más `max_tokens`) en Redis con Lua. Al terminar, se liquida el costo real y se libera el sobrante. Así, 50 requests concurrentes no pueden pasarse del tope, cosa que un simple "leer saldo y luego descontar" sí permite.
- **ADR-4 · Reintentar solo antes del primer token.** Una vez que hay tokens enviados al cliente, reintentar duplicaría el texto. Desde ahí, las fallas se reportan como un evento `error` en el stream.
- **ADR-5 · Caché solo para requests deterministas.** Se cachea con `temperature == 0` o con un flag explícito `cache: true`, nunca con herramientas o datos por usuario sin aislamiento. La clave incluye el alias, el hash del system prompt, los parámetros y la versión del prompt.
- **ADR-6 · Python async de punta a punta.** httpx async, sin clientes bloqueantes en el camino del streaming. La cancelación de asyncio se propaga desde el cliente hasta el proveedor.
- **ADR-7 · Fallar cerrado en el presupuesto.** Si Redis no responde, las llamadas a proveedores **pagos** se rechazan; `mock` y `ollama` siguen funcionando.

### Flujo de una request en streaming
1. Llega `POST /v1/chat/completions` con `stream: true` y el alias `smart`.
2. Se autentica la key, se aplica rate limiting y se resuelve la cadena (p. ej., `anthropic:claude-haiku-4-5` → `ollama:qwen3:1.7b`).
3. Se busca en caché: si hay hit, se responde como un stream simulado y el costo es $0.
4. `budget.reserve(costo_máximo)`. Si no alcanza, el error es `budget_exceeded`.
5. El adapter abre el stream con el proveedor. Si falla antes del primer token, entra la siguiente opción de la cadena.
6. Cada chunk se traduce al formato `chat.completion.chunk` de OpenAI y se envía por SSE.
7. Al final se registra el uso real, `budget.settle(real)`, la caché si aplica, y las métricas y el span.
8. Si el cliente se desconecta, se cancela el stream upstream, se liquida lo consumido y se marca `client_disconnect`.

---

## 3. Componentes
> Patrón: **🧠 Concepto → ⚙️ Cómo funciona aquí → 🔧 Detalle técnico → 🔁 Alternativas.**

| Componente | 🧠 / ⚙️ | 🔧 | 🔁 |
|---|---|---|---|
| **API layer** (`api/`) | Endpoints compatibles con OpenAI, admin y health | FastAPI con modelos Pydantic de request y response; `EventSourceResponse` nativo para SSE; errores en el formato de OpenAI | Starlette pura |
| **Auth y rate limit** (`security/`) | Una key por cliente, con un límite de requests y tokens por minuto | Keys con hash SHA-256 en `config/clients.yaml`; token bucket en Redis vía Lua | API gateway externo |
| **Router** (`routing/`) | Resuelve el alias en una cadena de proveedores y modelos | `routes.yaml` validado con Pydantic al arrancar; políticas por alias (`max_tokens` por defecto, timeouts, caché sí o no) | Routing por costo o latencia (trabajo futuro) |
| **Providers** (`providers/`) | Adapters con una interfaz común: `complete()` y `stream()` | Protocolo `Provider`; `MockProvider` (latencia, TPOT y tasa de fallas configurables; determinista por semilla), `OllamaProvider` (endpoint OpenAI-compatible de Ollama), `AnthropicProvider` (traduce mensajes y system, mapea `content_block_delta` a chunks) | SDKs oficiales (más peso) |
| **Resiliencia** (`resilience/`) | Timeouts, reintentos, breaker y bulkhead | Breaker por proveedor (cerrado → abierto → semiabierto, ventana deslizante de error rate); backoff exponencial con jitter completo; `asyncio.Semaphore` por proveedor | tenacity / aiobreaker |
| **Budget** (`budget/`) | Reserva y liquidación atómicas | Lua en Redis: `reserve(run, key, max_cost)` y `settle(id, real)`; topes por run, cliente y día; endpoint `GET /admin/budget` | Contabilidad en Postgres (más lenta) |
| **Pricing** (`pricing/`) | Del uso al costo en USD | `config/prices.yaml` (precios por MTok de entrada, salida y cache read); conteo de tokens desde el `usage` del proveedor, con estimación como respaldo | Tablas de LiteLLM |
| **Caché** (`cache/`) | Exacta (hash) + semántica (similitud de embeddings) | Exacta: Redis con TTL. Semántica: embeddings de un modelo e5-small en ONNX int8 (CPU) e índice vectorial HNSW de Redis 8 por alias, con umbral de similitud configurable (p. ej., 0.95) | GPTCache, Qdrant |
| **Structured output** (`structured/`) | Garantizar JSON válido según un schema | Se pasa `response_format` al proveedor que lo soporta (Ollama `format`, tool-use en Anthropic); se valida con `jsonschema`; un reintento con el error de validación como feedback | Outlines / Instructor del lado cliente |
| **Telemetry** (`telemetry/`) | Métricas, trazas y logs | `prometheus_client`; OTel con atributos `gen_ai.*`; logs JSON sin prompts (`LOG_PROMPTS=false`) | Langfuse SDK directo |

---

## 4. Resiliencia

### 4.1 Timeouts por fase
| Fase | Valor por defecto | Razón |
|---|---|---|
| Conexión | 2 s | Proveedor caído → fallback rápido |
| **Primer token (TTFT)** | 10 s (`smart`), 5 s (`fast`) | Lo que el usuario siente |
| **Entre tokens (stall)** | 10 s | Un stream congelado es una falla |
| Total | 120 s | Deadline duro |

### 4.2 Reintentos con jitter
Backoff con jitter completo: espera $= \text{rand}(0, \min(c, b \cdot 2^{n}))$. Así se evita la manada de reintentos sincronizados. Solo se reintentan errores **reintentables** (timeouts de conexión, 429, 5xx) y **solo antes del primer token** (ADR-4).

### 4.3 Circuit breaker por proveedor
Con una tasa de error $r$ en una ventana de $N$ requests y un umbral $\theta$: si $r > \theta$, el breaker se **abre** y las requests saltan directo al siguiente proveedor de la cadena. Tras un enfriamiento de $T$ segundos pasa a **semiabierto**, deja pasar algunas requests de prueba y se cierra si salen bien. Las métricas exponen el estado de cada breaker.

### 4.4 Bulkheads
Un semáforo por proveedor (p. ej., Ollama con 2 slots, porque la GPU de 4 GB no aguanta más). Si se excede, la request **espera** con un timeout o pasa al fallback. Así un proveedor lento no consume todos los workers.

### 4.5 Modos degradados
- Redis caído: sin caché, y los proveedores pagos rechazan la request (ADR-7).
- Todos los proveedores caídos: error `all_providers_unavailable`, que los clientes (P1 explainer, P2 agente) traducen a su propio fallback.

---

## 5. Control de costos

### 5.1 Contabilidad
$C = (t_{in}\,p_{in} + t_{out}\,p_{out} + t_{cache\_read}\,p_{cr}) / 10^6$, usando los tokens reportados por el proveedor. Los precios están en `prices.yaml` y se verifican antes de la prueba final.

### 5.2 Presupuesto atómico (reservar y liquidar)
```
reserve: if spent + reserved + max_cost > cap → reject
         else reserved += max_cost; return reservation_id      (Lua, atómico)
settle:  reserved -= max_cost; spent += real_cost              (Lua, atómico)
expira:  las reservas huérfanas (p. ej., un proceso que murió) caducan con TTL y se liberan
```
Los topes son `RUN_BUDGET_USD` (la prueba final de P3, p. ej., US$3), el tope por cliente y el diario. El test de concurrencia dispara 50 requests contra un tope que alcanza para 10, y verifica que pasan como máximo 10 y que el gasto real nunca supera el tope.

### 5.3 Cachés
| Caché | Clave | Cuándo aplica | Riesgo |
|---|---|---|---|
| Exacta | SHA-256(alias, mensajes normalizados, parámetros, versión del prompt) | `temperature == 0` o `cache: true` | Prácticamente ninguno |
| Semántica | Embedding de la última pregunta del usuario + hash del system prompt + alias | Solo aliases marcados (FAQ de P3) | **Falsos hits:** dos preguntas parecidas pero distintas ("comisión nacional" vs "internacional"). Por eso hay umbral alto, scope por system prompt, una lista de exclusión de términos numéricos o diferenciadores, y medición de la tasa de falsos hits en el set de evaluación de P3 |

Al servir un hit en modo streaming, se reenvía la respuesta cacheada como chunks, para que el cliente no distinga el camino.

---

## 6. Streaming, cancelación y salida estructurada
- **SSE de salida** con el formato `chat.completion.chunk` de OpenAI, cerrado con `data: [DONE]`, compatible con los SDKs.
- **Traducción del streaming de Anthropic:** `message_start`, `content_block_delta` (text_delta), `message_delta` (usage) y `message_stop` se convierten en chunks de OpenAI más el uso final.
- **Cancelación:** si el cliente se desconecta, se cancela la tarea, se cierra el stream httpx upstream y el proveedor deja de generar. Se liquida el costo de los tokens recibidos. Lo valida un test con un servidor real: corta en el token 5 y verifica que el upstream se cerró.
- **Stall:** si no llega un chunk en `stall_timeout`, se cancela el upstream y se emite un chunk de error.
- **Salida estructurada:** `response_format: {type: json_schema}`. Si la validación falla, se hace un reintento con el error como contexto. Si vuelve a fallar, se responde con un error tipado (`invalid_structured_output`), nunca con JSON roto.

---

## 7. Observabilidad y seguridad

### 7.1 Métricas (Prometheus)
`gw_requests_total{alias,provider,status}` · `gw_ttft_seconds{alias,provider}` · `gw_request_seconds` · `gw_tokens_total{provider,direction}` · `gw_cost_usd_total{provider,client}` · `gw_cache_requests_total{type,result}` · `gw_breaker_state{provider}` · `gw_budget_remaining_usd{scope}` · `gw_fallbacks_total{from,to}` · `gw_stream_outcomes_total{outcome}`. Solo labels acotados: nunca prompts ni IDs de usuario.

### 7.2 Trazas
Un span por request con atributos `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.*` y spans hijos por intento de proveedor (así se ven los fallbacks). Exportación OTLP hacia el collector, Tempo o Langfuse (todo opcional).

### 7.3 Seguridad
- Keys por cliente con hash; rotación vía archivo y recarga sin downtime.
- Secrets de proveedores solo por variables de entorno (Secret Manager en Cloud Run).
- `LOG_PROMPTS=false` por defecto, con un hook de redacción si se activa.
- Límites de tamaño de request (máximo de tokens de entrada) para evitar abusos de costo.
- CORS deshabilitado: es una API de servicio a servicio.

---

## 8. Evaluación
| ID | Prueba | Cómo | Resultado esperado |
|---|---|---|---|
| G1 | **Overhead** | Carga open-loop con k6 o un script contra `mock` con latencia 0, directo vs a través del gateway; p50/p95/p99 | p95 agregado < 5 ms |
| G2 | Overhead vs LiteLLM | Mismo test con LiteLLM proxy | Comparación honesta |
| G3 | **Budget bajo concurrencia** | 50 requests concurrentes con un tope que alcanza para 10 | 0 excesos |
| G4 | Fallback | `mock` primario con 100% de fallas → secundario | ≥ 99% de éxito; breaker abierto en métricas |
| G5 | Cancelación | El cliente corta en el token 5 | Upstream cancelado; tokens liquidados ≈ los consumidos |
| G6 | Caché semántica | Tráfico FAQ de P3 (Zipf) + set de pares "parecidos pero distintos" | Hit rate, costo ahorrado y **tasa de falsos hits** |
| G7 | Contratos | Suite con el SDK oficial de OpenAI (streaming, no streaming, `response_format`, errores) | 100% en verde |
| G8 | Caos | Redis caído, Ollama caído y lento (Toxiproxy) | Comportamiento según §4.5 |

---

## 9. Recursos, ejecución y despliegue
- **RAM:** el gateway ~150 MB y el modelo de embeddings ONNX ~150 MB. Redis se comparte con el proyecto que corra en ese momento. Ollama corre nativo en Windows (`host.docker.internal:11434`).
- **Modos de ejecución:** (1) **contenedor en el Compose de cada proyecto** (el perfil que necesite LLM incluye el servicio `llm-gateway` desde su imagen); (2) **standalone** con `docker compose up` en este repo, para desarrollo y benchmarks; (3) **Cloud Run** en la prueba final de P3 (1 instancia como máximo, ingress interno, keys en Secret Manager).
- **Imagen:** multi-stage con Python 3.12 slim y uv; el modelo de embeddings se descarga en el build o al primer arranque (documentado).

---

## 10. Estructura del repo y del README
```
llm-gateway/
├── README.md · PLAN.md · LICENSE · Makefile · docker-compose.yml · Dockerfile · .env.example · pyproject.toml
├── src/llm_gateway/
│   ├── main.py · settings.py
│   ├── api/ (chat.py, models.py, admin.py, health.py, schemas.py)
│   ├── providers/ (base.py, mock.py, ollama.py, anthropic.py)
│   ├── routing/ · resilience/ · budget/ (lua/) · pricing/ · cache/ · structured/ · security/ · telemetry/
├── config/ (routes.yaml, prices.yaml, clients.example.yaml)
├── tests/ (unit/, contract/, integration/, chaos/)
├── bench/ (overhead.py, k6/)
└── docs/ (adr/, results/)
```
**Esqueleto del README (en inglés):** Part I (problem; core concepts: gateways, circuit breakers, budgets, semantic caching, SSE), Part II (components), Part III (resilience and cost control in depth), Part IV (proof: G1–G8), Part V (run it: standalone, inside P1/P2/P3, Cloud Run), Part VI (reflection: lessons, limitations, why not LiteLLM).

---

## 11. Hitos de implementación
| Hito | Objetivo | Criterio de aceptación |
|---|---|---|
| **M0 · Bootstrap** ✅ | Repo, estructura, configs, CI, README esqueleto | Hecho en el setup inicial |
| **M1 · Núcleo + mock** | `/v1/chat/completions` (normal y streaming) con `MockProvider`, schemas OpenAI, health, métricas básicas | El SDK oficial de OpenAI funciona contra el gateway (G7 parcial) |
| **M2 · Ollama + routing** | `OllamaProvider`, `routes.yaml`, aliases | P2 y P1 pueden apuntar al gateway con `fast` |
| **M3 · Resiliencia** | Timeouts por fase, reintentos, breaker, bulkhead, fallback | G4 y G8 parcial |
| **M4 · Budget + pricing** | Lua de reserva y liquidación, topes, `/admin/budget` | **G3 en verde** |
| **M5 · Streaming robusto** | Cancelación, stalls, `[DONE]`, errores tipados | G5 |
| **M6 · Caché** | Exacta + semántica (Redis 8 vector + ONNX) | G6 con tasa de falsos hits reportada |
| **M7 · Structured + Anthropic** | `response_format`, validación, `AnthropicProvider` con traducción del streaming | Probado **solo contra mocks grabados** ($0); la prueba real queda para la final de P3 |
| **M8 · Observabilidad + seguridad** | OTel GenAI, dashboards, keys por cliente, rate limit | Dashboard del gateway en Grafana |
| **M9 · Benchmarks + publicación** | G1, G2, README completo, `v1.0` | Frase del CV con números |

**Orden respecto de los otros proyectos:** M1–M2 antes del M7 de P1 (explainer) y del M5 de P2 (System 2). M3–M7 antes de la prueba final de P3.

---

## 12. Riesgos y pendientes
| Riesgo | Mitigación |
|---|---|
| Subestimar la traducción del streaming de Anthropic | Fixtures grabados de streams reales (cuando se haga la prueba final) más tests de contrato; empezar por el modo no streaming |
| Falsos hits de la caché semántica (respuestas financieras incorrectas en P3) | Umbral alto, scope por system prompt, exclusión de términos numéricos, medición de G6; la caché semántica está **desactivada por defecto** |
| La búsqueda vectorial de Redis 8 no está en la imagen o versión usada | Verificar en M0/M6; plan B: Qdrant (ya está en el stack de P3) |
| Precios desactualizados | `prices.yaml` versionado; verificarlos antes de la prueba final |
| Scope creep (es tentador construir un LiteLLM entero) | Solo 3 proveedores; las features más allá de §1.2 van a "future work" |

**Por verificar al implementar:** el endpoint OpenAI-compatible de Ollama (streaming y `format`), la forma actual de los eventos de streaming de Anthropic, las APIs de vector search de Redis 8, y FastAPI SSE nativo con `POST` (ya verificado en 0.142).
