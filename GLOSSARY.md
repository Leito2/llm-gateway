# Glosario

Términos fijos del proyecto. Cada concepto tiene **un solo nombre**: usarlo igual en `PLAN.md`, en el código, en los ADRs y en las conversaciones. El README (en inglés) usa el término en inglés de la segunda columna.

## Actores y routing

| Término | En inglés / código | Definición |
|---|---|---|
| **Cliente** | client | Servicio o persona que llama al gateway con su propia key: P1, P2, P3, P4, la UI o `demo`. Se define en `config/clients.yaml`. |
| **Proveedor** | provider | Servicio que genera la respuesta: `mock`, `ollama`, `groq`, `google`, `anthropic`. Se define en `config/providers.yaml`. |
| **Proveedor pago** | paid provider | Proveedor que cobra dinero (`paid: true`). Hoy solo `anthropic`. Es el único que reserva presupuesto. |
| **Proveedor degradado** | degraded (`degraded: true`) | Eslabón de baja calidad (`mock`, `gemma3:1b`). Sus respuestas no se cachean, y los aliases con `allow_degraded: false` no lo usan. |
| **Alias** | alias | Nombre corto del tipo de trabajo que pide el cliente (`fast`, `smart`, `judge`, `smart_paid`, …), en lugar de un modelo. Se define en `config/routes.yaml`. |
| **Cadena** | chain | Lista ordenada de pares proveedor + modelo de un alias. El gateway prueba el primero y, si falla, el siguiente. |
| **Adapter** | adapter | Código que traduce entre el formato OpenAI y el de un proveedor: `OpenAICompatProvider` (Ollama, Groq, Google) y `AnthropicProvider`. |

## Request y respuesta

| Término | En inglés / código | Definición |
|---|---|---|
| **Request** | request | Una llamada del cliente a `/v1/chat/completions`. |
| **Stream** | stream (SSE) | Respuesta enviada en pedazos a medida que el proveedor genera texto. |
| **Chunk** | chunk | Un pedazo de un stream (`chat.completion.chunk`). |
| **Primer token / TTFT** | time to first token | Tiempo hasta el primer pedazo de texto. Antes de él se puede reintentar o cambiar de proveedor; después, no (ADR-4). |
| **Stall** | stall | Stream que deja de enviar chunks durante más de `stall_timeout_s`. |
| **Stream simulado** | replayed stream | Respuesta completa reenviada como stream (hits de caché, singleflight, JSON validado). |
| **Rechazo** | refusal → `content_filter` | El modelo se niega a responder. No hay fallback ni caché. |
| **Clamp** | clamp | Recortar un valor del cliente al tope del alias (p. ej., `max_tokens`). |

## Resiliencia

| Término | En inglés / código | Definición |
|---|---|---|
| **Fallback** | fallback | Pasar al siguiente proveedor de la cadena. |
| **Reintento** | retry | Repetir la llamada al mismo proveedor después de una espera aleatoria (jitter). Máximo 1 por proveedor. |
| **Intento** | attempt | Cada llamada a un proveedor dentro de una request. Máximo 5 por request. |
| **Circuit breaker** | breaker | Mecanismo que deja de llamar a un proveedor que falla mucho. Estados: cerrado, abierto, semiabierto. |
| **Load shedding** | load shedding | Rechazar a propósito requests `batch` cuando el gateway está saturado, para proteger las `interactive`. |
| **Prioridad** | priority (`interactive`, `batch`) | Importancia de una request. `interactive`: una persona espera. `batch`: un proceso automático. |
| **Singleflight** | singleflight | Varias requests idénticas y cacheables en vuelo comparten una sola llamada al proveedor. |
| **Hedging** | hedging (stretch) | Enviar la misma request a un segundo proveedor si el primero tarda, y usar la primera respuesta. |
| **Cuota** | quota | Límite de uso que impone un proveedor gratis (RPM, TPM, RPD). |
| **Rate limit** | rate limit | Límite de requests por minuto que el gateway impone a cada cliente. |

## Presupuesto

| Término | En inglés / código | Definición |
|---|---|---|
| **Presupuesto (budget)** | budget | Tope de gasto en dinero. Hay 3 topes: por corrida, por cliente por día y global por día (día UTC). |
| **Reserva** | reservation | Monto máximo posible de una request, apartado en Redis antes de llamar a un proveedor pago. |
| **Liquidación** | settle | Al terminar: liberar la reserva y sumar el costo real al gasto. |
| **Gasto** | spent | Suma de costos reales liquidados. |
| **Micro-dólar** | micro-dollar | Unidad interna de dinero: 1 USD = 1 000 000. Siempre entero. |
| **Costo de referencia** | reference price | Lo que costaría un proveedor gratis si se pagara. Solo para medir ahorro. |
| **Prompt caching** | prompt caching | Descuento del proveedor por prefijos de prompt repetidos (`cache_control` en Anthropic). Distinto de la caché del gateway. |

## Caché del gateway

| Término | En inglés / código | Definición |
|---|---|---|
| **Caché exacta** | exact cache (L1) | Encuentra requests con texto idéntico (hash). |
| **Caché semántica** | semantic cache (L2) | Encuentra preguntas con el mismo significado, comparando embeddings. |
| **Embedding** | embedding | Lista de números que representa el significado de un texto. |
| **Embedder** | embedder | Modelo que crea embeddings (`e5-small` por defecto). |
| **Similitud** | similarity | Número entre 0 y 1 que dice cuánto se parecen dos preguntas. |
| **Umbral alto / bajo** | `t_alto` / `t_bajo` | Similitud mínima para aceptar un hit directo / para entrar en la zona gris. Se calibran por embedder con G6. |
| **Zona gris** | gray zone | Similitud entre el umbral bajo y el alto. Ahí decide el verificador. |
| **Verificador** | verifier | Modelo pequeño (cross-encoder) que confirma si dos preguntas son equivalentes. |
| **Guardas léxicas** | lexical guards | Reglas que rechazan un hit si cambian números, montos, fechas o negaciones. |
| **Hit / miss** | hit / miss | La caché encontró / no encontró una respuesta usable. |
| **Falso hit** | false hit | Hit que devuelve la respuesta de otra pregunta. Meta: < 1%. |
| **Namespace** | namespace | Partición de la caché por cliente, alias, system prompt y embedder. Dos namespaces nunca comparten respuestas. |
| **Modo observación** | shadow mode | La caché semántica registra el hit que habría servido, pero no lo sirve. Se usa hasta calibrar el umbral. |
| **Tag** | tag | Etiqueta de una entrada (p. ej., `doc:123`) para borrarla cuando cambia un documento. |

## Operación y evaluación

| Término | En inglés / código | Definición |
|---|---|---|
| **Core / stretch** | core / stretch | Core: entra en la v1.0 y en la frase del CV. Stretch: solo si sobra tiempo (§1.6). |
| **G1–G15** | evals | Pruebas que producen los números del proyecto (§10). |
| **R01–R23** | EARS requirements | Requisitos verificables; cada uno tiene un test (§11). |
| **Prueba final de P3** | P3 final test | Única corrida con Anthropic real, en Cloud Run (ADR-0006). |
| **Métricas / logs / trazas** | metrics / logs / traces | Estado general (Prometheus) / una línea por request / línea de tiempo de una request (OTel, Jaeger). |
