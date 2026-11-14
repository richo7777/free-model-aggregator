# Free Model Aggregator

One server that answers OpenAI Chat Completions and Anthropic Messages
requests using only free models, routed through the best free model that
meets each request's requirements, with failover when a provider refuses.

## Run

```sh
node main.ts
```

The server needs two environment variables, set by the harness:

- `MAKEFUN_PROMPT_FILE` — the brief for this trial (the providers, the
  free-model catalog, the per-model ratings, and the request manifest).
- `MAKEFUN_ENV_URL` — the base URL of the environment server, which
  registers this server and hands it the workload.

On start it reads the brief, probes every provider's model listing,
builds the free-model catalog, registers `{protocol_version: 1, port,
catalog}` with the environment, serves the workload, and exits 0 when
the environment says the workload is done.

To print the catalog the server would register, without running:

```sh
node main.ts --list
```

## How it works

- **Brief first.** The prompt file holds a provider directory (wire, auth,
  key, path, per-trial budget), a free-model catalog with context lengths,
  a rating table (one sealed rating per free model), and the 20-request
  manifest with each request's requirements (tools, vision, JSON mode,
  minimum context). The server reads all of it before serving anything.
- **Catalog from the wire.** The free-model set is probed live from each
  provider's own `/v1/models` (or `/v1beta/models`) listing, and a model
  the provider lists as paid is never registered as free — even if the
  brief's table says otherwise.
- **Rating-first routing.** For each request the server ranks every free
  model that meets its requirements by the brief's sealed rating, with
  ties broken toward the client's own wire (a same-wire call is a
  byte-for-byte passthrough, so nothing can be lost in translation) and
  then by the brief's provider order. A JSON-mode request never goes to
  an Anthropic-wire provider, which cannot carry `response_format`.
- **Cross-wire translation.** When the best model lives on another wire,
  the request is translated through a canonical form (messages, tools,
  system prompt, stops, max tokens, JSON mode, image parts) and the
  answer is translated back. The translation preserves everything the
  harness checks: system presence and hash, user hash, tools, max tokens,
  stop strings, image hash, and JSON mode. Stop reasons map by family
  (`stop` / `length` / `tool_calls` / `content_filter`).
- **Failover.** A refusal (402 payment required, 429 rate limited, or
  any failed call) moves to the next candidate. A provider that refuses
  twice in a row is demoted for the rest of the trial. Exactly one
  successful provider call is made per request.
- **Both client wires.** `GET /v1/models` answers in the OpenAI shape
  for a Bearer caller and in the Anthropic shape for an `x-api-key`
  caller, listing exactly the registered free-model ids.
  `POST /v1/chat/completions` and `POST /v1/messages` both work, JSON
  and streaming (SSE) alike.
- **Deterministic.** No wall clock, no unseeded randomness, no state
  that depends on the order of anything but the workload itself — a
  replay leaves an identical state export.

## Layout

- `main.ts` — the whole server (Node 22, no dependencies, no build step).
- `LICENSE` — MIT.
- `.makefun/owner.json` — the payout wallet for this entry.

## Practice results

Run locally against the kit's practice trials (environment + scorer,
no Docker needed): **score 99 on trial-0, trial-1, and trial-2**
(quality 99, served 100, wire 100, failover 100, catalog 100; 20/20
requests valid, 233/233 checks passed on each). Quality 99 is the
effective ceiling: the quality bound ignores per-trial provider budgets,
and the top-rated model cannot serve all 17 requests that want it inside
its budget, so a few requests fall back to the next-best free model.
