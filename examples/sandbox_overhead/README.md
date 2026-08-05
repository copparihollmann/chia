# sandbox_overhead — what per-call isolation costs

`bwrap` and `docker` are not alternatives to each other in the way the framing usually
suggests. bwrap has no daemon and no instance model, so it cannot replace a long-lived
worker container — there is nothing to `exec` into. What it *can* do, and what
`chia.base.sandbox` uses it for, is wrap **one call**.

So the real question is not "which one", it is "how much does each add per call", because
a grid of hundreds of agent calls pays that cost hundreds of times.

```bash
python examples/sandbox_overhead/bench_sandbox.py --repeats 200   # measure
python examples/sandbox_overhead/bench_sandbox.py --plot          # draw from the CSV
```

The command benchmarked is `true` — a no-op — so the number is setup overhead and
nothing else. No model, no network, no tokens, $0.

## Result on this host (n = 200 each, interleaved)

| backend | median | p90 | added per call | over a 500-call grid |
|---|---|---|---|---|
| `none` | 0.84 ms | 1.11 ms | — | — |
| `bwrap` | 7.74 ms | 8.80 ms | +6.9 ms | +3 s |
| `docker` | 266 ms | 278 ms | +265 ms | +133 s |

Per call, docker costs **38x** what bwrap does. Over a 500-call grid that is 2.2 minutes
of pure container start-up versus 3 seconds.

That is the measurable half of the bwrap argument. The unmeasurable half matters at
least as much on a shared cluster: bwrap needs no daemon, no image, and no membership
of the `docker` group — none of which you get on a login node you do not administer.

## How the measurement is set up, and why

- **Interleaved, not blocked.** One repeat runs every backend before the next repeat
  starts. A block-per-backend design attributes any drift in host load to whichever
  backend happened to be running during it.
- **One untimed warm-up per backend**, so the first measured call does not carry
  page-cache and image-layer costs no later call pays.
- **A tiny image (`alpine:3`)** for docker, so the figure measures the daemon's
  start-up rather than a 4.8 GB agent image's layers. Using the real
  `chia-claude-code` image would make docker look worse, not better — this is the
  favourable case for it.
- **Absolute milliseconds are host-specific.** Container start-up is dominated by
  daemon and storage-driver behaviour that varies enormously between machines. What
  generalises is the ratio and the shape of the distribution, and the figure says so.

## Two design facts this benchmark discovered

Both were bugs found by running it, not by reading the code:

1. **`bwrap_available()` was probing wrong.** It ran `bwrap --ro-bind /usr /usr --dev
   /dev true` — a bare `true`, resolved *inside* the sandbox where `/bin` is not bound.
   bwrap worked fine; the probe reported it unusable. Fixed to bind the real system
   directories and invoke the probe binary by absolute path. This is the same class of
   mistake `resolve_executable` exists to prevent, which is a fair sign the mistake is
   easy to make.

2. **`SandboxSpec.extra_binds` must not be translated for docker.** Mounting the host's
   `/usr/bin` over an Alpine image's replaces musl-linked binaries with glibc-linked
   ones, and every command in the container dies with the dynamic loader's `no such
   file or directory`. In the container model the tools come from the image, so
   `docker_prefix` ignores host toolchain binds and the docker backend requires an
   image that already contains the agent CLI. `wrap_argv` likewise leaves `argv[0]`
   bare under docker and resolves it to an absolute host path only for `bwrap` and
   `none`.
