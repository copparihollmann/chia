Dependency pins: what they cost, and what relaxing them would take
==================================================================

chia pins three of its dependencies exactly:

.. code-block:: text

   "ray[default]==2.54.0",
   "mcp==1.27.1",
   "pydantic==2.12.4",

Exact pins on a *library* are load-bearing for whoever installs it. This page is the
evidence for relaxing two of them, and it is deliberately evidence rather than a patch:
the change is a maintainer's judgement about support surface, and it should not be made
on the strength of one contributor's environment.

What the pins cost, in the downstream projects' own words
---------------------------------------------------------

Two independent projects consume chia. Neither can install it alongside their own code,
and both say why in their source.

**oscar-merlin** keeps a second virtualenv and makes every chia import function-local
(``merlin/python/merlin/benchharness/chia_bridge.py``):

   *"CHIA lives in its own virtualenv (it hard-pins pydantic/ray[default] and is not
   pip-installable under the name chia), so every chia/ray import here is
   function-local. Importing this module from the main .venv, where Ray is absent, must
   keep working."*

Its error message spells out the consequence:

   *"installing it into the main .venv would downgrade pydantic and perturb concurrent
   sessions."*

The result is a three-hop interpreter dance: the main ``.venv`` launches a script under
``build/chia-venv``, which then shells *back* to the main ``.venv`` for the QA drivers —
which is why ``chia_bridge`` exports a ``driver_python`` helper at all.

**spec** goes further and refuses to import chia in-process ever
(``sites.yaml``):

   *"chia pins Python 3.10.19 + ray==2.54.0 — it CANNOT share this project's venv."*

So every single LLM call goes out as ``uv run --python <chia python> python
_chia_bridge.py`` with a JSON payload over stdio
(``specir/nl2spec/chia_provider.py``). A subprocess and a serialisation boundary per
call, purely because of the pins.

One of those two beliefs is already wrong, which is itself worth reporting: chia
declares ``requires-python = ">=3.10"``, not ``==3.10.19``, and merlin's own setup
instructions create its chia venv with ``--python 3.13``. The *Python* pin is a
misconception; the ``==`` pins on ray and pydantic are real.

Evidence that relaxation is viable
----------------------------------

Measured on this host, 2026-08-05, against ``origin/main`` at ``beda03d``.

A 3.12 environment was built with **relaxed** constraints —
``ray[default]>=2.54,<3`` and ``pydantic>=2.9`` — and resolved to versions *newer* than
chia's pins:

=========================  =====================  =========================
Environment                Interpreter            Resolved
=========================  =====================  =========================
chia's pins                Python 3.13.10         ray 2.54.0, pydantic 2.12.4
relaxed                    Python 3.12.13         ray 2.56.1, pydantic 2.13.4
=========================  =====================  =========================

Offline suite results:

===========================================  ==========================  ==========================
Suite                                        3.13 + exact pins           3.12 + relaxed
===========================================  ==========================  ==========================
``chia/models/tests`` + ``chia/trace/test``  (not re-run under pins)     331 passed, 125 skipped
``chia/base/test``                           9 failed, 43 passed, 1 err  9 failed, 43 passed, 1 err
``chia/cluster/test``                        73 passed, 27 errors        73 passed, 27 errors
===========================================  ==========================  ==========================

The failure *sets* are byte-identical between the two environments — compared by name,
not by count. A newer ray and a newer pydantic change nothing that chia's own tests can
see.

That is necessary but not sufficient. What it does *not* show:

- Anything about Python 3.10 or 3.11, which are inside the declared
  ``requires-python`` range and were not exercised.
- Anything about the live tiers (cluster bring-up, real providers), which are skipped
  offline.
- Anything about ray 3.x, which is why the suggested constraint keeps an upper bound.

Pre-existing failures, correctly attributed
-------------------------------------------

The 9 failures and 1 error above are **not** version-related, and it would be easy to
misread them as evidence for or against a pin change. They reproduce identically under
both environments, and in isolation:

``chia/base/test/test_bypass_tags.py::test_8_real_yaml_file``
   ``fixture 'yaml_path' not found``. The test signature takes a ``yaml_path``
   parameter, which pytest resolves as a fixture, and no such fixture exists — every
   other test in the file builds its YAML with the local ``write_yaml`` helper. This test
   has never executed. A plain latent bug in the test file, independent of any
   dependency.

``chia/base/test/test_cache_tags.py`` (8 failures)
   ``get_active_cache()`` returns ``None`` inside the Ray worker, so the bypass provider
   raises ``AttributeError: 'NoneType' object has no attribute 'read'``. Reproduces when
   the single test is run alone, so it is not test-ordering. A real defect in cache-actor
   discovery from a worker, or a missing fixture that starts the cache — either way,
   unrelated to pins.

``chia/base/test/test_colocated_live.py::test_unsatisfiable_dispatch_raises``
   Fails in a full-directory run and **passes on its own**, so this one *is*
   test-ordering: the files in this directory share a single Ray instance and this test
   depends on the cluster's resource state. A test-isolation problem rather than a code
   defect, and also unrelated to pins. Counted here because an earlier version of this
   page said "8 failures" and omitted it — a dossier arguing for a dependency change has
   to get its own baseline right, or every number after it is suspect.

``chia/cluster/test`` (27 errors)
   Cloud-tunnel tests that need AWS/GCP credentials. Environment, not code.

Reporting these under a pin-relaxation proposal without that attribution would be
misleading in both directions: it would overstate the risk of relaxing, and it would
hide two genuine defects behind a dependency discussion.

What relaxation would take
--------------------------

The proposal, in order:

1. **A CI matrix.** ``.github/workflows/chia-python-matrix.yml`` runs the offline suites
   across Python 3.10–3.13 and across two constraint sets (chia's current pins, and the
   relaxed floor). The pins cannot credibly be relaxed without this, because "it works
   on my machine with 3.12" is exactly the claim a matrix exists to replace.

2. **Relax ray and pydantic, keep an upper bound** —
   ``"ray[default]>=2.54,<3"`` and ``"pydantic>=2.9"``.

   The ray upper bound stays because a major version is a real risk and nothing here
   tests it. ``mcp==1.27.1`` is left alone: the MCP wire protocol is young, chia's tool
   servers depend on its shape, and no evidence was gathered for it.

3. **Make chia pip-installable under its own name**, which is a separate defect merlin
   names explicitly (*"not pip-installable under the name chia"*). Until that is fixed,
   relaxing the pins removes only half the reason both consumers keep a second venv.

The measurable payoff, if all three land: merlin drops its ``build/chia-venv`` and its
function-local-import discipline, and spec drops a subprocess and a JSON bridge per LLM
call. Both are described in their own source as workarounds for this and nothing else.
