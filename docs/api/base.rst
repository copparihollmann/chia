Base Runtime
============

API reference for :mod:`chia.base`. These pages are generated from the docstrings in the source, so they stay in sync with the code.

Bypass
------

.. automodule:: chia.base.bypass

Cache
-----

.. automodule:: chia.base.cache

Chia Function
-------------

.. automodule:: chia.base.ChiaFunction


Chia Wait
---------

.. automodule:: chia.base.chia_wait

Chia KV Store
-------------

.. automodule:: chia.base.chia_kv_store

Colocated
---------

.. automodule:: chia.base.colocated

Dispatch Proxy
--------------

.. automodule:: chia.base.dispatch_proxy

PID Registry
------------

.. automodule:: chia.base.pid_registry


LLM Call
--------

.. automodule:: chia.base.llm_call

Budget
------

A pre-flight spend ceiling. :func:`~chia.base.budget.check_budget` refuses a fan-out
*before* it is dispatched, projecting the grid's cost from the run's own recorded
per-call spend rather than from a price table. Subscription quota is reported beside the
metered figure and never counted against a dollar cap.

.. automodule:: chia.base.budget

Token Usage
-----------

Every :class:`~chia.base.llm_call.QueryResult` carries a
:class:`~chia.base.usage.TokenUsage`, so a caller reads a call's token counts and
cost off the public result rather than a backend's private metadata. The module
docstring explains the three accounting rules it enforces — separate input
classes, an unknown price that is ``None`` rather than zero, and subscription
quota that is never summed with metered spend.

.. automodule:: chia.base.usage
Rate limits
-----------

Waiting out a usage window, as a loop primitive rather than as each caller's problem.
Opt in with ``ClaudeCodeLLM(rate_limit_policy=RateLimitPolicy(max_waits=N))``; without
one, a limit propagates exactly as before. Pair it with the
:data:`~chia.base.ratelimit.CLAUDE_SESSION_RESOURCE` Ray gate, since a subscription seat
is a single concurrency slot and fanning ten calls at it produces ten rate limits rather
than one.

.. automodule:: chia.base.ratelimit

MCP tool servers
----------------

Chia Tool
~~~~~~~~~

.. automodule:: chia.base.tools.ChiaTool

Chia Tool Template
~~~~~~~~~~~~~~~~~~

.. automodule:: chia.base.tools.ChiaToolTemplate

Bash Tool
~~~~~~~~~

.. automodule:: chia.base.tools.BashTool

Async Bash Tool
~~~~~~~~~~~~~~~

.. automodule:: chia.base.tools.AsyncBashTool

Async Job Tool
~~~~~~~~~~~~~~

.. automodule:: chia.base.tools.AsyncJobTool


Util
~~~~

.. automodule:: chia.base.tools.util