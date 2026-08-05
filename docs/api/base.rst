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