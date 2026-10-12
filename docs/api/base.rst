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

Token Usage
-----------

Every :class:`~chia.base.llm_call.QueryResult` carries a
:class:`~chia.base.usage.TokenUsage`, so a caller reads a call's token counts and
cost off the public result rather than a backend's private metadata. The module
docstring explains the three accounting rules it enforces — separate input
classes, an unknown price that is ``None`` rather than zero, and subscription
quota that is never summed with metered spend.

.. automodule:: chia.base.usage

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