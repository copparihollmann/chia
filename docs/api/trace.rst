Tracing and Metrics
===================

API reference for :mod:`chia.trace`. These pages are generated from the docstrings in the source, so they stay in sync with the code.

:mod:`chia.trace` covers two independent concerns. The **profiler** records a per-call trace of a
running flow — which worker ran what, how long it took, and how results depend on one another — and
is a no-op unless a collector actor is running. The **metrics logger** is a separate, head-node-only
sink for scalar time series (loss, score, cost per iteration) that forwards to TensorBoard, W&B, or a
backend you register yourself.

See the :doc:`../user_guides/profiling` guide for how to turn the profiler on and read its output.

Metrics
-------

.. automodule:: chia.trace.metrics

Profiler
--------

.. automodule:: chia.trace.profiler

Profile Table
-------------

.. automodule:: chia.trace.profile_table

Profile Visualization
---------------------

.. automodule:: chia.trace.profile_viz

Profile Visualization (HTML)
----------------------------

.. automodule:: chia.trace.profile_viz_html

Flow Visualization
------------------

.. automodule:: chia.trace.viz
