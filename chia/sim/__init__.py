"""Simulation nodes for plain SystemVerilog.

CHIA's existing Verilator path goes through Chipyard: it builds a `CONFIG=`-selected harness from a
Chisel elaboration. That is the right tool for a Chipyard SoC and the wrong one for a standalone
candidate RTL tree, which has no Chipyard config, no Scala, and no harness — just `.sv` files and a
top module name.
"""

from chia.sim.verilator_sv_node import (
    VerilatorSvNode,
    VerilatorSvResult,
    verilator_elaborate,
    verilator_lint,
)

__all__ = [
    "VerilatorSvNode",
    "VerilatorSvResult",
    "verilator_lint",
    "verilator_elaborate",
]
