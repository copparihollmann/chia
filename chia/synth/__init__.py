"""Open-source synthesis nodes (Yosys + a liberty library).

CHIA's existing synthesis path is Cadence Genus through Hammer, which needs a licence and a full
VLSI flow. This package is the cheap end: Yosys mapping against an ASAP7-style liberty file, for
post-synthesis QoR — mapped area, cell counts, logic depth — where an area *ratio* between designs
is the question and absolute PPA is not.
"""

from chia.synth.yosys_node import YosysNode, YosysResult, yosys_synth

__all__ = ["YosysNode", "YosysResult", "yosys_synth"]
