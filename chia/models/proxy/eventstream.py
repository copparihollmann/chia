"""AWS event-stream framing, encode side.

Bedrock's ``invoke-with-response-stream`` answers in ``application/vnd.amazon.eventstream``
— AWS's binary framing — and the Claude Code CLI's Bedrock transport parses exactly that.
botocore ships a *decoder* for it (:mod:`botocore.eventstream`) and no encoder, because no
AWS SDK ever needs to produce one. A proxy standing in for Bedrock does.

Why this exists rather than plain SSE
-------------------------------------

The proxy originally replied with ``text/event-stream`` and set
``CLAUDE_CODE_DISABLE_BEDROCK_CONTENT_TYPE_GUARD=1`` on the client to make it accept that.
End to end it looked like it worked: the CLI printed the answer.

It was not working. With per-call usage recording turned on, every proxied turn showed up
at Bedrock **twice** — once on ``/invoke-with-response-stream`` and again, with the same
body, on ``/invoke``. The CLI was failing to parse the SSE stream and silently retrying
non-streaming, then reporting the cost of the one call it accepted. So the deviation was
not a deviation, it was a 2x bill that the client's own accounting could not see. Cache
tokens made it legible: the stream call wrote the prompt cache and the fallback call read
it, which is only possible if both reached the provider.

Proper framing removes the fallback, and with it the guard flag: the proxy now answers in
the content type the CLI actually asked for.

The format, which is all this module implements
-----------------------------------------------

Each message is::

    prelude   uint32 total_length | uint32 headers_length | uint32 crc32(prelude[0:8])
    headers   repeated: uint8 name_len | name | uint8 value_type | uint16 value_len | value
    payload   bytes
    trailer   uint32 crc32(everything before it)

Bedrock's streaming chunks carry three string headers — ``:message-type: event``,
``:event-type: chunk``, ``:content-type: application/json`` — and a payload of
``{"bytes": "<base64 of the model's event JSON>"}``. That double encoding is Bedrock's, not
ours.

Correctness here is not a matter of opinion, so the tests decode what this produces with
**botocore's** parser rather than with a second implementation of these rules.
"""

from __future__ import annotations

import base64
import json
import struct
from binascii import crc32
from typing import Dict, Iterable, Iterator, Optional

#: Header value type tag for a UTF-8 string. The only one Bedrock's chunk headers use, so
#: the only one implemented — an encoder that pretends to support types it never emits is
#: untested surface.
HEADER_TYPE_STRING = 7

#: The content type this framing must be served as. Sending it under any other content
#: type is what makes the CLI fall back to non-streaming.
CONTENT_TYPE = "application/vnd.amazon.eventstream"

#: Headers on a Bedrock streaming chunk, in the order botocore emits them.
CHUNK_HEADERS = {
    ":event-type": "chunk",
    ":content-type": "application/json",
    ":message-type": "event",
}

#: Headers on a modelled error (e.g. a throttle) mid-stream.
def exception_headers(shape: str) -> Dict[str, str]:
    """Headers marking a message as a modelled exception of *shape*.

    :param shape: The Bedrock exception shape name, e.g. ``"throttlingException"``.
    :type shape: str
    :rtype: Dict[str, str]

    A mid-stream failure has to arrive as an exception message rather than as a truncated
    stream, or the client reports a malformed response instead of the throttle it should
    back off from.
    """
    return {
        ":exception-type": shape,
        ":content-type": "application/json",
        ":message-type": "exception",
    }


def encode_headers(headers: Dict[str, str]) -> bytes:
    """Encode *headers* as an event-stream header block.

    :param headers: Header name to string value. Order is preserved.
    :type headers: Dict[str, str]
    :rtype: bytes
    """
    out = bytearray()
    for name, value in headers.items():
        name_bytes = name.encode("utf-8")
        value_bytes = str(value).encode("utf-8")
        if len(name_bytes) > 255:
            raise ValueError(f"header name too long: {name!r}")
        if len(value_bytes) > 0xFFFF:
            raise ValueError(f"header value too long for {name!r}")
        out.append(len(name_bytes))
        out += name_bytes
        out.append(HEADER_TYPE_STRING)
        out += struct.pack(">H", len(value_bytes))
        out += value_bytes
    return bytes(out)


def encode_message(payload: bytes, headers: Optional[Dict[str, str]] = None) -> bytes:
    """One complete event-stream message.

    :param payload: The message body, already serialised.
    :param headers: String headers; defaults to :data:`CHUNK_HEADERS`.
    :type payload: bytes
    :type headers: Optional[Dict[str, str]]
    :rtype: bytes

    Both CRCs are computed over exactly the ranges AWS specifies — the prelude CRC over
    the first 8 bytes only, the message CRC over everything that precedes it. A parser
    rejects the message on either mismatch, and rejection is indistinguishable
    client-side from a network fault, so there is no partial credit here.
    """
    header_block = encode_headers(CHUNK_HEADERS if headers is None else headers)
    total_length = 4 + 4 + 4 + len(header_block) + len(payload) + 4
    prelude = struct.pack(">II", total_length, len(header_block))
    message = bytearray()
    message += prelude
    message += struct.pack(">I", crc32(prelude) & 0xFFFFFFFF)
    message += header_block
    message += payload
    message += struct.pack(">I", crc32(bytes(message)) & 0xFFFFFFFF)
    return bytes(message)


def encode_chunk(event: dict) -> bytes:
    """A model event as a Bedrock streaming chunk message.

    :param event: The Anthropic event dict (``message_start``, ``content_block_delta``, …).
    :type event: dict
    :rtype: bytes

    Bedrock wraps the model's JSON in ``{"bytes": "<base64>"}`` inside the frame payload.
    The double encoding is Bedrock's convention; a client that unwraps one layer and not
    the other sees gibberish, so it is done here rather than left to the caller.
    """
    inner = json.dumps(event, separators=(",", ":")).encode("utf-8")
    payload = json.dumps(
        {"bytes": base64.b64encode(inner).decode("ascii")},
        separators=(",", ":"),
    ).encode("utf-8")
    return encode_message(payload)


def encode_exception(shape: str, message: str) -> bytes:
    """A modelled Bedrock exception as an event-stream message.

    :param shape: Exception shape name, e.g. ``"throttlingException"``.
    :param message: Human-readable reason.
    :type shape: str
    :type message: str
    :rtype: bytes
    """
    payload = json.dumps({"message": message}, separators=(",", ":")).encode("utf-8")
    return encode_message(payload, exception_headers(shape))


def frames_from_sse(sse_frames: Iterable[bytes]) -> Iterator[bytes]:
    """Re-encode a stream of Anthropic SSE frames as event-stream messages.

    :param sse_frames: Frames as produced by
        :func:`chia.models.proxy.translate.to_anthropic_sse`.
    :type sse_frames: Iterable[bytes]
    :rtype: Iterator[bytes]

    A shim, so the translation layer keeps producing one thing and the transport decides
    how to frame it. Parsing our own SSE back would be a second implementation of the
    frame contract; instead this reads only the ``data:`` line, which is the payload the
    binary framing needs anyway.

    A frame whose data is not JSON is skipped rather than forwarded: an unparseable
    payload inside valid framing fails further from the cause than dropping it does.
    """
    for frame in sse_frames:
        for line in frame.decode("utf-8", errors="replace").splitlines():
            if not line.startswith("data:"):
                continue
            try:
                event = json.loads(line[len("data:"):].strip())
            except ValueError:
                continue
            if isinstance(event, dict):
                yield encode_chunk(event)
