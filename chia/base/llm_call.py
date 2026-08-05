
import warnings
from typing import List, Optional, Sequence, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from chia.base.tools.ChiaTool import ChiaTool
from chia.base.sandbox import SandboxSpec, wrap_argv
from chia.base.usage import BillingMode, RetryAttempt, TokenUsage, sum_usages


# Sentinel for "argument not provided". Lets LLMCallBase tell an explicit value
# (which warrants a warning on a backend that ignores it) from the unset default.
UNSET = object()


@dataclass
class QueryResult:
    """
    Structured result from prompting an LLM or agent.
    
    :param result: The final response from the LLM or agent
    :param returncode: The returncode from running the prompt
    :param stderr: The stderr output from running the prompt (clis only)
    :param stream_result: The full transcript of all turns of the LLM or agent
    :param success: Whether the prompt completed successfully
    :param usage: Token and cost accounting for the call
    :param retry_attempts: The failed attempts the call retried through
    :type result: str
    :type returncode: int
    :type stderr: str
    :type stream_result: str
    :type success: bool
    :type usage: TokenUsage
    :type retry_attempts: Tuple[RetryAttempt, ...]

    ``usage`` is the public, backend-independent view of what the call consumed:
    the three separately-priced input classes, output and reasoning tokens, turns,
    and a cost annotated with where it came from (see :mod:`chia.base.usage`).
    Every backend populates it via :meth:`LLMCallBase.attach_usage`, so callers no
    longer need to read a backend's private ``_last_metadata``. An unattributed
    result (an error path, or a backend the provider gave no counts for) carries an
    all-zero usage whose ``cost_source`` is ``"unavailable"`` rather than ``None``,
    so ``result.usage.input_tokens`` is always safe to read.

    ``usage`` covers the *whole* retry sequence, not just the attempt that
    succeeded — a retried call is billed for every attempt. ``retry_attempts``
    itemises the failures behind that total (error class, backoff, and what each
    one consumed), so a caller can tell a clean call from one that reached the same
    answer after burning three attempts.
    """

    result: str
    returncode: int
    stderr: str
    stream_result: str
    success: bool = False
    usage: TokenUsage = field(default_factory=TokenUsage)
    retry_attempts: Tuple[RetryAttempt, ...] = ()

class LLMCallBase(ABC):
    """
    Polymorphic base container for generic LLM and agent
    configuration traits and behavior. Easy to switch between
    different backing providers, servers, and CLIs
    """

    # Capability flags — subclasses that honor these permission controls
    # override them to True. When False (the default), passing the corresponding
    # argument emits a warning that it will be ignored (see __init__).
    supports_dangerously_skip_permissions: bool = False
    supports_config: bool = False

    # How this backend's calls are paid for. Metered per-token billing is the
    # default because it is the only mode whose cost may be summed into a bill;
    # a backend authenticated by a seat/subscription overrides this (usually as a
    # property, since it can depend on which credentials the worker actually
    # found). See :mod:`chia.base.usage` for why the distinction is tracked.
    billing_mode: BillingMode = "per_token"

    # Whether this backend spawns a subprocess that a per-call sandbox can wrap.
    # False for the raw-API backends (bedrock, vertex, openai_compat, the claude
    # "api" backend): there is no child process, the request goes out over HTTP from
    # this one, and pretending otherwise would report isolation that does not exist.
    supports_sandbox: bool = False

    def __init__(
        self,
        system_message: str,
        dangerously_skip_permissions=UNSET,
        config=UNSET,
        sandbox_spec=UNSET,
        sandbox_backend: str = "none",
    ):
        self.system_message = system_message
        cls = type(self).__name__
        if (dangerously_skip_permissions is not UNSET
                and not self.supports_dangerously_skip_permissions):
            warnings.warn(
                f"{cls} does not support 'dangerously_skip_permissions'; the "
                f"argument is ignored (this backend has no permission gate).",
                stacklevel=2,
            )
        if config is not UNSET and not self.supports_config:
            warnings.warn(
                f"{cls} does not support a 'config' block; the "
                f"argument is ignored.",
                stacklevel=2,
            )
        # Maps ONLY to the backend's "dangerously skip permissions" CLI flag
        # (claude/opencode/antigravity --dangerously-skip-permissions, codex
        # --dangerously-bypass-approvals-and-sandbox, copilot --allow-all).
        # Honored only where supports_dangerously_skip_permissions is True.
        self.dangerously_skip_permissions = (
            True if dangerously_skip_permissions is UNSET else dangerously_skip_permissions
        )
        # The backend's config block (e.g. opencode's `permission`
        # object). ``None`` means "allow all". Honored only where
        # supports_config is True.
        self.config = None if config is UNSET else config
        # Optional record/replay store; see :meth:`recorded_prompt`. None means no
        # recording, which is the default: a cassette that intercepted every call
        # implicitly would be a surprising thing for a framework to do.
        self.cassette = None

    # ------------------------------------------------------------------
    # Record / replay
    # ------------------------------------------------------------------

    @property
    def provider_id(self) -> str:
        """Backend identity for a cassette key — the class name.

        The same prompt through a subprocess CLI and through an API is not the same call, so
        ``ClaudeCodeLLM`` and ``BedrockLLM`` must never share a recording even on an identical
        model id. Overridable by a backend that fronts more than one transport.
        """
        return type(self).__name__

    def recorded_prompt(
        self,
        user_message: str,
        tools: Optional[List[ChiaTool]] = None,
        *,
        variant: str = "",
    ) -> QueryResult:
        """:meth:`prompt`, served from ``self.cassette`` when it has this call recorded.

        :param user_message: The prompt.
        :param tools: Passed through to :meth:`prompt`.
        :param variant: Discriminator for two calls that send identical bytes — in practice the
            repetition index. **Pass it for every repetition of a grid cell.** Without it,
            repetition 2 of a condition is a cache hit on repetition 1, and a run that looks
            like N samples per cell is one sample reported N times.
        :type user_message: str
        :type tools: Optional[List[ChiaTool]]
        :type variant: str
        :rtype: QueryResult
        :raises chia.base.cassette.CassetteMiss: In ``replay_only`` mode with nothing recorded.

        Defined here rather than in each backend so every backend inherits it, and calls
        ``self.prompt`` polymorphically so a backend needs no changes to become recordable.
        With no cassette set this is exactly :meth:`prompt`, minus the ``@ChiaFunction``
        dispatch — a caller wanting remote execution *and* recording should set the cassette on
        the driver and call this from the driver, because a per-worker cassette would shard the
        store by worker and lose most of its hits.
        """
        import time

        cassette = self.cassette
        if cassette is None:
            return self.prompt(user_message, tools if tools is not None else [])

        from chia.base.cassette import CassetteEntry

        model_version = getattr(self, "model", "") or ""
        key, entry = cassette.lookup(
            user_message, self.provider_id, model_version,
            system_message=getattr(self, "system_message", "") or "",
            variant=variant,
        )
        if entry is not None:
            # The recorded usage is returned verbatim: re-deriving the same cost with nothing
            # spent is the point. It is the *original* call's usage, so a spend rollup must
            # consult Cassette.summary() rather than adding a replayed run to a ledger.
            return entry.result

        started = time.perf_counter()
        result = self.prompt(user_message, tools if tools is not None else [])
        cassette.put(CassetteEntry(
            key=key,
            prompt=user_message,
            provider_id=self.provider_id,
            result=result,
            model_version=model_version,
            system_message=getattr(self, "system_message", "") or "",
            variant=variant,
            original_wall_s=time.perf_counter() - started,
        ))
        return result

        if sandbox_spec is not UNSET and not self.supports_sandbox:
            warnings.warn(
                f"{cls} does not support a per-call sandbox; 'sandbox_spec' is "
                f"ignored. This backend issues its request from this process rather "
                f"than spawning an agent, so there is no child to isolate.",
                stacklevel=2,
            )
        # What one call may see, and how to enforce it. ``None`` plus
        # sandbox_backend="none" is the default: unisolated, and recorded as such.
        self.sandbox_spec: Optional[SandboxSpec] = (
            None if sandbox_spec is UNSET else sandbox_spec)
        self.sandbox_backend = sandbox_backend

    def sandbox_argv(self, argv: Sequence[str]) -> List[str]:
        """Return *argv* wrapped for this instance's sandbox.

        :param argv: The command a backend is about to spawn.
        :type argv: Sequence[str]
        :rtype: List[str]
        :raises chia.base.sandbox.SandboxError: The configured backend is unusable
            on this host. Deliberately fatal: silently running unisolated would
            invalidate every number the run produces, and a run whose isolation
            claim is wrong is worse than one that failed.

        Every CLI backend calls this immediately before spawning. It also resolves
        ``argv[0]`` to an absolute path even with no sandbox configured, because a
        bare program name resolves against the *sandbox's* ``PATH`` once wrapped —
        and a not-found binary surfaces as an empty response, indistinguishable from
        a model that declined to answer.
        """
        return wrap_argv(argv, self.sandbox_spec, self.sandbox_backend)

    def attach_usage(
        self,
        result: QueryResult,
        meta: Optional[dict] = None,
        *,
        model: str = "",
    ) -> QueryResult:
        """Populate ``result.usage`` from this call's raw usage *meta*.

        :param result: The result to annotate, returned unchanged for chaining.
        :param meta: Raw per-call metadata; defaults to this instance's
            ``_last_metadata`` when omitted.
        :param model: Model id override; defaults to ``self.model`` when the
            backend has one.
        :type result: QueryResult
        :type meta: Optional[dict]
        :type model: str
        :rtype: QueryResult

        Every backend calls this at the same point in ``prompt`` — once the raw
        metadata is final and before the result is handed back — so that
        ``billing_mode`` and the cost-source resolution are applied identically
        across backends instead of once per backend.

        The retry ledger is folded in here, so ``result.usage`` covers every
        attempt rather than only the one that succeeded, and ``result.retry_attempts``
        itemises the failures. Pass ``meta={}`` on the all-attempts-failed path,
        where there is no successful attempt to account for but the failures still
        need reporting.
        """
        if meta is None:
            meta = getattr(self, "_last_metadata", None)
        usage = TokenUsage.from_metadata(
            meta,
            model=model or getattr(self, "model", "") or "",
            billing_mode=self.billing_mode,
        )
        result.usage = sum_usages([usage] + self._billable_retry_usages())
        result.retry_attempts = self.retry_attempts
        # Mirror the two resolved annotations back onto the raw metadata, which is
        # what the profiler writes to its trace. Without them a reader of the trace
        # cannot tell subscription quota from metered spend, nor a billed figure
        # from an estimate — exactly the two distinctions the accounting turns on,
        # and both are decided here rather than being present in the raw counts.
        if isinstance(meta, dict) and meta:
            meta.setdefault("cost_source", result.usage.cost_source)
            meta.setdefault("billing_mode", result.usage.billing_mode)
        return result

    # ------------------------------------------------------------------
    # Retry accounting
    # ------------------------------------------------------------------
    #
    # A retried call is billed for every attempt, not just the one that
    # succeeded. Backends reset their per-attempt metadata at the top of each
    # attempt, so without a ledger the tokens burned on attempts that raised are
    # simply gone by the time a result exists to report them on.

    def begin_retry_ledger(self) -> None:
        """Start a fresh retry ledger. Called once per ``prompt``, before the loop.

        An explicit reset (rather than clearing on read) keeps a call that
        propagated an exception from leaking its attempts into the next call.
        """
        self._retry_attempts: List[RetryAttempt] = []

    def note_retry(
        self,
        attempt: int,
        exc: BaseException,
        *,
        backoff_s: float = 0.0,
        meta: Optional[dict] = None,
    ) -> RetryAttempt:
        """Record a failed attempt that is about to be retried.

        :param attempt: 0-based attempt index, as the backends' loop counters run;
            stored 1-based on the :class:`~chia.base.usage.RetryAttempt`.
        :param exc: The exception that ended the attempt.
        :param backoff_s: Seconds about to be slept, for an exponential-backoff
            branch; ``0.0`` for an immediate retry.
        :param meta: Raw usage metadata for the failed attempt; defaults to this
            instance's ``_last_metadata``, which still holds it at this point.
        :type attempt: int
        :type exc: BaseException
        :type backoff_s: float
        :type meta: Optional[dict]
        :rtype: RetryAttempt

        Call this from each retrying ``except`` branch, before the backoff sleep.
        It both accumulates the attempt's tokens (so the eventual
        :attr:`QueryResult.usage` covers the whole sequence) and emits a
        ``llm_retry`` profiler event — previously a retry left nothing behind but a
        ``logger.warning``, so retry rates could not be measured after the fact.

        A typed error the backend re-raises instead of retrying (a rate limit, an
        auth failure) is *not* recorded here: it produces no result to carry the
        accounting on. Those tokens are visible only in the profiler log.
        """
        if meta is None:
            meta = getattr(self, "_last_metadata", None)
        record = RetryAttempt(
            attempt=attempt + 1,
            error_type=type(exc).__name__,
            error=str(exc),
            backoff_s=backoff_s,
            usage=TokenUsage.from_metadata(
                meta,
                model=getattr(self, "model", "") or "",
                billing_mode=self.billing_mode,
            ),
        )
        if not hasattr(self, "_retry_attempts"):
            self.begin_retry_ledger()
        self._retry_attempts.append(record)

        try:
            from chia.trace.profiler import get_profiler

            get_profiler().log_event(
                "llm_retry",
                backend=type(self).__name__,
                total_attempts=getattr(self, "retries", 0),
                **record.as_event(),
            )
        except Exception:  # telemetry must never fail a call
            pass
        return record

    @property
    def retry_attempts(self) -> Tuple[RetryAttempt, ...]:
        """The failed attempts recorded for the call in progress.

        :rtype: Tuple[RetryAttempt, ...]
        """
        return tuple(getattr(self, "_retry_attempts", ()))

    def _billable_retry_usages(self) -> List[TokenUsage]:
        """Ledger entries that actually consumed tokens.

        Zero-token attempts are dropped rather than summed: an attempt that died
        before the provider reported anything changed no number, and folding its
        empty usage in would needlessly degrade the total's ``cost_source`` from
        ``"billed"`` to ``"estimated"``.
        """
        return [a.usage for a in self.retry_attempts if a.usage.total_tokens]

    @abstractmethod
    def prompt(self, user_message: str, tools: Optional[List[ChiaTool]] = []) -> QueryResult:
        """
        Send a prompt to this LLM

        :param user_message: Message used to prompt the LLM
        :param tools: Tools available to the LLM during the call
        :type user_message: str
        :type tools: Optional[List[ChiaTool]]
        """
        pass
