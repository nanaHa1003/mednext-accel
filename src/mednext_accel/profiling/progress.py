"""Terminal-independent profiling progress events and reporters."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Literal, Protocol, TextIO

ProgressKind = Literal[
    "environment", "workload", "stage_start", "status", "item_start", "advance", "complete"
]


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    kind: ProgressKind
    stage: str
    completed: int | None = None
    total: int | None = None
    message: str = ""
    secondary_completed: int | None = None
    secondary_total: int | None = None


class ProgressReporter(Protocol):
    def emit(self, event: ProgressEvent) -> None: ...

    def close(self) -> None: ...


def _duration(seconds: float) -> str:
    value = max(0, round(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _progress_facts(
    stage: str,
    completed: int,
    total: int | None,
    elapsed: float,
    secondary_completed: int | None = None,
    secondary_total: int | None = None,
) -> str:
    if stage == "batch-search":
        noun, unit = "probes", "probe"
    elif stage == "kernel-groups":
        noun, unit = "groups", "group"
    elif stage == "validation":
        noun, unit = "validations", "validation"
    else:
        noun, unit = "items", "item"

    count = f"{completed}/{total if total is not None else '?'}"
    parts = [f"{noun} {count}"]
    if total is not None and total > 0:
        parts.append(f"{completed / total * 100:.1f}%")
    if secondary_completed is not None and secondary_total is not None:
        parts.append(f"experiments {secondary_completed}/{secondary_total}")
    parts.append(f"elapsed {_duration(elapsed)}")
    if completed > 0:
        mean = elapsed / completed
        parts.append(f"{mean:.1f} s/{unit}")
        if total is not None:
            parts.append(f"ETA {_duration((total - completed) * mean)}")
    return " ".join(parts)


class NullProgressReporter:
    def emit(self, event: ProgressEvent) -> None:
        pass

    def close(self) -> None:
        pass


class PlainProgressReporter:
    def __init__(self, stream: TextIO = sys.stderr, *, clock=time.monotonic) -> None:
        self.stream = stream
        self.clock = clock
        self._started: dict[str, float] = {}
        self._completed: dict[str, int] = {}

    def emit(self, event: ProgressEvent) -> None:
        now = self.clock()
        if event.kind == "stage_start":
            self._started[event.stage] = now
            self._completed[event.stage] = 0
            suffix = f" total={event.total}" if event.total is not None else ""
            self._write(f"[{event.stage}] started{suffix}")
            return
        if event.kind == "status" and event.stage == "batch-search":
            started = self._started.setdefault(event.stage, now)
            self._write(
                f"[batch search] estimating workload; elapsed {_duration(now - started)}; "
                f"{event.message}"
            )
            return
        if event.kind == "advance" and event.completed is not None:
            started = self._started.setdefault(event.stage, now)
            elapsed = now - started
            self._completed[event.stage] = event.completed
            facts = _progress_facts(
                event.stage,
                event.completed,
                event.total,
                elapsed,
                event.secondary_completed,
                event.secondary_total,
            )
            self._write(f"[{event.stage} {facts}] {event.message}".rstrip())
            return
        prefix = f"[{event.stage}]"
        self._write(f"{prefix} {event.message}".rstrip())

    def _write(self, value: str) -> None:
        print(value, file=self.stream, flush=True)

    def close(self) -> None:
        pass


class RichProgressReporter:
    def __init__(self, stream: TextIO = sys.stderr) -> None:
        from rich.console import Console
        from rich.progress import BarColumn, Progress, ProgressColumn, TextColumn
        from rich.text import Text

        class ProgressFactsColumn(ProgressColumn):
            def render(self, task) -> Text:
                return Text(
                    _progress_facts(
                        str(task.fields["stage"]),
                        int(task.completed),
                        int(task.total) if task.total is not None else None,
                        task.elapsed or 0.0,
                        task.fields.get("secondary_completed"),
                        task.fields.get("secondary_total"),
                    )
                )

        self.console = Console(file=stream, stderr=True)
        self.progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            ProgressFactsColumn(),
            console=self.console,
        )
        self._tasks: dict[str, int] = {}
        self._started = False

    def emit(self, event: ProgressEvent) -> None:
        if event.kind == "stage_start":
            if not self._started:
                self.progress.start()
                self._started = True
            for task in self._tasks.values():
                self.progress.remove_task(task)
            self._tasks.clear()
            self._tasks[event.stage] = self.progress.add_task(
                event.message or event.stage,
                total=event.total,
                stage=event.stage,
                secondary_completed=event.secondary_completed,
                secondary_total=event.secondary_total,
            )
            return
        if event.kind == "advance" and event.stage in self._tasks:
            description = event.message or event.stage
            self.progress.update(
                self._tasks[event.stage],
                completed=event.completed,
                total=event.total,
                description=description,
                secondary_completed=event.secondary_completed,
                secondary_total=event.secondary_total,
                refresh=True,
            )
            return
        if event.kind == "item_start" and event.stage in self._tasks:
            self.progress.update(self._tasks[event.stage], description=event.message, refresh=True)
            return
        if event.kind == "status" and event.stage == "batch-search":
            task = self._tasks.get(event.stage)
            if task is not None:
                self.progress.update(
                    task,
                    description=f"Batch search · estimating workload · {event.message}",
                    refresh=True,
                )
            else:
                self.console.print(
                    f"[cyan]Batch search[/cyan] · estimating workload · {event.message}"
                )
            return
        self.console.print(f"[bold]{event.stage}[/bold] · {event.message}".rstrip(" ·"))

    def close(self) -> None:
        if self._started:
            self.progress.stop()
            self._started = False


def create_progress_reporter(
    mode: Literal["auto", "plain", "quiet"], *, stream: TextIO = sys.stderr
) -> ProgressReporter:
    if mode == "quiet":
        return NullProgressReporter()
    if mode == "plain" or not stream.isatty():
        return PlainProgressReporter(stream)
    return RichProgressReporter(stream)
