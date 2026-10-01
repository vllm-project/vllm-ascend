"""Immutable work selected before KV Pool Backend I/O begins."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import BlockHash

from ...coordinates import TokenRange
from .representation import RemoteKVObject
from .transfer import TransferRows, TransferSource

ChunkMask = tuple[bool, ...] | None
RowIndices = range | tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GroupSelection:
    """Logical chunks selected for one original vLLM cache group."""

    group_id: int
    chunk_mask: ChunkMask

    def includes(self, start_token: int, block_size: int) -> bool:
        chunk_index = start_token // block_size
        return self.chunk_mask is None or (chunk_index < len(self.chunk_mask) and self.chunk_mask[chunk_index])


@dataclass(frozen=True, slots=True)
class KVSelection:
    """Content-identified semantic KV selected on the token axis."""

    token_range: TokenRange
    block_hashes: tuple[BlockHash | str, ...]
    groups: tuple[GroupSelection, ...]


@dataclass(frozen=True, slots=True)
class RowSelection:
    """Session facts computed once on a request's stable row axes."""

    rows: TransferRows
    runs_by_coordinate: tuple[tuple[range, ...], ...]

    def __post_init__(self) -> None:
        if len(self.runs_by_coordinate) != len(self.rows.plan.coordinates):
            raise ValueError("Row selection must align the bound coordinate axis")
        for runs in self.runs_by_coordinate:
            previous_stop = 0
            for run in runs:
                if run.step != 1 or run.start < previous_stop or run.stop > self.rows.row_count:
                    raise ValueError("Row selection runs must be ordered, disjoint and in bounds")
                previous_stop = run.stop

    def selected_ranges(self, coordinate_index: int, row_indices: RowIndices) -> tuple[RowIndices, ...]:
        if isinstance(row_indices, range):
            intersections = []
            for run in self.runs_by_coordinate[coordinate_index]:
                start = max(run.start, row_indices.start)
                stop = min(run.stop, row_indices.stop)
                if start < stop:
                    intersections.append(range(start, stop))
            return tuple(intersections)

        selected = set(row_index for run in self.runs_by_coordinate[coordinate_index] for row_index in run)
        return _compress_rows(tuple(row_index for row_index in row_indices if row_index in selected))

    def includes(self, coordinate_index: int, row_index: int) -> bool:
        return any(row_index in run for run in self.runs_by_coordinate[coordinate_index])


@dataclass(frozen=True, slots=True)
class TransferSpan:
    """A compact row-major rectangle of already-bound execution cells."""

    rows: TransferRows
    layout_indices: tuple[int, ...]
    row_indices: RowIndices
    selection: RowSelection | None = None

    def __post_init__(self) -> None:
        if not self.layout_indices or any(
            index < 0 or index >= len(self.rows.plan.layouts) for index in self.layout_indices
        ):
            raise ValueError("Transfer span references an unknown static layout")
        if isinstance(self.row_indices, range):
            valid_rows = (
                self.row_indices.step == 1
                and self.row_indices.start >= 0
                and self.row_indices.stop <= self.rows.row_count
            )
        else:
            valid_rows = all(index >= 0 and index < self.rows.row_count for index in self.row_indices)
        if not valid_rows:
            raise ValueError("Transfer span references an unknown request row")
        if self.selection is not None and self.selection.rows is not self.rows:
            raise ValueError("Transfer span selection belongs to different request rows")

    def selected_ranges(self, layout_index: int) -> tuple[RowIndices, ...]:
        if layout_index not in self.layout_indices:
            raise ValueError("Transfer span does not contain the requested layout")
        if self.selection is None:
            return (self.row_indices,)
        coordinate_index = self.rows.plan.layouts[layout_index].coordinate_index
        return self.selection.selected_ranges(coordinate_index, self.row_indices)

    def includes(self, layout_index: int, row_index: int) -> bool:
        if self.selection is None:
            return True
        coordinate_index = self.rows.plan.layouts[layout_index].coordinate_index
        return self.selection.includes(coordinate_index, row_index)


@dataclass(frozen=True, slots=True)
class TransferWork:
    """Backend-enumerated spans for one bulk or physical-layer fence."""

    physical_layer_id: int | None
    spans: tuple[TransferSpan, ...]

    @property
    def empty(self) -> bool:
        return next(self.iter_cells(), None) is None

    def iter_cells(self) -> Iterator[tuple[TransferSpan, int, int]]:
        """Yield selected cells in the already-fixed Backend order."""

        for span in self.spans:
            for row_index in span.row_indices:
                for layout_index in span.layout_indices:
                    if span.includes(layout_index, row_index):
                        yield span, row_index, layout_index

    @property
    def remote_objects(self) -> tuple[RemoteKVObject, ...]:
        return tuple(
            span.rows.remote_objects_by_coordinate[layout.coordinate_index][row_index]
            for span, row_index, layout_index in self.iter_cells()
            for layout in (span.rows.plan.layouts[layout_index],)
        )

    @property
    def sources(self) -> tuple[TransferSource, ...]:
        return tuple(
            TransferSource(
                span.rows,
                row_index,
                layout.physical_layer_ids,
                span.rows.keys_by_coordinate[layout.coordinate_index][row_index],
            )
            for span, row_index, layout_index in self.iter_cells()
            for layout in (span.rows.plan.layouts[layout_index],)
        )

    def select_keys(self, keys: set[str], claimed_keys: set[str] | None = None) -> TransferWork:
        """Apply session/admission facts without reconstructing plan or row axes."""

        return select_work_keys((self,), keys, claimed_keys)[0]


@dataclass(frozen=True, slots=True)
class LoadTransfer:
    """One request's rows plus pre-enumerated execution fences."""

    request_id: str
    rows: tuple[TransferRows, ...]
    work: tuple[TransferWork, ...]

    def work_for_layer(self, physical_layer_id: int) -> TransferWork | None:
        return next((item for item in self.work if item.physical_layer_id == physical_layer_id), None)


@dataclass(frozen=True, slots=True)
class StoreTransfer:
    """One request's rows plus pre-enumerated execution fences."""

    request_id: str
    rows: tuple[TransferRows, ...]
    work: tuple[TransferWork, ...]
    store_job_id: int | None = None

    def work_for_layer(self, physical_layer_id: int) -> TransferWork | None:
        return next((item for item in self.work if item.physical_layer_id == physical_layer_id), None)


def merge_transfer_work(work: tuple[TransferWork, ...]) -> TransferWork:
    """Batch request-local spans at one execution fence without rebuilding them."""

    if not work:
        raise ValueError("Cannot merge an empty transfer-work batch")
    layer_id = work[0].physical_layer_id
    if any(item.physical_layer_id != layer_id for item in work):
        raise ValueError("Merged transfer work must share one execution fence")
    return TransferWork(layer_id, tuple(span for item in work for span in item.spans))


def select_work_keys(
    work: tuple[TransferWork, ...],
    keys: set[str],
    claimed_keys: set[str] | None = None,
) -> tuple[TransferWork, ...]:
    """Select keys once for a work family and share the result across fences."""

    row_domains, existing_selections = _collect_row_domains(work)

    selections: dict[TransferRows, RowSelection] = {}
    for rows, domains_by_coordinate in row_domains.items():
        runs_by_coordinate = []
        for coordinate_index, domains in enumerate(domains_by_coordinate):
            selected_rows = []
            key_axis = rows.keys_by_coordinate[coordinate_index]
            candidate_runs = _candidate_runs(
                domains,
                existing_selections[rows],
                coordinate_index,
            )
            for row_index in (row for run in candidate_runs for row in run):
                key = key_axis[row_index]
                if key not in keys or (claimed_keys is not None and key in claimed_keys):
                    continue
                selected_rows.append(row_index)
                if claimed_keys is not None:
                    claimed_keys.add(key)
            runs_by_coordinate.append(_compress_rows(tuple(selected_rows)))
        selections[rows] = RowSelection(rows, tuple(runs_by_coordinate))

    return tuple(
        TransferWork(
            item.physical_layer_id,
            tuple(
                TransferSpan(span.rows, span.layout_indices, span.row_indices, selections[span.rows])
                for span in item.spans
            ),
        )
        for item in work
    )


def selected_work_keys(work: tuple[TransferWork, ...]) -> tuple[str, ...]:
    """Return each selected row key once, independent of layer fences."""

    row_domains, existing_selections = _collect_row_domains(work)
    return tuple(
        rows.keys_by_coordinate[coordinate_index][row_index]
        for rows, domains_by_coordinate in row_domains.items()
        for coordinate_index, domains in enumerate(domains_by_coordinate)
        for run in _candidate_runs(domains, existing_selections[rows], coordinate_index)
        for row_index in run
    )


def selected_work_sources(work: tuple[TransferWork, ...]) -> tuple[TransferSource, ...]:
    """Describe selected remote rows once when no layer copy has begun."""

    row_domains, existing_selections = _collect_row_domains(work)
    physical_layers: dict[TransferRows, list[set[int]]] = {
        rows: [set() for _ in rows.plan.coordinates] for rows in row_domains
    }
    for item in work:
        for span in item.spans:
            for layout_index in span.layout_indices:
                layout = span.rows.plan.layouts[layout_index]
                physical_layers[span.rows][layout.coordinate_index].update(layout.physical_layer_ids)
    return tuple(
        TransferSource(
            rows,
            row_index,
            tuple(sorted(physical_layers[rows][coordinate_index])),
            rows.keys_by_coordinate[coordinate_index][row_index],
        )
        for rows, domains_by_coordinate in row_domains.items()
        for coordinate_index, domains in enumerate(domains_by_coordinate)
        for run in _candidate_runs(domains, existing_selections[rows], coordinate_index)
        for row_index in run
    )


def _collect_row_domains(
    work: tuple[TransferWork, ...],
) -> tuple[
    dict[TransferRows, list[set[RowIndices]]],
    dict[TransferRows, RowSelection | None],
]:
    """Collect compact row domains without expanding them once per layer."""

    domains: dict[TransferRows, list[set[RowIndices]]] = {}
    selections: dict[TransferRows, RowSelection | None] = {}
    for item in work:
        for span in item.spans:
            rows = span.rows
            if rows in selections and selections[rows] != span.selection:
                raise ValueError("One work family cannot contain inconsistent row selections")
            selections.setdefault(rows, span.selection)
            domains_by_coordinate = domains.setdefault(
                rows,
                [set() for _ in rows.plan.coordinates],
            )
            for layout_index in span.layout_indices:
                coordinate_index = rows.plan.layouts[layout_index].coordinate_index
                domains_by_coordinate[coordinate_index].add(span.row_indices)
    return domains, selections


def _candidate_runs(
    domains: set[RowIndices],
    selection: RowSelection | None,
    coordinate_index: int,
) -> tuple[range, ...]:
    candidate_runs = _merge_row_domains(domains)
    if selection is None:
        return candidate_runs
    return _intersect_runs(candidate_runs, selection.runs_by_coordinate[coordinate_index])


def _merge_row_domains(domains: set[RowIndices]) -> tuple[range, ...]:
    """Merge compact ranges, expanding only an explicitly enumerated tuple."""

    ranges = [domain for domain in domains if isinstance(domain, range)]
    explicit_rows = {row_index for domain in domains if not isinstance(domain, range) for row_index in domain}
    ranges.extend(_compress_rows(tuple(sorted(explicit_rows))))
    if not ranges:
        return ()

    merged: list[range] = []
    for candidate in sorted(ranges, key=lambda item: (item.start, item.stop)):
        if merged and candidate.start <= merged[-1].stop:
            previous = merged[-1]
            merged[-1] = range(previous.start, max(previous.stop, candidate.stop))
        else:
            merged.append(candidate)
    return tuple(merged)


def _intersect_runs(left: tuple[range, ...], right: tuple[range, ...]) -> tuple[range, ...]:
    intersections = []
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_run = left[left_index]
        right_run = right[right_index]
        start = max(left_run.start, right_run.start)
        stop = min(left_run.stop, right_run.stop)
        if start < stop:
            intersections.append(range(start, stop))
        if left_run.stop <= right_run.stop:
            left_index += 1
        else:
            right_index += 1
    return tuple(intersections)


def _compress_rows(row_indices: tuple[int, ...]) -> tuple[range, ...]:
    if not row_indices:
        return ()
    runs = []
    start = previous = row_indices[0]
    for row_index in row_indices[1:]:
        if row_index != previous + 1:
            runs.append(range(start, previous + 1))
            start = row_index
        previous = row_index
    runs.append(range(start, previous + 1))
    return tuple(runs)
