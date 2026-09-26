"""Thread-safe quorum-replicated object storage with integrity repair.

StorageNode is an in-memory reference backend. Production deployments can
implement its read/write interface using durable storage and transport RPCs.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

__all__ = [
	"VaultError", "QuorumError", "ObjectNotFound", "CorruptionError",
	"Record", "StorageNode", "Vault",
]

logger = logging.getLogger(__name__)

# Sentinel distinguishing "node was unreachable" from "node had no record"
# (the latter is a legitimate `None` read). Kept private to this module.
_UNREACHABLE = object()


class VaultError(Exception):
	"""Base class for all errors raised by this module."""
	pass


class QuorumError(VaultError):
	"""Raised when an operation could not reach enough replicas to proceed."""
	pass


class ObjectNotFound(VaultError):
	"""Raised when a key has no live (non-tombstoned) value."""
	pass


class CorruptionError(VaultError):
	"""Raised when quorum was reached but every replica failed its checksum.

	Distinct from ``ObjectNotFound`` so callers and monitoring can tell
	"never existed / already deleted" apart from "data is on disk but
	unreadable" -- the latter usually needs a `Vault.repair()` run against
	the surviving nodes, or manual recovery, rather than being treated as
	a normal miss.
	"""
	pass


@dataclass(frozen=True)
class Record:
	"""An immutable, checksum-verified value for a single key/version."""

	key: str
	version: int
	data: bytes
	checksum: str
	deleted: bool = False

	@classmethod
	def make(cls, key: str, version: int, data: bytes,
			 deleted: bool = False) -> "Record":
		"""Build a Record with its checksum computed from ``data``."""
		return cls(key, version, data, hashlib.sha256(data).hexdigest(), deleted)

	def is_valid(self) -> bool:
		"""Return whether ``data`` still matches the stored checksum."""
		return hashlib.sha256(self.data).hexdigest() == self.checksum


class StorageNode:
	"""Independently available storage endpoint.

	Each node is a single logical replica. ``online`` can be flipped to
	simulate a network partition or outage without discarding the node's
	data, and ``corrupt`` can be used to simulate on-disk bit rot for
	testing repair logic.
	"""

	def __init__(self, node_id: str):
		self.node_id = node_id
		self.online = True
		self._records: Dict[str, Record] = {}
		self._lock = threading.RLock()

	def read(self, key: str) -> Optional[Record]:
		"""Return the stored record for ``key``, or None if absent.

		Raises:
			ConnectionError: if this node is currently offline.
		"""
		if not self.online:
			raise ConnectionError(self.node_id)
		with self._lock:
			return self._records.get(key)

	def write(self, record: Record) -> bool:
		"""Accept ``record`` unless a *valid* local copy is already newer.

		A corrupted local record must never block an incoming write: its
		version field cannot be trusted once its checksum fails, so it is
		always eligible for replacement regardless of version ordering.

		Returns:
			True if the record was stored, False if rejected as stale.

		Raises:
			ConnectionError: if this node is currently offline.
		"""
		if not self.online:
			raise ConnectionError(self.node_id)
		with self._lock:
			old = self._records.get(record.key)
			if old is None or not old.is_valid() or record.version >= old.version:
				self._records[record.key] = record
				return True
			return False

	def corrupt(self, key: str, data: bytes) -> None:
		"""Diagnostic hook to simulate disk corruption.

		Overwrites the stored bytes for ``key`` while keeping the old
		version and checksum, so ``Record.is_valid()`` will fail on it.
		"""
		with self._lock:
			old = self._records[key]
			self._records[key] = Record(old.key, old.version, data,
										old.checksum, old.deleted)

	def __repr__(self) -> str:  # pragma: no cover - debugging aid only
		state = "online" if self.online else "offline"
		return f"StorageNode({self.node_id!r}, {state}, {len(self._records)} keys)"


class Vault:
	"""Replicated object store with quorum writes, reads, and read repair."""

	def __init__(self, nodes: Iterable[StorageNode], replication: int = 3,
				 write_quorum: Optional[int] = None,
				 read_quorum: Optional[int] = None):
		self.nodes = list(nodes)
		if not self.nodes or not 1 <= replication <= len(self.nodes):
			raise ValueError("replication must be within the node count")
		self.replication = replication
		self.write_quorum = replication if write_quorum is None else write_quorum
		self.read_quorum = replication // 2 + 1 if read_quorum is None else read_quorum
		if (not 1 <= self.write_quorum <= replication or
				not 1 <= self.read_quorum <= replication or
				self.write_quorum + self.read_quorum <= replication):
			raise ValueError("quorums must be valid and intersect")
		self._key_locks: Dict[str, threading.RLock] = {}
		self._locks_guard = threading.Lock()
		# Placement is a pure function of (key, node list), and the node
		# list is fixed after construction, so it is safe -- and, for hot
		# keys, considerably cheaper -- to memoize it instead of hashing
		# and slicing on every put/get/delete/repair call.
		self._target_cache: Dict[str, List[StorageNode]] = {}
		self._target_cache_guard = threading.Lock()

	def _lock(self, key: str) -> threading.RLock:
		with self._locks_guard:
			return self._key_locks.setdefault(key, threading.RLock())

	def _targets(self, key: str) -> List[StorageNode]:
		"""Deterministic placement; rendezvous-style rotation is stable per key."""
		cached = self._target_cache.get(key)
		if cached is not None:
			return cached
		start = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
		offset = start % len(self.nodes)
		computed = (self.nodes[offset:] + self.nodes[:offset])[:self.replication]
		with self._target_cache_guard:
			return self._target_cache.setdefault(key, computed)

	@staticmethod
	def _next_version(nodes: Iterable[StorageNode], key: str) -> int:
		versions = []
		for node in nodes:
			try:
				record = node.read(key)
				if record is not None and record.is_valid():
					versions.append(record.version)
			except ConnectionError:
				pass
		return max(versions, default=0) + 1

	@staticmethod
	def _validate_key(key: str) -> None:
		if not isinstance(key, str) or not key:
			raise ValueError("key must be a non-empty string")

	def _commit(self, key: str, data: bytes, deleted: bool = False) -> int:
		targets = self._targets(key)
		record = Record.make(key, self._next_version(targets, key), data, deleted)
		successes = 0
		for node in targets:
			try:
				if node.write(record):
					successes += 1
			except ConnectionError:
				continue
		if successes < self.write_quorum:
			raise QuorumError(f"write reached {successes}/{self.write_quorum} nodes")
		return record.version

	def put(self, key: str, data: bytes) -> int:
		"""Write ``data`` as a new version of ``key``.

		Returns:
			The new version number.

		Raises:
			ValueError: if ``key`` is not a non-empty string.
			TypeError: if ``data`` is not bytes.
			QuorumError: if fewer than ``write_quorum`` replicas accepted it.
		"""
		self._validate_key(key)
		if not isinstance(data, bytes):
			raise TypeError("data must be bytes")
		with self._lock(key):
			return self._commit(key, data)

	def get(self, key: str) -> bytes:
		"""Read the latest live value for ``key``, repairing stale replicas.

		A single round of reads is used both to decide the answer and to
		know which replicas need a repair write, instead of reading each
		target twice.

		Raises:
			ValueError: if ``key`` is not a non-empty string.
			QuorumError: if fewer than ``read_quorum`` replicas responded.
			CorruptionError: quorum responded, but every record present
				failed its checksum (data exists but is unreadable).
			ObjectNotFound: no live record exists (never written, or the
				latest version is a tombstone from ``delete``).
		"""
		self._validate_key(key)
		with self._lock(key):
			targets = self._targets(key)
			snapshot: Dict[str, object] = {}
			responses = 0
			valid_records: List[Record] = []
			corrupt_seen = False
			for node in targets:
				try:
					record = node.read(key)
				except ConnectionError:
					snapshot[node.node_id] = _UNREACHABLE
					continue
				snapshot[node.node_id] = record
				responses += 1
				if record is not None:
					if record.is_valid():
						valid_records.append(record)
					else:
						corrupt_seen = True
			if responses < self.read_quorum:
				raise QuorumError(f"read reached {responses}/{self.read_quorum} nodes")
			if not valid_records:
				if corrupt_seen:
					raise CorruptionError(
						f"all quorum-reachable replicas of {key!r} failed checksum")
				raise ObjectNotFound(key)
			newest = max(valid_records, key=lambda item: item.version)
			for node in targets:
				current = snapshot.get(node.node_id)
				if current is _UNREACHABLE:
					continue
				if (current is None or not current.is_valid() or
						current.version < newest.version):
					try:
						node.write(newest)
					except ConnectionError:
						pass
			if newest.deleted:
				raise ObjectNotFound(key)
			return newest.data

	def delete(self, key: str) -> int:
		"""Replicate a tombstone so stale replicas cannot resurrect deleted data.

		Raises:
			ValueError: if ``key`` is not a non-empty string.
			QuorumError: if fewer than ``write_quorum`` replicas accepted it.
		"""
		self._validate_key(key)
		with self._lock(key):
			return self._commit(key, b"", deleted=True)

	def repair(self, key: Optional[str] = None) -> dict:
		"""Reconcile replicas for one object, or all objects known to local nodes.

		Only keys within a node's target set for that key are reconciled;
		a replica stored on a node outside its rendezvous set is orphaned
		and intentionally left untouched (it is not part of this object's
		quorum group). As in ``get``, each target is read once per pass.

		Returns:
			A dict with ``checked`` (replica slots examined) and
			``repaired`` (replica slots actually rewritten) counts.
		"""
		if key is not None:
			self._validate_key(key)
			keys = {key}
		else:
			keys = set()
			for node in self.nodes:
				if not node.online:
					continue
				with node._lock:
					keys.update(node._records)
		checked = repaired = 0
		for object_key in keys:
			with self._lock(object_key):
				targets = self._targets(object_key)
				snapshot: Dict[str, object] = {}
				valid = []
				for node in targets:
					try:
						record = node.read(object_key)
					except ConnectionError:
						snapshot[node.node_id] = _UNREACHABLE
						continue
					snapshot[node.node_id] = record
					if record is not None and record.is_valid():
						valid.append(record)
				if not valid:
					continue
				newest = max(valid, key=lambda item: item.version)
				for node in targets:
					checked += 1
					current = snapshot.get(node.node_id)
					if current is _UNREACHABLE:
						continue
					if (current is None or not current.is_valid() or
							current.version < newest.version):
						try:
							if node.write(newest):
								repaired += 1
								logger.info(
									"repair: restored %r v%d on node %s",
									object_key, newest.version, node.node_id)
						except ConnectionError:
							pass
		return {"checked": checked, "repaired": repaired}

	def verify(self) -> dict:
		"""Count healthy, absent, corrupt, and offline replica slots."""
		counts = {"healthy": 0, "missing": 0, "corrupt": 0, "offline": 0}
		keys = set()
		for node in self.nodes:
			if not node.online:
				continue
			with node._lock:
				keys.update(node._records)
		for key in keys:
			for node in self._targets(key):
				try:
					record = node.read(key)
					counts["missing" if record is None else
						   "healthy" if record.is_valid() else "corrupt"] += 1
				except ConnectionError:
					counts["offline"] += 1
		return counts
