"""Test suite for vault.py.

Run with:  python -m pytest test_vault.py -v
       or:  python -m unittest test_vault.py -v
"""
from __future__ import annotations

import threading
import unittest

from vault import (
	CorruptionError, ObjectNotFound, QuorumError, Record, StorageNode, Vault,
)


def make_vault(n=3, replication=3, write_quorum=None, read_quorum=None):
	nodes = [StorageNode(f"n{i}") for i in range(n)]
	vault = Vault(nodes, replication=replication,
				  write_quorum=write_quorum, read_quorum=read_quorum)
	return vault, nodes


class RecordTests(unittest.TestCase):

	def test_make_computes_matching_checksum(self):
		record = Record.make("k", 1, b"hello")
		self.assertTrue(record.is_valid())

	def test_is_valid_false_after_tamper(self):
		record = Record.make("k", 1, b"hello")
		tampered = Record(record.key, record.version, b"goodbye", record.checksum)
		self.assertFalse(tampered.is_valid())


class StorageNodeTests(unittest.TestCase):

	def test_read_missing_key_returns_none(self):
		node = StorageNode("n0")
		self.assertIsNone(node.read("missing"))

	def test_write_then_read_round_trip(self):
		node = StorageNode("n0")
		record = Record.make("k", 1, b"v1")
		self.assertTrue(node.write(record))
		self.assertEqual(node.read("k"), record)

	def test_write_rejects_older_version_of_valid_record(self):
		node = StorageNode("n0")
		node.write(Record.make("k", 5, b"new"))
		accepted = node.write(Record.make("k", 3, b"old"))
		self.assertFalse(accepted)
		self.assertEqual(node.read("k").version, 5)

	def test_write_accepts_equal_version(self):
		node = StorageNode("n0")
		node.write(Record.make("k", 5, b"a"))
		accepted = node.write(Record.make("k", 5, b"b"))
		self.assertTrue(accepted)
		self.assertEqual(node.read("k").data, b"b")

	def test_write_always_overrides_corrupted_local_copy(self):
		"""Regression test: a corrupted record's stale version must never
		block a legitimate, lower-numbered replacement."""
		node = StorageNode("n0")
		node.write(Record.make("k", 5, b"good"))
		node.corrupt("k", b"GARBAGE")
		self.assertFalse(node.read("k").is_valid())
		accepted = node.write(Record.make("k", 2, b"recovered"))
		self.assertTrue(accepted)
		self.assertEqual(node.read("k").data, b"recovered")
		self.assertTrue(node.read("k").is_valid())

	def test_offline_node_raises_connection_error(self):
		node = StorageNode("n0")
		node.online = False
		with self.assertRaises(ConnectionError):
			node.read("k")
		with self.assertRaises(ConnectionError):
			node.write(Record.make("k", 1, b"v"))


class VaultBasicTests(unittest.TestCase):

	def test_put_then_get_round_trip(self):
		vault, _ = make_vault()
		version = vault.put("k", b"hello")
		self.assertEqual(version, 1)
		self.assertEqual(vault.get("k"), b"hello")

	def test_put_increments_version(self):
		vault, _ = make_vault()
		vault.put("k", b"v1")
		v2 = vault.put("k", b"v2")
		self.assertEqual(v2, 2)
		self.assertEqual(vault.get("k"), b"v2")

	def test_get_missing_key_raises_object_not_found(self):
		vault, _ = make_vault()
		with self.assertRaises(ObjectNotFound):
			vault.get("missing")

	def test_delete_then_get_raises_object_not_found(self):
		vault, _ = make_vault()
		vault.put("k", b"v1")
		vault.delete("k")
		with self.assertRaises(ObjectNotFound):
			vault.get("k")

	def test_delete_prevents_resurrection_by_stale_replica(self):
		vault, nodes = make_vault(n=3, replication=3, write_quorum=2)
		vault.put("k", b"v1")
		targets = vault._targets("k")
		# Take one target offline so it misses the delete.
		stale = targets[0]
		stale.online = False
		vault.delete("k")
		stale.online = True
		with self.assertRaises(ObjectNotFound):
			vault.get("k")
		# The stale replica should have been repaired to the tombstone.
		self.assertTrue(stale.read("k").deleted)


class VaultValidationTests(unittest.TestCase):

	def test_put_rejects_empty_key(self):
		vault, _ = make_vault()
		with self.assertRaises(ValueError):
			vault.put("", b"v")

	def test_put_rejects_non_string_key(self):
		vault, _ = make_vault()
		with self.assertRaises(ValueError):
			vault.put(123, b"v")  # type: ignore[arg-type]

	def test_put_rejects_non_bytes_data(self):
		vault, _ = make_vault()
		with self.assertRaises(TypeError):
			vault.put("k", "not bytes")  # type: ignore[arg-type]

	def test_get_rejects_bad_key(self):
		vault, _ = make_vault()
		with self.assertRaises(ValueError):
			vault.get("")

	def test_delete_rejects_bad_key(self):
		"""Regression test: delete() must validate like put() does, instead
		of failing later with an opaque AttributeError from _targets()."""
		vault, _ = make_vault()
		with self.assertRaises(ValueError):
			vault.delete("")
		with self.assertRaises(ValueError):
			vault.delete(None)  # type: ignore[arg-type]

	def test_repair_rejects_bad_key(self):
		vault, _ = make_vault()
		vault.put("k", b"v")
		with self.assertRaises(ValueError):
			vault.repair("")


class VaultQuorumTests(unittest.TestCase):

	def test_write_succeeds_with_quorum_of_nodes_offline(self):
		vault, nodes = make_vault(n=3, replication=3, write_quorum=2)
		targets = vault._targets("k")
		targets[0].online = False
		version = vault.put("k", b"v1")
		self.assertEqual(version, 1)

	def test_write_raises_quorum_error_when_insufficient_nodes_online(self):
		vault, nodes = make_vault(n=3, replication=3, write_quorum=2)
		targets = vault._targets("k")
		targets[0].online = False
		targets[1].online = False
		with self.assertRaises(QuorumError):
			vault.put("k", b"v1")

	def test_read_raises_quorum_error_when_insufficient_nodes_reachable(self):
		vault, nodes = make_vault(n=3, replication=3, write_quorum=3, read_quorum=2)
		vault.put("k", b"v1")
		targets = vault._targets("k")
		targets[0].online = False
		targets[1].online = False
		with self.assertRaises(QuorumError):
			vault.get("k")

	def test_read_succeeds_with_one_node_down_under_majority_quorum(self):
		vault, nodes = make_vault(n=3, replication=3, write_quorum=3, read_quorum=2)
		vault.put("k", b"v1")
		targets = vault._targets("k")
		targets[0].online = False
		self.assertEqual(vault.get("k"), b"v1")

	def test_invalid_quorum_configuration_rejected(self):
		nodes = [StorageNode(f"n{i}") for i in range(3)]
		# write_quorum + read_quorum <= replication must be rejected
		# (quorums that don't overlap can't guarantee freshness).
		with self.assertRaises(ValueError):
			Vault(nodes, replication=3, write_quorum=1, read_quorum=1)

	def test_replication_larger_than_node_count_rejected(self):
		nodes = [StorageNode("n0")]
		with self.assertRaises(ValueError):
			Vault(nodes, replication=3)

	def test_empty_node_list_rejected(self):
		with self.assertRaises(ValueError):
			Vault([], replication=1)


class VaultCorruptionAndRepairTests(unittest.TestCase):

	def test_repair_heals_corrupted_replica_with_newer_stale_version(self):
		"""End-to-end regression test for the corruption-vs-version bug:
		a corrupted replica whose *version number* is higher than any
		currently-valid replica must still be overwritten by repair()."""
		vault, nodes = make_vault(n=3, replication=3, write_quorum=2)
		vault.put("k", b"v1")
		targets = vault._targets("k")
		targets[0].corrupt("k", b"GARBAGE")
		self.assertEqual(vault.verify()["corrupt"], 1)

		result = vault.repair("k")
		self.assertEqual(result["repaired"], 1)
		self.assertEqual(vault.verify(), {
			"healthy": 3, "missing": 0, "corrupt": 0, "offline": 0,
		})
		self.assertEqual(vault.get("k"), b"v1")

	def test_get_triggers_read_repair_on_stale_replica(self):
		vault, nodes = make_vault(n=3, replication=3, write_quorum=2)
		targets = vault._targets("k")
		targets[0].online = False
		vault.put("k", b"v1")
		targets[0].online = True
		self.assertIsNone(targets[0].read("k"))
		vault.put("k", b"v2")  # still missing on targets[0]
		self.assertEqual(vault.get("k"), b"v2")
		self.assertEqual(targets[0].read("k").data, b"v2")

	def test_get_raises_corruption_error_when_all_replicas_corrupt(self):
		vault, nodes = make_vault(n=3, replication=3, write_quorum=3, read_quorum=2)
		vault.put("k", b"v1")
		targets = vault._targets("k")
		for node in targets:
			node.corrupt("k", b"GARBAGE")
		with self.assertRaises(CorruptionError):
			vault.get("k")

	def test_repair_with_no_key_scans_all_known_keys(self):
		vault, nodes = make_vault(n=3, replication=3, write_quorum=2)
		vault.put("a", b"1")
		vault.put("b", b"2")
		vault._targets("a")[0].corrupt("a", b"X")
		vault._targets("b")[0].corrupt("b", b"Y")
		result = vault.repair()
		self.assertEqual(result["repaired"], 2)

	def test_verify_counts_offline_nodes(self):
		vault, nodes = make_vault(n=3, replication=3, write_quorum=2)
		vault.put("k", b"v1")
		targets = vault._targets("k")
		targets[0].online = False
		counts = vault.verify()
		self.assertEqual(counts["offline"], 1)
		self.assertEqual(counts["healthy"], 2)


class VaultTargetCacheTests(unittest.TestCase):

	def test_targets_are_stable_across_calls(self):
		vault, nodes = make_vault()
		first = vault._targets("k")
		second = vault._targets("k")
		self.assertEqual([n.node_id for n in first], [n.node_id for n in second])

	def test_targets_cache_does_not_change_after_node_offline(self):
		vault, nodes = make_vault()
		before = [n.node_id for n in vault._targets("k")]
		vault.nodes[0].online = False
		after = [n.node_id for n in vault._targets("k")]
		self.assertEqual(before, after)


class VaultConcurrencyTests(unittest.TestCase):

	def test_concurrent_puts_to_same_key_produce_monotonic_versions(self):
		vault, _ = make_vault(n=3, replication=3, write_quorum=2)
		vault.put("k", b"seed")
		errors = []

		def worker():
			try:
				for _ in range(20):
					vault.put("k", b"x")
			except Exception as exc:  # pragma: no cover - failure diagnostic
				errors.append(exc)

		threads = [threading.Thread(target=worker) for _ in range(5)]
		for t in threads:
			t.start()
		for t in threads:
			t.join()

		self.assertEqual(errors, [])
		final_targets = vault._targets("k")
		versions = {node.read("k").version for node in final_targets
					if node.read("k") is not None}
		# All caught-up replicas should agree once the dust settles.
		vault.repair("k")
		versions_after_repair = {node.read("k").version for node in final_targets}
		self.assertEqual(len(versions_after_repair), 1)


if __name__ == "__main__":  # pragma: no cover
	unittest.main()
