import json
import os
import time

import pandas as pd
import pytest

from ml.ingest import atomic


def frame(value=1):
    return pd.DataFrame({"a": [value]})


def test_parquet_write_creates_parents_and_leaves_no_temp_file(tmp_path):
    target = tmp_path / "deep" / "dir" / "x.parquet"
    atomic.write_parquet_atomic(frame(), target)
    assert pd.read_parquet(target)["a"].tolist() == [1]
    assert [p.name for p in target.parent.iterdir()] == ["x.parquet"]


def test_failed_parquet_write_keeps_the_old_file_and_removes_the_temp(tmp_path, monkeypatch):
    target = tmp_path / "x.parquet"
    atomic.write_parquet_atomic(frame(1), target)

    def boom(self, path, **kwargs):
        with open(path, "wb") as f:
            f.write(b"half a file")  # a partial write, then a crash
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(OSError):
        atomic.write_parquet_atomic(frame(2), target)
    monkeypatch.undo()
    assert pd.read_parquet(target)["a"].tolist() == [1]  # old content intact
    assert [p.name for p in tmp_path.iterdir()] == ["x.parquet"]


def test_json_write_is_atomic_and_cleans_up_on_failure(tmp_path):
    target = tmp_path / "m.json"
    atomic.write_json_atomic({"ok": 1}, target)
    assert json.loads(target.read_text()) == {"ok": 1}
    with pytest.raises(TypeError):
        atomic.write_json_atomic({"bad": object()}, target)  # not serialisable
    assert json.loads(target.read_text()) == {"ok": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["m.json"]


def test_replace_retries_a_transient_permission_error(tmp_path, monkeypatch):
    target = tmp_path / "m.json"
    real, attempts = os.replace, []

    def flaky(src, dst):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError("file in use")
        real(src, dst)

    monkeypatch.setattr(atomic.os, "replace", flaky)
    monkeypatch.setattr(atomic.time, "sleep", lambda s: None)
    atomic.write_json_atomic({"ok": 1}, target)
    assert len(attempts) == 3 and target.exists()


def test_stale_temp_files_are_found_and_swept_but_fresh_ones_are_kept(tmp_path):
    stale, fresh = tmp_path / "a.parquet.tmp", tmp_path / "b.parquet.tmp"
    stale.write_text("x")
    fresh.write_text("x")
    old = time.time() - 2 * atomic.STALE_TMP_SECONDS
    os.utime(stale, (old, old))
    assert atomic.stale_tmp_files(tmp_path) == [stale]
    assert atomic.sweep_stale_tmp(tmp_path) == 1
    assert not stale.exists() and fresh.exists()


def test_a_temp_file_vanishing_during_the_scan_is_ignored(tmp_path, monkeypatch):
    ghost = tmp_path / "a.parquet.1.tmp"
    ghost.write_text("x")
    real_stat = type(ghost).stat

    def vanishing(self, *a, **k):
        if self == ghost:
            raise FileNotFoundError(self)  # a live writer renamed it between listing and stat
        return real_stat(self, *a, **k)

    monkeypatch.setattr(type(ghost), "stat", vanishing)
    assert atomic.stale_tmp_files(tmp_path) == []


def test_permanent_replace_failure_keeps_the_old_file_and_removes_the_temp(tmp_path, monkeypatch):
    target = tmp_path / "m.json"
    atomic.write_json_atomic({"v": 1}, target)

    def locked(src, dst):
        raise PermissionError("file in use")

    monkeypatch.setattr(atomic.os, "replace", locked)
    monkeypatch.setattr(atomic.time, "sleep", lambda s: None)
    with pytest.raises(PermissionError):
        atomic.write_json_atomic({"v": 2}, target)
    monkeypatch.undo()
    assert json.loads(target.read_text()) == {"v": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["m.json"]


def test_temp_names_are_per_process_so_writers_do_not_share_a_temp_file(tmp_path):
    name = atomic._tmp_path(tmp_path / "x.parquet").name
    assert str(os.getpid()) in name and name.endswith(atomic.TMP_SUFFIX)


def test_csv_write_is_atomic_and_round_trips(tmp_path):
    target = tmp_path / "r.csv"
    atomic.write_csv_atomic(frame(5), target, index=False)
    assert pd.read_csv(target)["a"].tolist() == [5]
    assert [p.name for p in tmp_path.iterdir()] == ["r.csv"]
