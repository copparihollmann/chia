"""Content-addressed cache tags — pure, no Ray cluster needed.

The bug these exist to prevent: a cached Verilator or synthesis result served after its input RTL
changed, because ``_chia_tag`` was a hand-written string that did not mention the RTL. The test that
matters most is :meth:`TestInvalidation.test_editing_one_byte_changes_the_tag`.

Run:
  pytest chia/base/test/test_content_tag.py
"""

from __future__ import annotations

import pytest

from chia.base.content_tag import (
    DEFAULT_EXCLUDE,
    MissingCacheInput,
    content_tag,
    explain,
    sha256_dir,
    sha256_file,
)


@pytest.fixture
def tree(tmp_path):
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "top.sv").write_text("module top; endmodule\n")
    (src / "sub" / "leaf.sv").write_text("module leaf; endmodule\n")
    return src


class TestDeterminism:
    def test_the_same_inputs_give_the_same_tag(self, tree):
        a = content_tag("synth", dirs=[tree], params={"clock_ns": 1.0})
        b = content_tag("synth", dirs=[tree], params={"clock_ns": 1.0})
        assert a == b

    def test_argument_order_does_not_matter(self, tmp_path):
        x, y = tmp_path / "x.sv", tmp_path / "y.sv"
        x.write_text("x")
        y.write_text("y")
        assert content_tag("t", files=[x, y]) == content_tag("t", files=[y, x]), (
            "two call sites naming the same inputs describe the same work"
        )

    def test_param_key_order_does_not_matter(self, tree):
        a = content_tag("t", dirs=[tree], params={"a": 1, "b": 2})
        b = content_tag("t", dirs=[tree], params={"b": 2, "a": 1})
        assert a == b

    def test_the_readable_prefix_survives(self, tree):
        tag = content_tag("synth_mxu0", dirs=[tree])
        assert tag.startswith("synth_mxu0@"), "_tag_filename slugs this into the cache filename"
        assert len(tag.split("@")[1]) == 16


class TestInvalidation:
    def test_editing_one_byte_changes_the_tag(self, tree):
        """The whole point. With a hand-written tag this edit is invisible and the cache serves the
        previous design's result under the new design's name."""
        before = content_tag("synth", dirs=[tree])
        (tree / "top.sv").write_text("module top; wire w; endmodule\n")
        assert content_tag("synth", dirs=[tree]) != before

    def test_renaming_a_file_changes_the_tag(self, tree):
        """Same bytes, different name. A build that reads by name is not the same build."""
        before = content_tag("synth", dirs=[tree])
        (tree / "top.sv").rename(tree / "renamed.sv")
        assert content_tag("synth", dirs=[tree]) != before

    def test_adding_a_file_changes_the_tag(self, tree):
        before = content_tag("synth", dirs=[tree])
        (tree / "extra.sv").write_text("module extra; endmodule\n")
        assert content_tag("synth", dirs=[tree]) != before

    def test_a_tool_upgrade_changes_the_tag(self, tree):
        """A yosys upgrade must invalidate a synthesis result. Without `tools` it does not, and the
        cache quietly mixes results from two toolchains into one table."""
        a = content_tag("synth", dirs=[tree], tools={"yosys": "0.38"})
        b = content_tag("synth", dirs=[tree], tools={"yosys": "0.39"})
        assert a != b

    def test_a_param_change_changes_the_tag(self, tree):
        a = content_tag("synth", dirs=[tree], params={"clock_ns": 1.0})
        b = content_tag("synth", dirs=[tree], params={"clock_ns": 0.9})
        assert a != b

    def test_the_name_participates(self, tree):
        assert content_tag("synth", dirs=[tree]) != content_tag("sim", dirs=[tree])

    def test_a_file_and_a_dir_with_the_same_digest_do_not_collide(self, tmp_path):
        """`files=` and `dirs=` are domain-separated, so a path moving between them is a change."""
        f = tmp_path / "only.sv"
        f.write_text("x")
        d = tmp_path / "d"
        d.mkdir()
        (d / "only.sv").write_text("x")
        assert content_tag("t", files=[f]) != content_tag("t", dirs=[d])


class TestMissingInputs:
    def test_a_missing_file_raises(self, tmp_path):
        """Skipping it would hash the same whether or not the input was there — the very failure
        this module exists to prevent, one level up."""
        with pytest.raises(MissingCacheInput):
            content_tag("t", files=[tmp_path / "nope.sv"])

    def test_a_missing_dir_raises(self, tmp_path):
        with pytest.raises(MissingCacheInput):
            content_tag("t", dirs=[tmp_path / "nope"])

    def test_a_directory_passed_as_a_file_raises(self, tree):
        with pytest.raises(MissingCacheInput):
            sha256_file(tree)

    def test_a_dangling_symlink_is_recorded_not_skipped(self, tree):
        """A tree with a broken link must not hash the same as one without the link at all."""
        before = sha256_dir(tree)
        (tree / "broken").symlink_to(tree / "does_not_exist")
        assert sha256_dir(tree) != before


class TestExclusions:
    def test_vcs_and_build_scratch_are_ignored(self, tree):
        """These churn without changing what a build reads; including them defeats the cache."""
        before = sha256_dir(tree)
        (tree / ".git").mkdir()
        (tree / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        (tree / "__pycache__").mkdir()
        (tree / "__pycache__" / "x.pyc").write_bytes(b"\x00\x01")
        assert sha256_dir(tree) == before

    def test_exclusions_are_overridable(self, tree):
        (tree / ".git").mkdir()
        (tree / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        assert sha256_dir(tree, exclude=()) != sha256_dir(tree, exclude=DEFAULT_EXCLUDE)


class TestExplain:
    def test_explain_reports_each_input_digest(self, tree, tmp_path):
        f = tmp_path / "constraints.sdc"
        f.write_text("create_clock -period 1.0\n")
        out = explain("synth", files=[f], dirs=[tree], params={"flatten": "none"},
                      tools={"yosys": "0.38"})
        assert out["tag"] == content_tag("synth", files=[f], dirs=[tree],
                                         params={"flatten": "none"}, tools={"yosys": "0.38"})
        assert out["files"][str(f)] == sha256_file(f)
        assert out["dirs"][str(tree)] == sha256_dir(tree)
        assert "flatten" in out["params"] and "yosys" in out["tools"]

    def test_explain_answers_why_did_this_rerun(self, tree):
        """A cache whose misses are unexplained gets disabled."""
        a = explain("synth", dirs=[tree])
        (tree / "top.sv").write_text("changed\n")
        b = explain("synth", dirs=[tree])
        assert a["tag"] != b["tag"]
        assert a["dirs"][str(tree)] != b["dirs"][str(tree)], "the differing input is named"


class TestCacheIntegration:
    def test_the_cache_filename_stays_greppable(self, tree):
        """`_tag_filename` slugs `@` to `_`, which is fine — what matters is that both halves of the
        tag survive into the filename, so a cache directory can still be read by a human."""
        from chia.base.cache import _tag_filename

        tag = content_tag("synth_mxu0", dirs=[tree])
        name, digest = tag.split("@")
        fname = _tag_filename(tag)
        assert fname.startswith(f"{name}_"), "the readable prefix survives slugging"
        assert digest in fname, "so does the content digest"

    def test_the_slug_truncation_cannot_hide_the_digest(self, tree):
        """`_tag_filename` truncates the slug at 80 chars. A long enough name would push the content
        digest out of the readable half — the filename would still be unique (it appends its own
        hash of the whole tag) but would no longer show WHICH content it addressed."""
        from chia.base.cache import _tag_filename

        long_name = "synth_" + "x" * 90
        tag = content_tag(long_name, dirs=[tree])
        digest = tag.split("@")[1]
        assert digest not in _tag_filename(tag), (
            "documents the limit: keep `name` short enough that name+1+16 <= 80"
        )

    def test_distinct_content_gives_distinct_cache_filenames(self, tree):
        from chia.base.cache import _tag_filename

        a = _tag_filename(content_tag("synth", dirs=[tree]))
        (tree / "top.sv").write_text("different\n")
        b = _tag_filename(content_tag("synth", dirs=[tree]))
        assert a != b
