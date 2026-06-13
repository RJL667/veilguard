"""Pure unit tests for tools/ingest_core.py (ingest architecture Phase 2).

Stdlib-only — exercises classification, the sentence-aware chunker, and the
recursive archive expander + zip-bomb guards with synthetic in-memory zips
and tars. No sub-agents stack, no network.
"""
import io
import os
import sys
import tarfile
import unittest
import zipfile

_TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
sys.path.insert(0, _TOOLS)

import ingest_core as ic  # noqa: E402


def _zip(files: dict) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
        for n, d in files.items():
            z.writestr(n, d)
    return bio.getvalue()


class TestClassify(unittest.TestCase):
    def test_ext_of(self):
        self.assertEqual(ic.ext_of("a/b/c.PDF"), "pdf")
        self.assertEqual(ic.ext_of("x.tar.gz"), "tar.gz")
        self.assertEqual(ic.ext_of("C:\\d\\f.XlsX"), "xlsx")
        self.assertEqual(ic.ext_of("noext"), "")

    def test_kind_of(self):
        self.assertEqual(ic.kind_of("m.pdf"), "pdf")
        self.assertEqual(ic.kind_of("s.xlsx"), "xlsx")
        self.assertEqual(ic.kind_of("w.docx"), "docx")
        self.assertEqual(ic.kind_of("deck.pptx"), "pptx")
        self.assertEqual(ic.kind_of("p.png"), "image")
        self.assertEqual(ic.kind_of("r.md"), "text")
        self.assertEqual(ic.kind_of("a.zip"), "archive")
        # legacy binary Office formats stay unsupported (python-docx/-pptx
        # only read OOXML)
        self.assertEqual(ic.kind_of("old.doc"), "unsupported")
        self.assertEqual(ic.kind_of("old.ppt"), "unsupported")
        self.assertEqual(ic.kind_of("weird.xyz"), "unsupported")

    def test_is_archive(self):
        self.assertTrue(ic.is_archive("x.zip"))
        self.assertTrue(ic.is_archive("x.tgz"))
        self.assertFalse(ic.is_archive("x.pdf"))


class TestChunk(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(ic.chunk_text(""), [])
        self.assertEqual(ic.chunk_text("   \n  "), [])

    def test_single_small(self):
        c = ic.chunk_text("Hello world. This is fine.")
        self.assertEqual(len(c), 1)
        self.assertIn("Hello world", c[0])

    def test_packs_multiple_with_size_bound(self):
        sents = " ".join(
            f"Sentence number {i} has some filler words here." for i in range(200)
        )
        c = ic.chunk_text(sents, target_words=50, overlap_words=10)
        self.assertGreater(len(c), 1)
        for ch in c:
            # each flushed chunk stays near target (sentence granularity)
            self.assertLessEqual(len(ch.split()), 62)

    def test_paragraph_input(self):
        c = ic.chunk_text("Para one sentence.\n\nPara two sentence.",
                          target_words=5, overlap_words=1)
        self.assertGreaterEqual(len(c), 1)
        self.assertTrue(all(ch.strip() for ch in c))


class TestExpand(unittest.TestCase):
    def test_plain_file_passthrough(self):
        entries, skips = ic.expand("notes.txt", b"hello")
        self.assertEqual(entries, [("notes.txt", b"hello")])
        self.assertEqual(skips, [])

    def test_zip_flatten_and_skip_junk(self):
        data = _zip({
            "a.txt": b"alpha",
            "sub/b.md": b"# beta",
            "__MACOSX/._a.txt": b"junk",
            ".DS_Store": b"junk",
        })
        entries, _ = ic.expand("bundle.zip", data)
        names = sorted(p for p, _ in entries)
        self.assertEqual(names, ["bundle.zip/a.txt", "bundle.zip/sub/b.md"])

    def test_nested_zip(self):
        inner = _zip({"deep.txt": b"deep"})
        outer = _zip({"inner.zip": inner, "top.txt": b"top"})
        entries, _ = ic.expand("outer.zip", outer)
        names = sorted(p for p, _ in entries)
        self.assertIn("outer.zip/top.txt", names)
        self.assertIn("outer.zip/inner.zip/deep.txt", names)

    def test_depth_cap(self):
        inner = _zip({"deep.txt": b"deep"})
        outer = _zip({"inner.zip": inner})
        entries, skips = ic.expand("outer.zip", outer, ic.Limits(max_depth=1))
        self.assertNotIn("outer.zip/inner.zip/deep.txt", [p for p, _ in entries])
        self.assertTrue(any("max recursion depth" in r for _, r in skips))

    def test_per_file_size_cap(self):
        data = _zip({"big.txt": b"x" * 2000, "small.txt": b"ok"})
        entries, skips = ic.expand("b.zip", data, ic.Limits(max_file_bytes=1000))
        names = [p for p, _ in entries]
        self.assertIn("b.zip/small.txt", names)
        self.assertNotIn("b.zip/big.txt", names)
        self.assertTrue(any("too large" in r for _, r in skips))

    def test_file_count_cap(self):
        data = _zip({f"f{i}.txt": b"x" for i in range(10)})
        entries, skips = ic.expand("many.zip", data, ic.Limits(max_files=3))
        self.assertEqual(len(entries), 3)
        self.assertTrue(any("file-count cap" in r for _, r in skips))

    def test_zip_bomb_ratio_guard(self):
        bomb = b"\0" * (2 * 1024 * 1024)  # 2 MB of zeros → tiny compressed
        data = _zip({"bomb.bin": bomb})
        entries, skips = ic.expand("z.zip", data, ic.Limits(max_ratio=50))
        self.assertEqual(entries, [])
        self.assertTrue(any("compression ratio" in r for _, r in skips))

    def test_tar_gz(self):
        bio = io.BytesIO()
        with tarfile.open(fileobj=bio, mode="w:gz") as tf:
            b = b"tarred content"
            ti = tarfile.TarInfo("t.txt")
            ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))
        entries, _ = ic.expand("a.tgz", bio.getvalue())
        self.assertEqual([p for p, _ in entries], ["a.tgz/t.txt"])

    def test_unreadable_archive(self):
        entries, skips = ic.expand("broken.zip", b"not a zip at all")
        self.assertEqual(entries, [])
        self.assertTrue(any("unreadable archive" in r for _, r in skips))


if __name__ == "__main__":
    unittest.main(verbosity=2)
