import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lib import security


class SecurityHelpersTest(unittest.TestCase):
    def test_cache_keys_accept_expected_node_ids(self):
        for value in ("42", "node-42", "MiniMaxH3DirectorGroupsCombine"):
            with self.subTest(value=value):
                self.assertEqual(security.safe_cache_key(value), value)

    def test_cache_keys_reject_path_components(self):
        for value in ("", ".", "..", "../input", "node/42", "node\\42", "x" * 129):
            with self.subTest(value=value):
                self.assertIsNone(security.safe_cache_key(value))

    def test_input_files_must_remain_under_input_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            nested = input_dir / "clips"
            nested.mkdir(parents=True)
            expected = nested / "sample.mp4"
            expected.write_bytes(b"video")
            outside = root / "outside.mp4"
            outside.write_bytes(b"outside")
            (input_dir / "escape.mp4").symlink_to(outside)

            with patch.object(security.folder_paths, "get_input_directory", return_value=str(input_dir)):
                self.assertEqual(
                    security.resolve_input_file("sample.mp4", subfolder="clips"),
                    expected.resolve(),
                )
                for value in ("../outside.mp4", str(outside), "escape.mp4"):
                    with self.subTest(value=value):
                        with self.assertRaisesRegex(ValueError, "Invalid input file path"):
                            security.resolve_input_file(value)
                with self.assertRaisesRegex(FileNotFoundError, "Input file not found"):
                    security.resolve_input_file("missing.mp4")

    def test_llm_urls_accept_local_http_and_remote_https(self):
        urls = (
            "http://127.0.0.1:11434/v1",
            "http://[::1]:11434/v1",
            "http://ollama:11434/v1",
            "http://host.docker.internal:11434/v1",
            "http://10.0.0.8:8080/v1",
            "https://open.bigmodel.cn/api/paas/v4",
            "https://api.example.com/v1",
        )
        for url in urls:
            with self.subTest(url=url):
                self.assertEqual(security.validate_llm_url(url), url)

    def test_llm_urls_reject_unsafe_destinations(self):
        urls = (
            "ftp://127.0.0.1/model",
            "http://user:pass@localhost:11434/v1",
            "http://169.254.169.254/latest/meta-data",
            "http://2852039166/latest/meta-data",
            "http://0251.0376.0251.0376/latest/meta-data",
            "http://0xa9.0xfe.0xa9.0xfe/latest/meta-data",
            "http://8.8.8.8/v1",
            "http://api.example.com/v1",
        )
        for url in urls:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    security.validate_llm_url(url)


if __name__ == "__main__":
    unittest.main()
