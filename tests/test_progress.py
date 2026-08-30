import sys
import types
import unittest
from unittest.mock import patch

from ComfyUI_MiniMaxH3_Director.director.progress import report_director_progress


class _PromptServer:
    instance = None


class ProgressPayloadTest(unittest.TestCase):
    def test_payload_keeps_progress_presentation_language_neutral(self):
        server = _PromptServer()
        server.client_id = "test-client"
        sent = []
        server.send_sync = lambda event, payload, client: sent.append((event, payload, client))
        _PromptServer.instance = server
        server_module = types.SimpleNamespace(PromptServer=_PromptServer)

        with patch.dict(sys.modules, {"server": server_module}):
            report_director_progress(
                "node-1",
                segment_index=0,
                segment_total=2,
                phase="sample",
                frame_start=0,
                frame_end=362,
                frame_count=362,
            )

        payload = sent[0][1]
        self.assertEqual(payload["phase"], "sample")
        self.assertEqual(payload["frame_start"], 0)
        self.assertEqual(payload["frame_end"], 362)
        self.assertEqual(payload["frame_count"], 362)
        self.assertNotIn("phase_label", payload)
        self.assertNotIn("frames_label", payload)


if __name__ == "__main__":
    unittest.main()
