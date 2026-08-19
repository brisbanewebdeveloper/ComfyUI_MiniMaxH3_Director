import unittest
from types import SimpleNamespace

from ComfyUI_MiniMaxH3_Director.director.reference_semantics import (
    collect_reference_semantics,
    compile_reference_prompt,
    reference_semantics_for_segment,
)


class ReferenceSemanticsTest(unittest.TestCase):
    def test_compiles_subject_retention_voice_and_dialogue(self):
        records = collect_reference_semantics(
            [
                {
                    "index": 0,
                    "subjectKind": "person",
                    "description": "the woman in the red coat",
                    "retention": "fully_preserved",
                    "retained": "face, coat, and hairstyle stay unchanged",
                }
            ],
            [],
            [
                {
                    "index": 0,
                    "description": "her calm speaking voice",
                    "retention": "fully_copy",
                    "voiceOf": "Subject 1",
                }
            ],
        )

        prompt = compile_reference_prompt("<Picture 1> says: We leave now.", records)

        self.assertIn("subject_definitions:", prompt)
        self.assertIn("<Subject 1> is the woman in the red coat.", prompt)
        self.assertIn("<Audio 1> is a voice reference for <Subject 1> (S1).", prompt)
        self.assertIn("retention_analysis:", prompt)
        self.assertIn("fully_preserved - face, coat, and hairstyle stay unchanged.", prompt)
        self.assertIn("<Subject 1> (S1): <d>We leave now.</d>", prompt)

    def test_local_metadata_overrides_common_metadata_by_slot(self):
        plan = SimpleNamespace(
            raw={
                "global": {
                    "commonEnabled": True,
                    "refs": [{"index": 0, "description": "common description"}],
                },
                "segments": [
                    {"refs": [{"index": 0, "description": "local description"}]}
                ],
            }
        )
        segment = SimpleNamespace(index=0, ui_index=0, use_global=False, task_key="r2v")

        records = reference_semantics_for_segment(plan, segment)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["description"], "local description")

    def test_plain_references_do_not_change_existing_prompts(self):
        records = collect_reference_semantics([{"index": 0, "imageFile": "person.png"}], [], [])

        self.assertEqual(records, [])
        self.assertEqual(compile_reference_prompt("A quiet close-up.", records), "A quiet close-up.")


if __name__ == "__main__":
    unittest.main()
