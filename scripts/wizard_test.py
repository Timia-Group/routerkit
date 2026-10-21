#!/usr/bin/env python3
"""Unit tests for wizard.py's pure(ish) logic — not named test_*.py on
purpose, so it's never confused with test_route.py (an interactive CLI
tool, not a test suite) or picked up by test-discovery tooling that
assumes every test_*.py is safe to just run. Stdlib only (unittest +
unittest.mock), no pip install needed, no Docker/network/input required.

Run directly:
    python scripts/wizard_test.py
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import wizard


class SourceKind(unittest.TestCase):
    def test_none_stays_none(self):
        self.assertIsNone(wizard._source_kind(None))

    def test_calculate_stays_calculate(self):
        self.assertEqual(wizard._source_kind("calculate"), "calculate")

    def test_file_path_collapses_to_file(self):
        self.assertEqual(wizard._source_kind("data/examples/features.jsonl"), "file")
        self.assertEqual(wizard._source_kind("/abs/path.jsonl"), "file")


class FilledCopyPath(unittest.TestCase):
    def test_adds_local_json_suffix(self):
        original = Path("data/examples/models_exp.json")
        self.assertEqual(wizard._filled_copy_path(original).name, "models_exp.local.json")

    def test_already_local_json_is_unchanged(self):
        already_filled = Path("data/examples/models_exp.local.json")
        self.assertEqual(wizard._filled_copy_path(already_filled), already_filled)


class KeyEnvFor(unittest.TestCase):
    def test_known_provider_without_explicit_override(self):
        self.assertEqual(wizard._key_env_for({"provider": "azure"}), "AZURE_OPENAI_API_KEY")

    def test_explicit_api_key_env_wins(self):
        model = {"provider": "azure", "api_key_env": "AZURE_FOUNDRY_API_KEY"}
        self.assertEqual(wizard._key_env_for(model), "AZURE_FOUNDRY_API_KEY")

    def test_no_auth_provider_needs_no_key(self):
        self.assertIsNone(wizard._key_env_for({"provider": "spheron"}))

    def test_unknown_custom_provider_needs_no_key_check(self):
        self.assertIsNone(wizard._key_env_for({"provider": "my-self-hosted-thing"}))


class MissingRoles(unittest.TestCase):
    def test_role_present(self):
        models = [{"role": "feature"}, {"role": "candidate"}]
        self.assertEqual(wizard._missing_roles(models, frozenset({"feature"})), [])

    def test_role_missing(self):
        models = [{"role": "candidate"}]
        self.assertEqual(wizard._missing_roles(models, frozenset({"judge", "feature"})),
                         ["feature", "judge"])


class LiveEndpointFixes(unittest.TestCase):
    def test_placeholder_in_endpoint_is_fixable(self):
        models = [{"model_id": 16, "name": "gpt-5.4", "role": "feature",
                   "endpoint": "https://<YOUR_AZURE_FOUNDRY_RESOURCE>.openai.azure.com"}]
        fixes = wizard._live_endpoint_fixes(models, frozenset({"feature"}))
        self.assertEqual(fixes, {"<YOUR_AZURE_FOUNDRY_RESOURCE>": [(16, "gpt-5.4")]})

    def test_model_without_model_id_is_not_fixable(self):
        # Came from a not-yet-imported file (models_file, not the
        # snapshot) — there's no row in the database yet to PATCH.
        models = [{"name": "gpt-5.4", "role": "feature",
                   "endpoint": "https://<YOUR_AZURE_FOUNDRY_RESOURCE>.openai.azure.com"}]
        self.assertEqual(wizard._live_endpoint_fixes(models, frozenset({"feature"})), {})

    def test_wrong_role_is_ignored(self):
        models = [{"model_id": 9, "name": "Grok 4.6", "role": "candidate",
                   "endpoint": "https://<SPHERON_GATEWAY_URL>"}]
        self.assertEqual(wizard._live_endpoint_fixes(models, frozenset({"feature"})), {})

    def test_placeholder_outside_endpoint_is_not_fixable(self):
        # By design — PATCH /models/{id} (and this helper) only ever
        # targets the endpoint field, the one place this has actually
        # happened in practice so far.
        models = [{"model_id": 16, "name": "gpt-5.4", "role": "feature",
                   "endpoint": "https://real.example.com", "deployment": "<SOME_PLACEHOLDER>"}]
        self.assertEqual(wizard._live_endpoint_fixes(models, frozenset({"feature"})), {})

    def test_no_placeholder_returns_empty(self):
        models = [{"model_id": 16, "name": "gpt-5.4", "role": "feature", "endpoint": "https://real.example.com"}]
        self.assertEqual(wizard._live_endpoint_fixes(models, frozenset({"feature"})), {})


class FixLiveEndpoints(unittest.TestCase):
    """_fix_live_endpoints shells out to scripts/edit_model.py (so the fix
    is still "a plain command", consistent with every other wizard step) —
    mock subprocess.run rather than hitting a real API."""

    def test_calls_edit_model_with_the_typed_value_for_every_affected_model(self):
        fixes = {"<PLACEHOLDER>": [(16, "gpt-5.4"), (17, "other-feature-model")]}
        with patch.object(wizard, "_input", return_value="https://real.example.com"), \
             patch.object(wizard.subprocess, "run") as mock_run:
            mock_run.return_value = wizard.subprocess.CompletedProcess(args=[], returncode=0)
            wizard._fix_live_endpoints(fixes)
        self.assertEqual(mock_run.call_count, 2)
        called_argvs = [call.args[0] for call in mock_run.call_args_list]
        self.assertTrue(all("16" in argv or "17" in argv for argv in called_argvs))
        self.assertTrue(all("--endpoint=https://real.example.com" in argv for argv in called_argvs))

    def test_empty_input_skips_that_placeholder_entirely(self):
        fixes = {"<PLACEHOLDER>": [(16, "gpt-5.4")]}
        with patch.object(wizard, "_input", return_value=""), \
             patch.object(wizard.subprocess, "run") as mock_run:
            wizard._fix_live_endpoints(fixes)
        mock_run.assert_not_called()


class LiveCallGaps(unittest.TestCase):
    """_live_call_gaps reads .env via _read_env(), which reads
    wizard.ENV_FILE (or .env.example as a fallback) — point both at a
    throwaway temp file instead of touching the real repo's .env."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        tmp_root = Path(self._tmpdir.name)
        (tmp_root / ".env.example").write_text("", encoding="utf-8")
        self._patched_env_file = patch.object(wizard, "ENV_FILE", tmp_root / ".env")
        self._patched_repo_root = patch.object(wizard, "REPO_ROOT", tmp_root)
        self._patched_env_file.start()
        self._patched_repo_root.start()

    def tearDown(self):
        self._patched_env_file.stop()
        self._patched_repo_root.stop()
        self._tmpdir.cleanup()

    def test_placeholder_detected(self):
        models = [{"name": "gpt-5.4", "role": "feature", "provider": "azure",
                   "endpoint": "https://<YOUR_AZURE_FOUNDRY_RESOURCE>.openai.azure.com"}]
        placeholders, keys = wizard._live_call_gaps(models, frozenset({"feature"}))
        self.assertIn("<YOUR_AZURE_FOUNDRY_RESOURCE>", placeholders)
        self.assertEqual(placeholders["<YOUR_AZURE_FOUNDRY_RESOURCE>"], ["gpt-5.4 (feature)"])

    def test_missing_api_key_detected(self):
        models = [{"name": "gpt-5.4", "role": "feature", "provider": "azure", "endpoint": "https://real"}]
        placeholders, keys = wizard._live_call_gaps(models, frozenset({"feature"}))
        self.assertEqual(placeholders, {})
        self.assertEqual(keys, {"AZURE_OPENAI_API_KEY": ["gpt-5.4 (feature)"]})

    def test_fully_ready_model_has_no_gaps(self):
        wizard.ENV_FILE.write_text("AZURE_OPENAI_API_KEY=sk-real-value\n", encoding="utf-8")
        models = [{"name": "gpt-5.4", "role": "feature", "provider": "azure", "endpoint": "https://real"}]
        placeholders, keys = wizard._live_call_gaps(models, frozenset({"feature"}))
        self.assertEqual((placeholders, keys), ({}, {}))

    def test_models_with_other_roles_are_ignored(self):
        models = [{"name": "Grok 4.6", "role": "candidate", "endpoint": "https://<SPHERON_GATEWAY_URL>"}]
        placeholders, keys = wizard._live_call_gaps(models, frozenset({"feature"}))
        self.assertEqual((placeholders, keys), ({}, {}))


class ResumeProgram(unittest.TestCase):
    """_resume_program only prompts when more than one of the command's own
    steps remains — the single-step case must return unchanged without
    ever touching stdin, so it's safe to call directly in a test."""

    def test_single_remaining_step_returns_unchanged_without_asking(self):
        program = ["python", "scripts/run_pipeline.py", "--start-from=train"]
        with patch.object(wizard, "_ask_choice", side_effect=AssertionError("should not prompt")):
            result = wizard._resume_program(program)
        self.assertEqual(result, program)

    def test_multi_step_prompts_and_sets_start_from(self):
        program = ["python", "scripts/run_pipeline.py"]  # full default range: tasks..train
        with patch.object(wizard, "_ask_choice", return_value=2):  # 0=tasks,1=questions,2=models,...
            result = wizard._resume_program(program)
        self.assertIn("--start-from=models", result)

    def test_choosing_the_true_first_step_drops_start_from_entirely(self):
        # No --start-from: own_steps[0] is "tasks", the global first step too
        # (STEP_ORDER[0]) — that's the one case the flag gets dropped, since
        # it's then equivalent to the command's own default.
        program = ["python", "scripts/run_pipeline.py"]
        with patch.object(wizard, "_ask_choice", return_value=0):
            result = wizard._resume_program(program)
        self.assertTrue(all(not a.startswith("--start-from=") for a in result))

    def test_choosing_the_submenus_own_first_step_keeps_it_explicit(self):
        # own_steps[0] here is "solutions" (not the global first step
        # "tasks"), so it stays explicit even though it's "from the
        # beginning" of what *this* command covers.
        program = ["python", "scripts/run_pipeline.py", "--start-from=solutions"]
        with patch.object(wizard, "_ask_choice", return_value=0):
            result = wizard._resume_program(program)
        self.assertIn("--start-from=solutions", result)


class CommandHelpers(unittest.TestCase):
    def test_argv_swaps_python_for_sys_executable(self):
        command = wizard.Command("t", ["python", "scripts/start.py"])
        self.assertEqual(command.argv()[0], wizard.sys.executable)
        self.assertEqual(command.argv()[1:], ["scripts/start.py"])

    def test_argv_leaves_non_python_commands_alone(self):
        command = wizard.Command("t", ["docker", "pull", "some/image"])
        self.assertEqual(command.argv(), ["docker", "pull", "some/image"])

    def test_display_quotes_args_with_spaces(self):
        command = wizard.Command("t", ["python", "scripts/test_route.py", "a prompt with spaces"])
        self.assertIn('"a prompt with spaces"', command.display())

    def test_is_pipeline_and_is_start(self):
        pipeline = wizard.Command("t", ["python", "scripts/run_pipeline.py"])
        start_cmd = wizard.Command("t", ["python", "scripts/start.py"])
        self.assertTrue(pipeline.is_pipeline)
        self.assertFalse(pipeline.is_start)
        self.assertTrue(start_cmd.is_start)
        self.assertFalse(start_cmd.is_pipeline)


class ActiveModelsSnapshot(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._patched = patch.object(
            wizard, "ACTIVE_MODELS_SNAPSHOT", Path(self._tmpdir.name) / "models.json"
        )
        self._patched.start()

    def tearDown(self):
        self._patched.stop()
        self._tmpdir.cleanup()

    def test_missing_snapshot_returns_none(self):
        self.assertIsNone(wizard._load_active_models_snapshot())

    def test_existing_snapshot_is_parsed_as_is(self):
        models = [{"name": "Grok 4.6", "role": "candidate"}]
        wizard.ACTIVE_MODELS_SNAPSHOT.write_text(json.dumps(models), encoding="utf-8")
        self.assertEqual(wizard._load_active_models_snapshot(), models)


if __name__ == "__main__":
    unittest.main()
