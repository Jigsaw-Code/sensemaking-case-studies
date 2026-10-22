# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for phase_3_verify_citations."""

import asyncio
from unittest import mock
import pandas as pd
from panoramic_insights import phase_3_verify_citations


def test_verify_single_insight_deep_research_success():
  async def _test():
    mock_client = mock.MagicMock()

    mock_interaction = mock.MagicMock()
    mock_interaction.id = "interaction_dr_123"
    mock_client.interactions.create.return_value = mock_interaction

    with mock.patch(
        "panoramic_insights.phase_3_verify_citations.poll_interaction",
        return_value="Fetched page content: CARTA bus delays averaged 18 minutes on Broad St.",
    ):
      mock_eval_response = mock.MagicMock()
      mock_eval_response.parsed.verified = True
      mock_eval_response.parsed.reason = "Claim matches fetched webpage content."
      mock_client.models.generate_content.return_value = mock_eval_response

      row = {
          "full_insight": "Bus delays on Broad Street average 18 minutes during peak hours.",
          "citation_text": "Fetch webpage and check average delay metric.",
          "citation_link": "https://gocarta.org/stats",
          "source": "deep_research",
      }

      status, reason = await phase_3_verify_citations.verify_single_insight(
          client=mock_client,
          eval_model="gemini-3-flash-preview",
          file_agent_env_id="env_file_agent",
          fact_checking_env_id="fact_checking_sandbox",
          row=row,
          index=1,
      )

      assert status == phase_3_verify_citations.VerificationStatus.PASS
      assert "matches" in reason
      mock_client.interactions.create.assert_called_once()
      call_kwargs = mock_client.interactions.create.call_args.kwargs
      assert call_kwargs["environment"] == {"type": "remote"}
      assert "fact_checking_sandbox" in call_kwargs["input"]

  asyncio.run(_test())


def test_verify_single_insight_file_agent_success():
  async def _test():
    mock_client = mock.MagicMock()

    mock_interaction = mock.MagicMock()
    mock_interaction.id = "interaction_fa_123"
    mock_client.interactions.create.return_value = mock_interaction

    with mock.patch(
        "panoramic_insights.phase_3_verify_citations.poll_interaction",
        return_value="Script computed total 311 pothole count = 412.",
    ):
      mock_eval_response = mock.MagicMock()
      mock_eval_response.parsed.verified = True
      mock_eval_response.parsed.reason = "Computed pothole count matches raw data."
      mock_client.models.generate_content.return_value = mock_eval_response

      row = {
          "full_insight": "311 received 412 pothole complaints in FY2025.",
          "citation_text": "Run pandas query on requests_311.csv for pothole category.",
          "citation_link": "/data/requests_311.csv",
          "source": "file_agent",
      }

      status, reason = await phase_3_verify_citations.verify_single_insight(
          client=mock_client,
          eval_model="gemini-3-flash-preview",
          file_agent_env_id="env_file_agent",
          fact_checking_env_id="fact_checking_sandbox",
          row=row,
          index=1,
      )

      assert status == phase_3_verify_citations.VerificationStatus.PASS
      assert "matches" in reason
      call_kwargs = mock_client.interactions.create.call_args.kwargs
      assert call_kwargs["environment"] == "env_file_agent"

  asyncio.run(_test())


def test_verify_all_insights_loads_file_agent_metadata(tmp_path):
  """Ensures the file agent metadata mapping loads env_id and sandbox_work_dir."""

  async def _test():
    metadata_csv = tmp_path / "phase_2_file_agent_research.csv"
    meta_df = pd.DataFrame([{
        "topic": "Roads & Bridges",
        "proposal": "Repave Main Street.",
        "env_id": "env_abc",
        "sandbox_work_dir": "/workspace/runs/row_0",
        "research_output": "Found 412 pothole reports.",
    }])
    meta_df.to_csv(metadata_csv, index=False)

    insights_df = pd.DataFrame([{
        "topic": "Roads & Bridges",
        "proposal": "Repave Main Street.",
        "full_insight": "412 pothole reports were filed.",
        "citation_text": "Count rows in requests_311.csv.",
        "citation_link": "/data/requests_311.csv",
        "source": "file_agent",
    }])

    captured = {}

    async def fake_verify_single_insight(*args, **kwargs):
      captured["file_agent_env_id"] = args[2]
      captured["mapped_work_dir"] = kwargs.get("mapped_work_dir")
      return (
          phase_3_verify_citations.VerificationStatus.PASS,
          "Verified.",
      )

    with mock.patch.object(
        phase_3_verify_citations,
        "verify_single_insight",
        side_effect=fake_verify_single_insight,
    ):
      status_list, _ = await phase_3_verify_citations.verify_all_insights(
          client=mock.MagicMock(),
          eval_model="gemini-3-flash-preview",
          file_agent_env_id="env_fallback",
          fact_checking_env_id="fact_checking_sandbox",
          df=insights_df,
          file_agent_metadata_csv=str(metadata_csv),
          output_csv="",
      )

    assert status_list == ["PASS"]
    # Both values come from the metadata CSV, proving the mapping was built.
    assert captured["file_agent_env_id"] == "env_abc"
    assert captured["mapped_work_dir"] == "/workspace/runs/row_0"

  asyncio.run(_test())


def test_verify_single_insight_honours_sandbox_timeout():
  """Ensures the configured sandbox timeout reaches poll_interaction.

  The 600s default silently converted slow sandbox runs into SKIP results,
  which is how a single contended environment produced a mass of SKIPs.
  """

  async def _test():
    mock_client = mock.MagicMock()
    mock_interaction = mock.MagicMock()
    mock_interaction.id = "interaction_timeout_1"
    mock_client.interactions.create.return_value = mock_interaction

    with mock.patch(
        "panoramic_insights.phase_3_verify_citations.poll_interaction",
        return_value="Computed 412 potholes.",
    ) as mock_poll:
      mock_eval_response = mock.MagicMock()
      mock_eval_response.parsed.verified = True
      mock_eval_response.parsed.reason = "Matches."
      mock_client.models.generate_content.return_value = mock_eval_response

      row = {
          "full_insight": "311 received 412 pothole complaints in FY2025.",
          "citation_text": "Count pothole rows in requests_311.csv.",
          "citation_link": "/data/requests_311.csv",
          "source": "file_agent",
      }

      await phase_3_verify_citations.verify_single_insight(
          client=mock_client,
          eval_model="gemini-3-flash-preview",
          file_agent_env_id="env_file_agent",
          fact_checking_env_id="fact_checking_sandbox",
          row=row,
          index=1,
          sandbox_timeout_sec=1800,
      )

    assert mock_poll.call_args.kwargs["timeout_sec"] == 1800

  asyncio.run(_test())


def test_verify_all_insights_caps_runs_per_environment_across_sources():
  """At most 3 sandbox runs share one environment, whatever their source.

  Regression test: the cap used to be the LLM --concurrency (10) and File
  Agent and fact checking runs were counted separately, so all 9 insights of
  a run hit one environment at once and every File Agent check timed out.
  """

  async def _test():
    rows = []
    for i in range(9):
      if i % 3 == 0:
        source = "deep_research"
      else:
        source = "file_agent"
      rows.append({
          "topic": "T",
          "proposal": "P",
          "full_insight": f"insight {i}",
          "citation_text": "trace",
          "citation_link": "/data/x.csv",
          "source": source,
      })
    insights_df = pd.DataFrame(rows)

    in_flight = 0
    peak = 0

    async def fake_verify_single_insight(*args, **kwargs):
      nonlocal in_flight, peak
      del args, kwargs
      in_flight += 1
      peak = max(peak, in_flight)
      await asyncio.sleep(0.01)
      in_flight -= 1
      return phase_3_verify_citations.VerificationStatus.PASS, "ok"

    with mock.patch.object(
        phase_3_verify_citations,
        "verify_single_insight",
        side_effect=fake_verify_single_insight,
    ):
      status_list, _ = await phase_3_verify_citations.verify_all_insights(
          client=mock.MagicMock(),
          eval_model="m",
          file_agent_env_id="shared_env",
          fact_checking_env_id="shared_env",
          df=insights_df,
          file_agent_metadata_csv="",
          output_csv="",
      )

    assert status_list == ["PASS"] * 9
    assert peak == phase_3_verify_citations.DEFAULT_MAX_SANDBOX_CONCURRENCY == 3

  asyncio.run(_test())


def test_verify_insights_with_retries_reverifies_only_skipped_rows(tmp_path):
  """A second pass re-runs SKIP rows only and keeps PASS/FAIL results.

  Row 2 is skipped on its first verification (as a sandbox timeout would be)
  and passes on the second; rows 1 and 3 must not be verified again.
  """

  async def _test():
    insights_df = pd.DataFrame([
        {"topic": "T", "proposal": "P", "full_insight": "one",
         "citation_text": "t", "citation_link": "l", "source": "file_agent"},
        {"topic": "T", "proposal": "P", "full_insight": "two",
         "citation_text": "t", "citation_link": "l", "source": "deep_research"},
        {"topic": "T", "proposal": "P", "full_insight": "three",
         "citation_text": "t", "citation_link": "l", "source": "data_commons"},
    ])
    first_results = {
        1: (phase_3_verify_citations.VerificationStatus.PASS, "ok"),
        2: (phase_3_verify_citations.VerificationStatus.SKIP, "timed out"),
        3: (phase_3_verify_citations.VerificationStatus.FAIL, "mismatch"),
    }
    calls_by_pass = {1: [], 2: []}
    seen_indices = set()

    async def fake_verify_single_insight(*args, **kwargs):
      del kwargs
      index = args[5]
      if index in seen_indices:
        calls_by_pass[2].append(index)
        return phase_3_verify_citations.VerificationStatus.PASS, "ok on retry"
      seen_indices.add(index)
      calls_by_pass[1].append(index)
      return first_results[index]

    output_csv = tmp_path / "verified.csv"
    with mock.patch.object(
        phase_3_verify_citations,
        "verify_single_insight",
        side_effect=fake_verify_single_insight,
    ):
      status_list, reason_list = (
          await phase_3_verify_citations.verify_insights_with_retries(
              max_passes=2,
              client=mock.MagicMock(),
              eval_model="m",
              file_agent_env_id="env_fa",
              fact_checking_env_id="env_fc",
              df=insights_df,
              file_agent_metadata_csv="",
              output_csv=str(output_csv),
          )
      )

    assert status_list == ["PASS", "PASS", "FAIL"]
    assert reason_list == ["ok", "ok on retry", "mismatch"]
    assert sorted(calls_by_pass[1]) == [1, 2, 3]
    assert calls_by_pass[2] == [2]
    written = pd.read_csv(output_csv)
    assert list(written["verification_status"]) == ["PASS", "PASS", "FAIL"]

  asyncio.run(_test())


def test_verify_insights_with_retries_stops_after_max_passes():
  """Rows still SKIP after the last pass stay SKIP; no extra pass runs."""

  async def _test():
    insights_df = pd.DataFrame([
        {"topic": "T", "proposal": "P", "full_insight": "one",
         "citation_text": "t", "citation_link": "l", "source": "file_agent"},
    ])
    call_count = 0

    async def always_skip(*args, **kwargs):
      nonlocal call_count
      del args, kwargs
      call_count += 1
      return phase_3_verify_citations.VerificationStatus.SKIP, "down"

    with mock.patch.object(
        phase_3_verify_citations,
        "verify_single_insight",
        side_effect=always_skip,
    ):
      status_list, _ = (
          await phase_3_verify_citations.verify_insights_with_retries(
              max_passes=3,
              client=mock.MagicMock(),
              eval_model="m",
              file_agent_env_id="env_fa",
              fact_checking_env_id="env_fc",
              df=insights_df,
              file_agent_metadata_csv="",
              output_csv="",
          )
      )

    assert status_list == ["SKIP"]
    assert call_count == 3

  asyncio.run(_test())


def test_verify_insights_with_retries_skips_second_pass_without_skips():
  """No further pass runs when the first pass leaves nothing to retry."""

  async def _test():
    insights_df = pd.DataFrame([
        {"topic": "T", "proposal": "P", "full_insight": "one",
         "citation_text": "t", "citation_link": "l", "source": "file_agent"},
    ])
    call_count = 0

    async def always_pass(*args, **kwargs):
      nonlocal call_count
      del args, kwargs
      call_count += 1
      return phase_3_verify_citations.VerificationStatus.PASS, "ok"

    with mock.patch.object(
        phase_3_verify_citations,
        "verify_single_insight",
        side_effect=always_pass,
    ):
      status_list, _ = (
          await phase_3_verify_citations.verify_insights_with_retries(
              max_passes=2,
              client=mock.MagicMock(),
              eval_model="m",
              file_agent_env_id="env_fa",
              fact_checking_env_id="env_fc",
              df=insights_df,
              file_agent_metadata_csv="",
              output_csv="",
          )
      )

    assert status_list == ["PASS"]
    assert call_count == 1

  asyncio.run(_test())


def test_max_passes_flag_defaults_to_two():
  parser = phase_3_verify_citations.build_arg_parser()
  args = parser.parse_args(["--input_csv", "in.csv"])
  assert args.max_passes == 2
  assert args.sandbox_timeout_sec == 600
  args = parser.parse_args(["--input_csv", "in.csv", "--max_passes", "4"])
  assert args.max_passes == 4
