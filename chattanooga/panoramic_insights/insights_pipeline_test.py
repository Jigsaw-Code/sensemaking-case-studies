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

"""Integration test for the hybrid civic insights generation pipeline.

This script executes the entire 3-phase pipeline end-to-end inside the local
sandbox.
To ensure the test is fast, cheap, and repeatable, it uses:
1. **Flash Model**: Runs all steps using `gemini-3.8-flash` instead of
Pro.
2. **Mock Ingress**: Generates a small, 2-row dataset of realistic resident
opinions
   (one regarding bus delays, one regarding streetlight safety).
3. **Hybrid Path Mocking**: Instead of running a real, heavy Deep Research task
(which
   takes minutes/hours and has high cost), the test executes the live **Data
   Commons**
   pathway, generates a **mock qualitative Deep Research dossier** in-memory,
   and feeds
   BOTH to Phase 3 to verify the parallel aggregation and structured synthesis
   logic.
"""

import os
import shutil
import subprocess
import sys
import pandas as pd

# Global variables to be initialized in main
GEMINI_API_KEY = None
DC_API_KEY = None

# Target Geographic Area for Data Commons queries (Chattanooga, TN)
PLACE_NAME = "Chattanooga, Tennessee"
PLACE_DCID = "geoId/4714000"

# Temporary directory where all intermediate pipeline CSVs will be written
TEMP_DIR = "panoramic_insights/test_temp"


def setup_temp_dir():
  """Wipes and recreates the temporary test directory to ensure test isolation."""
  if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)
  os.makedirs(TEMP_DIR, exist_ok=True)


def create_mock_input():
  """Generates a mock input CSV representing the output of the prior categorization step.

  Contains two distinct resident opinions requiring different types of research:
  1. "Buses are unreliable and late" (requires transit and commute data).
  2. "Need more streetlights for safety at night" (requires crime and
  infrastructure data).
  """
  data = [
      {
          "participant_id": "p1",
          "survey_text": (
              "The buses in Chattanooga are always late. I waited 40 minutes on"
              " Broad St yesterday."
          ),
          "quote": (
              "The buses in Chattanooga are always late. I waited 40 minutes on"
              " Broad St yesterday."
          ),
          "topic": "Public Transportation",
          "proposal": "Buses are unreliable and late",
      },
      {
          "participant_id": "p2",
          "survey_text": (
              "We need more streetlights in the Alton Park neighborhood. It's"
              " pitch black at night and feels unsafe."
          ),
          "quote": (
              "We need more streetlights in the Alton Park neighborhood. It's"
              " pitch black at night."
          ),
          "topic": "Public Safety",
          "proposal": "Need more streetlights for safety at night",
      },
  ]
  df = pd.DataFrame(data)
  input_path = os.path.join(TEMP_DIR, "mock_categorized.csv")
  df.to_csv(input_path, index=False)
  print(f"[Test Setup] Created mock input CSV at {input_path}")
  return input_path


def run_step(command_args):
  """Helper to run a pipeline phase as a subprocess, printing outputs on failure."""
  print(f"\n[Running Step] {' '.join(command_args)}")
  result = subprocess.run(command_args, capture_output=True, text=True)
  if result.returncode != 0:
    print(f"[Step Failed] Exit code {result.returncode}")
    print(f"Stdout:\n{result.stdout}")
    print(f"Stderr:\n{result.stderr}")
    raise RuntimeError(f"Step failed: {command_args[1]}")
  print("[Step Success]")
  return result.stdout


def main():
  global GEMINI_API_KEY, DC_API_KEY
  GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
  DC_API_KEY = os.environ.get("DC_API_KEY")

  if not GEMINI_API_KEY or not DC_API_KEY:
    raise ValueError(
        "Both GEMINI_API_KEY and DC_API_KEY environment variables must be set "
        "to run this integration test."
    )

  setup_temp_dir()
  input_csv = create_mock_input()

  # Define paths for intermediate output artifacts
  phase_1_output = os.path.join(TEMP_DIR, "phase_1_prompts.csv")
  phase_2_dc_output = os.path.join(
      TEMP_DIR, "phase_2_data_commons_research.csv"
  )
  phase_2_deep_mock_output = os.path.join(
      TEMP_DIR, "phase_2_deep_research_mock.csv"
  )
  phase_3_output = os.path.join(TEMP_DIR, "phase_3_structured_insights.csv")

  # --------------------------------------------------------------------------
  # 1. Run Phase 1: Research Prompt Generation
  # --------------------------------------------------------------------------
  run_step([
      sys.executable,
      "-m",
      "panoramic_insights.phase_1_generate_research_ideas_prompts",
      "--input_csv",
      input_csv,
      "--output_csv",
      phase_1_output,
      "--place_name",
      PLACE_NAME,
      "--gemini_api_key",
      GEMINI_API_KEY,
      "--model_name",
      "gemini-3.8-flash",
  ])

  # Verify Phase 1 schemas
  df1 = pd.read_csv(phase_1_output)
  print(f"[Verify Phase 1] Columns: {list(df1.columns)}")
  assert "research_ideas_prompt" in df1.columns
  print(
      "[Verify Phase 1] Sample"
      f" prompt:\n{df1['research_ideas_prompt'].iloc[0][:200]}..."
  )

  # --------------------------------------------------------------------------
  # 2. Run Phase 2 (Pathway 2): Live Data Commons Research
  # --------------------------------------------------------------------------
  # We execute the real, live Data Commons agent to query the MCP server.
  run_step([
      sys.executable,
      "-m",
      "panoramic_insights.phase_2_run_datacommons_research",
      "--input_csv",
      phase_1_output,
      "--output_csv",
      phase_2_dc_output,
      "--place_name",
      PLACE_NAME,
      "--place_dcid",
      PLACE_DCID,
      "--gemini_api_key",
      GEMINI_API_KEY,
      "--model_name",
      "gemini-3.8-flash",
  ])

  # Verify Phase 2 Data Commons schemas and check that a dossier was generated
  df2_dc = pd.read_csv(phase_2_dc_output)
  print(f"[Verify Phase 2 DC] Columns: {list(df2_dc.columns)}")
  assert "research_output" in df2_dc.columns
  print(
      "[Verify Phase 2 DC] Sample research"
      f" output:\n{df2_dc['research_output'].iloc[0][:300]}...\n"
  )

  # --------------------------------------------------------------------------
  # 3. Create a Mock Deep Research Output CSV
  # --------------------------------------------------------------------------
  # To test the parallel pathway combination without running a real Deep Research
  # task (which is slow/expensive), we copy the Phase 1 output and inject a
  # mock qualitative dossier containing Knoxville transit and NY streetlight case studies.
  print(
      "[Test Setup] Creating mock Deep Research output to test combination..."
  )
  df2_deep_mock = df1.copy()
  deep_research_outputs = []
  for _, row in df2_deep_mock.iterrows():
    deep_research_outputs.append(
        f"### Deep Research findings for {row['topic']} -"
        f" {row['proposal']}\nQualitative analysis shows that in other similar"
        " cities (like Knoxville), bus delays were reduced by 15% after"
        " implementing dedicated transit lanes. A study by TransitCenter"
        " indicates that bus reliability is the #1 factor in ridership."
    )
  df2_deep_mock["research_output"] = deep_research_outputs
  df2_deep_mock["interaction_id"] = "mock_deep_research"
  df2_deep_mock.to_csv(phase_2_deep_mock_output, index=False)

  # --------------------------------------------------------------------------
  # 4. Run Phase 3: Structured Insights Synthesis (Combining BOTH)
  # --------------------------------------------------------------------------
  # We pass BOTH the live Data Commons CSV and the mock Deep Research CSV.
  # Phase 3 will merge the qualitative and quantitative dossiers, group them,
  # and call Gemini to extract structured, cited insights.
  run_step([
      sys.executable,
      "-m",
      "panoramic_insights.phase_3_generate_structured_insights",
      "--data_commons_csv",
      phase_2_dc_output,
      "--deep_research_csv",
      phase_2_deep_mock_output,
      "--output_csv",
      phase_3_output,
      "--place_name",
      PLACE_NAME,
      "--gemini_api_key",
      GEMINI_API_KEY,
      "--model_name",
      "gemini-3.8-flash",
  ])

  # Verify Phase 3 output schema and print the generated insights
  df3 = pd.read_csv(phase_3_output)
  print(f"[Verify Phase 3] Columns: {list(df3.columns)}")
  print(f"[Verify Phase 3] Generated {len(df3)} structured insights:")
  for idx, row in df3.iterrows():
    print(
        f" - Insight {idx+1}: Category='{row['category']}',"
        f" Insight='{row['full_insight']}'"
    )
    print(f"   Source: {row['citation_text']} ({row['citation_link']})")

  print("\n[Integration Test Complete] Hybrid pipeline verified successfully!")


if __name__ == "__main__":
  main()
