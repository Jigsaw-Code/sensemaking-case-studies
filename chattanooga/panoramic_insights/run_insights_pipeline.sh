#!/bin/bash
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

set -ex  # print all commands and exit on failures.

# Ensure we are in the root of the repo
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_STUDY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$CASE_STUDY_DIR" || exit 1

# Check for required arguments
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <input_csv> <output_dir> [gemini_api_key] [agent_environment_id] [fact_checking_environment_id]"
  exit 1
fi

INPUT_CSV=$1
OUTPUT_DIR=$2
GEMINI_API_KEY=${3:-$GEMINI_API_KEY}
if [ -z "$GEMINI_API_KEY" ]; then
  echo "Error: GEMINI_API_KEY must be provided as 3rd argument or set in environment."
  exit 1
fi

AGENT_ENVIRONMENT_ID=${4:-"34fc829c826a588c9fc60a83eb6c19cc"}
FACT_CHECKING_ENVIRONMENT_ID=${5:-"34fc829c826a588c9fc60a83eb6c19cc"}
MODEL_NAME=${MODEL_NAME:-"gemini-3.8-flash"}
DR_AGENT=${DR_AGENT:-"deep-research-preview-04-2026"}
FILE_AGENT_MODEL_NAME=${FILE_AGENT_MODEL_NAME:-"antigravity-preview-05-2026"}
DC_MODEL_NAME=${DC_MODEL_NAME:-"gemini-3.7-flash"}

mkdir -p "$OUTPUT_DIR"

# Intermediate files
PHASE_1_OUTPUT="${OUTPUT_DIR}/phase_1_prompts.csv"
PHASE_2_DEEP_OUTPUT="${OUTPUT_DIR}/phase_2_deep_research.csv"
PHASE_2_DC_OUTPUT="${OUTPUT_DIR}/phase_2_data_commons_research.csv"
PHASE_2_FILE_AGENT_OUTPUT="${OUTPUT_DIR}/phase_2_file_agent_research.csv"
PHASE_3_OUTPUT="${OUTPUT_DIR}/phase_3_structured_insights.csv"
PHASE_3_REPORTS_OUTPUT="${OUTPUT_DIR}/phase_3_policy_reports.csv"
PHASE_4_OUTPUT="${OUTPUT_DIR}/phase_4_evals"

# Target place and peer cities configuration
PLACE_NAME="Chattanooga, Tennessee"
PLACE_DCID="geoId/4714000"
PEER_CITIES="Knoxville, Huntsville, Mobile, Clarksville, Des Moines, Richmond"

if [ -f "$PHASE_1_OUTPUT" ]; then
  echo "Phase 1 output already exists at $PHASE_1_OUTPUT, skipping generation."
else
  echo "Running Phase 1: Generating research ideas prompts..."
  python3 -m panoramic_insights.phase_1_generate_research_ideas_prompts \
    --input_csv "$INPUT_CSV" \
    --output_csv "$PHASE_1_OUTPUT" \
    --place_name "$PLACE_NAME" \
    --peer_cities "$PEER_CITIES" \
    --gemini_api_key "$GEMINI_API_KEY" \
    --model_name "$MODEL_NAME"
fi

if [ -f "$PHASE_2_DEEP_OUTPUT" ]; then
  echo "Phase 2 (Pathway 1) output already exists at $PHASE_2_DEEP_OUTPUT, skipping generation."
else
  echo "Running Phase 2 (Pathway 1): Running Deep Research..."
  python3 -m panoramic_insights.phase_2_run_deep_research \
    --input_csv "$PHASE_1_OUTPUT" \
    --output_csv "$PHASE_2_DEEP_OUTPUT" \
    --gemini_api_key "$GEMINI_API_KEY" \
    --dr_agent "$DR_AGENT"
fi

if [ -f "$PHASE_2_DC_OUTPUT" ]; then
  echo "Phase 2 (Pathway 2) output already exists at $PHASE_2_DC_OUTPUT, skipping generation."
else
  echo "Running Phase 2 (Pathway 2): Running Data Commons Research..."
  python3 -m panoramic_insights.phase_2_run_datacommons_research \
    --input_csv "$PHASE_1_OUTPUT" \
    --output_csv "$PHASE_2_DC_OUTPUT" \
    --place_name "$PLACE_NAME" \
    --place_dcid "$PLACE_DCID" \
    --gemini_api_key "$GEMINI_API_KEY" \
    --model_name "$DC_MODEL_NAME" \
    --max_concurrency 2
fi

if [ -f "$PHASE_2_FILE_AGENT_OUTPUT" ]; then
  echo "Phase 2 (Pathway 3) output already exists at $PHASE_2_FILE_AGENT_OUTPUT, skipping generation."
else
  echo "Running Phase 2 (Pathway 3): Running File Agent Research..."
  python3 -m panoramic_insights.phase_2_run_file_agent \
    --input_csv "$PHASE_1_OUTPUT" \
    --output_csv "$PHASE_2_FILE_AGENT_OUTPUT" \
    --place_name "$PLACE_NAME" \
    --agent_environment_id "$AGENT_ENVIRONMENT_ID" \
    --gemini_api_key "$GEMINI_API_KEY" \
    --file_agent_model_name "$FILE_AGENT_MODEL_NAME"
fi

if [ -f "$PHASE_3_OUTPUT" ]; then
  echo "Phase 3 insights output already exists at $PHASE_3_OUTPUT, skipping generation."
else
  echo "Running Phase 3 (Takeaway Distillation): Generating structured insights..."
  python3 -m panoramic_insights.phase_3_generate_structured_insights \
    --stage insights \
    --deep_research_csv "$PHASE_2_DEEP_OUTPUT" \
    --data_commons_csv "$PHASE_2_DC_OUTPUT" \
    --file_agent_csv "$PHASE_2_FILE_AGENT_OUTPUT" \
    --output_csv "$PHASE_3_OUTPUT" \
    --place_name "$PLACE_NAME" \
    --gemini_api_key "$GEMINI_API_KEY" \
    --model_name "$MODEL_NAME"
fi

echo "Running Phase 3 (Fact-Checking): Verifying citation traces..."
python3 -m panoramic_insights.phase_3_verify_citations \
  --input_csv "$PHASE_3_OUTPUT" \
  --file_agent_environment_id "$AGENT_ENVIRONMENT_ID" \
  --fact_checking_agent_environment_id "$FACT_CHECKING_ENVIRONMENT_ID" \
  --file_agent_metadata_csv "$PHASE_2_FILE_AGENT_OUTPUT" \
  --gemini_api_key "$GEMINI_API_KEY" \
  --model_name "$MODEL_NAME"

echo "Running Phase 4 (Report Synthesis): Composing policy reports from verified insights..."
python3 -m panoramic_insights.phase_3_generate_structured_insights \
  --stage reports \
  --deep_research_csv "$PHASE_2_DEEP_OUTPUT" \
  --data_commons_csv "$PHASE_2_DC_OUTPUT" \
  --file_agent_csv "$PHASE_2_FILE_AGENT_OUTPUT" \
  --output_csv "$PHASE_3_OUTPUT" \
  --reports_csv "$PHASE_3_REPORTS_OUTPUT" \
  --place_name "$PLACE_NAME" \
  --gemini_api_key "$GEMINI_API_KEY" \
  --model_name "$MODEL_NAME"

echo "Running Phase 4 (Quality Evaluation): Evaluating insights..."
python3 -m panoramic_insights.evals \
  --baseline_csv "$PHASE_3_OUTPUT" \
  --output_dir "$PHASE_4_OUTPUT" \
  --api_key "$GEMINI_API_KEY" \
  --metric_name "insights_metric" \
  --model_name "$MODEL_NAME"

echo "Pipeline complete. Results are in $OUTPUT_DIR"


