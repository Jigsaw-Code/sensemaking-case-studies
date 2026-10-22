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

"""
Runs file agent using prompts from a CSV.

Example usage:
python3 -m panoramic_insights.phase_2_run_file_agent \
  --input_csv <input_csv> \
  --output_csv <output_csv> \
  --gemini_api_key <api_key> \
  --agent_environment_id <env_id>

Input csv should contain the columns topic and proposal.
If there is an agent_interaction_id column set it will be used, rather
than launching new file agent jobs (interactions).

Output csv will contain the original columns plus agent_interaction_id,
agent_environment_id, sandbox_work_dir, and research_output.
"""

import argparse
import asyncio
import concurrent.futures
import enum
import logging
import os
import queue
import time

from panoramic_insights import insights_prompts
from panoramic_insights import phase_2_run_deep_research
from google import genai
import pandas as pd
from tqdm import tqdm


class InteractionStatus(str, enum.Enum):
  COMPLETED = "completed"
  FAILED = "failed"
  CANCELLED = "cancelled"


MAX_POLL_ERRORS = 5


def process_single_file_agent_row(
    index: int,
    row: pd.Series,
    existing_interaction_id: str | None,
    client: genai.Client,
    agent_name: str,
    env_id: str | None,
    place_name: str,
    timeout_seconds: int = 2700,
    network_mode: str = "disabled",
) -> tuple[int, str, str, str | None, str, str, str]:
  """Runs a single file agent task end-to-end with retries and polling.

  Args:
    index: Row index, used to isolate the agent's sandbox working directory.
    row: The phase 1 output row supplying topic, proposal and research ideas.
    existing_interaction_id: Resume an in-flight interaction instead of
      starting a new one.
    client: The GenAI client used to create and poll interactions.
    agent_name: Agent model name to run.
    env_id: Identifier of a pre-provisioned sandbox environment.
    place_name: Target municipality, used to frame the prompt.
    timeout_seconds: How long to poll before giving up on the interaction.
    network_mode: Outbound network setting for a freshly created sandbox.

  Returns:
    A tuple of `(index, interaction_id, output_text, env_id, sandbox_work_dir,
    status, error)`. On failure `output_text` is empty so downstream synthesis
    never mistakes an error message for research findings.
  """
  sandbox_work_dir = f"/workspace/runs/row_{index}"
  interaction_id = (
      str(existing_interaction_id).strip()
      if existing_interaction_id and pd.notna(existing_interaction_id)
      else ""
  )
  if not interaction_id:
    prompt = insights_prompts.get_file_agent_prompt(
        topic=row["topic"],
        proposal=row["proposal"],
        sandbox_work_dir=sandbox_work_dir,
        place_name=place_name,
        research_ideas_prompt=row.get("research_ideas_prompt", ""),
    )

    kwargs = {
        "input": prompt,
        "agent": agent_name,
        "background": True,
        "tools": [{"type": "code_execution"}],
    }
    if env_id:
      kwargs["environment"] = env_id
    else:
      kwargs["environment"] = {
          "type": "remote",
          "network": network_mode,
      }

    last_error = ""
    for attempt in range(5):
      try:
        interaction = client.interactions.create(**kwargs)
        interaction_id = interaction.id
        break
      except Exception as e:
        last_error = str(e)
        if "429" in last_error or "503" in last_error:
          sleep_time = (2 ** attempt) * 2 + 1
          logging.warning(
              f"Rate limit / transient error on row {index}: {e}. Retrying in {sleep_time}s..."
          )
          time.sleep(sleep_time)
        else:
          logging.error(f"Failed to submit file agent for row {index}: {e}")
          return (
              index,
              "",
              "",
              env_id,
              sandbox_work_dir,
              InteractionStatus.FAILED.value,
              last_error,
          )

    if not interaction_id:
      return (
          index,
          "",
          "",
          env_id,
          sandbox_work_dir,
          InteractionStatus.FAILED.value,
          last_error or "Failed to start after retries.",
      )

  # Poll until completed
  start_time = time.time()
  cancel_count = 0
  poll_error_count = 0
  while True:
    if time.time() - start_time > timeout_seconds:
      timeout_min = int(timeout_seconds / 60)
      return (
          index,
          interaction_id,
          "",
          env_id,
          sandbox_work_dir,
          InteractionStatus.FAILED.value,
          f"Job timed out after {timeout_min} minutes.",
      )
    try:
      interaction = client.interactions.get(interaction_id)
      poll_error_count = 0
      if interaction.status == InteractionStatus.COMPLETED:
        output_text = (
            phase_2_run_deep_research.extract_interaction_output_text(
                interaction
            )
        )
        return (
            index,
            interaction_id,
            output_text,
            env_id,
            sandbox_work_dir,
            InteractionStatus.COMPLETED.value,
            "",
        )
      elif interaction.status == InteractionStatus.FAILED:
        return (
            index,
            interaction_id,
            "",
            env_id,
            sandbox_work_dir,
            InteractionStatus.FAILED.value,
            str(interaction.error or "Interaction failed."),
        )
      elif interaction.status == InteractionStatus.CANCELLED:
        cancel_count += 1
        if cancel_count > 3:
          return (
              index,
              interaction_id,
              "",
              env_id,
              sandbox_work_dir,
              InteractionStatus.CANCELLED.value,
              "Job was cancelled.",
          )
        time.sleep(10 * cancel_count)
      else:
        time.sleep(10)
    except Exception as e:
      poll_error_count += 1
      if poll_error_count >= MAX_POLL_ERRORS:
        logging.error(
            f"Giving up polling file agent row {index} ({interaction_id}) after"
            f" {poll_error_count} errors: {e}"
        )
        return (
            index,
            interaction_id,
            "",
            env_id,
            sandbox_work_dir,
            InteractionStatus.FAILED.value,
            str(e),
        )
      logging.warning(
          f"Transient error polling file agent row {index} ({interaction_id}): {e}"
      )
      time.sleep(10)


async def main(args):
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
  )
  client = genai.Client(api_key=args.gemini_api_key)

  df = pd.read_csv(args.input_csv)
  if "proposal" not in df.columns:
    raise ValueError("Input CSV must contain a 'proposal' column.")

  for col in [
      "agent_interaction_id",
      "env_id",
      "sandbox_work_dir",
      "research_output",
      "status",
      "error",
  ]:
    if col not in df.columns:
      df[col] = None
  df["source"] = "file_agent"

  if os.path.exists(args.output_csv):
    try:
      existing_df = pd.read_csv(args.output_csv)
      if len(existing_df) == len(df):
        for col in [
            "agent_interaction_id",
            "env_id",
            "sandbox_work_dir",
            "research_output",
            "status",
            "error",
        ]:
          if col in existing_df.columns:
            df[col] = existing_df[col]
    except Exception as e:
      logging.warning(f"Failed to read existing output checkpoint: {e}")

  # Build sandbox environment pool
  env_pool = []
  if args.agent_environment_id:
    raw_parts = str(args.agent_environment_id).split(",")
    for part in raw_parts:
      cleaned = part.strip()
      if cleaned:
        env_pool.append(cleaned)

  # Thread-safe work-stealing queue of available sandbox tokens
  env_queue = queue.Queue()
  if len(env_pool) > 1:
    for env in env_pool:
      for _ in range(args.max_workers_per_env):
        env_queue.put(env)
    effective_concurrency = env_queue.qsize()
    logging.info(
        f"Multi-sandbox scheduling active: {len(env_pool)} sandboxes with"
        f" {args.max_workers_per_env} worker(s) each (total pool capacity:"
        f" {effective_concurrency})."
    )
  elif len(env_pool) == 1:
    for _ in range(args.max_concurrency):
      env_queue.put(env_pool[0])
    effective_concurrency = args.max_concurrency
    logging.info(
        f"Single sandbox scheduling active: {env_pool[0]} with concurrency"
        f" {effective_concurrency}."
    )
  else:
    for _ in range(args.max_concurrency):
      env_queue.put(None)
    effective_concurrency = args.max_concurrency
    logging.info(
        f"Remote sandbox scheduling (no custom env_id) with concurrency"
        f" {effective_concurrency}."
    )

  already_done = 0
  for val in df["research_output"]:
    if pd.notna(val) and str(val).strip():
      already_done += 1
  logging.info(
      f"Starting File Agent for {len(df)} rows with effective concurrency {effective_concurrency}"
      f" ({already_done}/{len(df)} already completed)..."
  )

  futures = []
  pbar = tqdm(total=len(df), initial=int(already_done), desc="File Agent Progress")

  def _scheduled_worker(
      row_idx: int,
      row_data: pd.Series,
      existing_inter_id: str | None,
  ):
    assigned_env = env_queue.get()
    try:
      if assigned_env:
        logging.info(
            f"Dispatching row {row_idx} ({row_data.get('topic')}) to sandbox"
            f" environment {assigned_env}"
        )
      return process_single_file_agent_row(
          index=row_idx,
          row=row_data,
          existing_interaction_id=existing_inter_id,
          client=client,
          agent_name=args.file_agent_model_name,
          env_id=assigned_env,
          place_name=args.place_name,
          timeout_seconds=args.timeout_seconds,
          network_mode=args.network,
      )
    finally:
      env_queue.put(assigned_env)
      env_queue.task_done()

  with concurrent.futures.ThreadPoolExecutor(max_workers=effective_concurrency) as executor:
    for i, row in df.reset_index(drop=True).iterrows():
      if pd.notna(row.get("research_output")) and str(row.get("research_output")).strip():
        continue
      futures.append(
          executor.submit(
              _scheduled_worker,
              i,
              row,
              row.get("agent_interaction_id"),
          )
      )

    for future in concurrent.futures.as_completed(futures):
      (
          idx,
          interaction_id,
          output_text,
          used_env_id,
          sandbox_work_dir,
          status,
          error,
      ) = future.result()
      df.at[idx, "agent_interaction_id"] = interaction_id
      df.at[idx, "env_id"] = used_env_id
      df.at[idx, "sandbox_work_dir"] = sandbox_work_dir
      df.at[idx, "research_output"] = output_text
      df.at[idx, "status"] = status
      df.at[idx, "error"] = error
      pbar.update(1)
      df.to_csv(args.output_csv, index=False)

  pbar.close()
  logging.info(f"Successfully processed. Output written to {args.output_csv}")


def get_args():
  parser = argparse.ArgumentParser(
      description="Run file agent on topic and opinion pairs."
  )
  parser.add_argument(
      "--input_csv", required=True, help="Path to the input CSV file."
  )
  parser.add_argument(
      "--output_csv", required=True, help="Path to save the output CSV file."
  )
  parser.add_argument(
      "--place_name",
      type=str,
      default="",
      help="Name of the place/city/region under investigation.",
  )
  parser.add_argument(
      "--gemini_api_key", required=False, help="Google AI Studio API Key."
  )
  parser.add_argument(
      "--file_agent_model_name",
      type=str,
      default="antigravity-preview-05-2026",
      help="The name of the Gemini file agent.",
  )
  parser.add_argument(
      "--agent_environment_id",
      type=str,
      default=os.environ.get("AGENT_ENVIRONMENT_ID", ""),
      help=(
          "The execution environment ID(s) for the data sandbox(es). Pass a single"
          " ID or a comma-separated list of IDs."
      ),
  )
  parser.add_argument(
      "--max_workers_per_env",
      type=int,
      default=1,
      help=(
          "Max concurrent worker threads allowed per individual sandbox"
          " environment when multiple environments are provided (default: 1)."
      ),
  )
  parser.add_argument(
      "--timeout_seconds",
      type=int,
      default=2700,
      help="Timeout in seconds for polling a single file agent interaction (default: 2700s / 45m).",
  )
  parser.add_argument(
      "--max_concurrency",
      type=int,
      default=2,
      help="Max concurrent agent API calls running at any given time. Default: 2.",
  )
  parser.add_argument(
      "--network",
      type=str,
      default="enabled" if bool(insights_prompts.TOOLS_REGISTRY) else "disabled",
      choices=["enabled", "disabled"],
      help=(
          "Network mode for remote code execution environment ('enabled' or"
          " 'disabled'). Defaults to 'enabled' if tools are configured."
      ),
  )
  return parser.parse_args()


if __name__ == "__main__":
  asyncio.run(main(get_args()))
