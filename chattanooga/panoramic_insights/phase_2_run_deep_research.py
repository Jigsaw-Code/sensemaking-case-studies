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
Runs deep research using prompts from a CSV.

Example usage:
python3 -m panoramic_insights.phase_2_run_deep_research \
  --input_csv <input_csv> \
  --output_csv <output_csv> \
  --gemini_api_key <api_key>

Input csv should contain the column research_ideas_prompt.
If there is an interaction_id column set it will be used, rather
than launching new Deep Research jobs (interactions).

Output csv will contain the original columns plus interaction_id
and research_output.
"""

import argparse
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import re
import time

from panoramic_insights import insights_prompts

GROUNDING_REDIRECT_PATTERN = (
    r"\[([^\]]+)\]\((https://vertexaisearch\.cloud\.google\.com/grounding-api-redirect/[^\s\)]+)\)"
)
from google import genai
import pandas as pd
import requests
from tqdm import tqdm


def resolve_redirect_url(url: str, timeout: int = 5) -> str:
  """Follows HTTP redirects for a given URL and returns the final destination URL."""
  headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
  try:
    response = requests.get(
        url, headers=headers, allow_redirects=False, timeout=timeout
    )
    if response.status_code in (301, 302, 303, 307, 308):
      return response.headers.get("Location", url)
    return response.url
  except requests.RequestException as e:
    logging.warning(f"Failed to resolve redirect for {url}: {e}")
    return url


def unshorten_deep_research_urls(text: str, max_workers: int = 10) -> str:
  """Parses text, resolves vertexaisearch proxy links to actual URLs, and replaces Markdown links."""
  if not text or not isinstance(text, str):
    return text

  matches = re.findall(GROUNDING_REDIRECT_PATTERN, text)
  if not matches:
    return text

  unique_urls = list({url for _, url in matches})

  url_map = {}
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {
        executor.submit(resolve_redirect_url, url): url for url in unique_urls
    }
    for future in futures:
      orig_url = futures[future]
      try:
        url_map[orig_url] = future.result()
      except Exception:
        url_map[orig_url] = orig_url

  def replacer(match):
    _, orig_url = match.group(1), match.group(2)
    return url_map.get(orig_url, orig_url)

  return re.sub(GROUNDING_REDIRECT_PATTERN, replacer, text)


def extract_interaction_output_text(interaction) -> str:
  """Reassembles the full report text from a Deep Research interaction.

  A Deep Research agent streams a long report back as a sequence of
  `model_output` steps. The convenience property `interaction.output_text`
  exposes only the final step, which silently discards the earlier sections of
  the report. Joining the text of every `model_output` step recovers the whole
  report. Non-text content such as generated charts is ignored.

  Args:
    interaction: A completed interaction returned by the Interactions API.

  Returns:
    The full report text. Falls back to `interaction.output_text` when no
    `model_output` step carries text, so behaviour never regresses.
  """
  segments = []
  interaction_dict = interaction.to_dict()
  steps = interaction_dict.get("steps") or []
  for step in steps:
    if step.get("type") != "model_output":
      continue
    content = step.get("content") or []
    for element in content:
      if not isinstance(element, dict):
        continue
      text = element.get("text")
      if text:
        segments.append(text)
  if not segments:
    return interaction.output_text or ""
  return "".join(segments)


MAX_POLL_ERRORS = 5


def process_single_deep_research_row(
    index: int,
    prompt: str,
    existing_interaction_id: str | None,
    client: genai.Client,
    dr_agent: str,
) -> tuple[int, str, str, str, str]:
  """Runs a single deep research task end-to-end with retries and polling.

  Returns:
    A tuple of `(index, interaction_id, output_text, status, error)`. On
    failure `output_text` is empty so downstream synthesis never treats an
    error string as research findings.
  """
  interaction_id = (
      str(existing_interaction_id).strip()
      if existing_interaction_id and pd.notna(existing_interaction_id)
      else ""
  )
  if not interaction_id:
    formatted_prompt = insights_prompts.get_deep_research_prompt(str(prompt))
    last_error = ""
    for attempt in range(5):
      try:
        interaction = client.interactions.create(
            input=formatted_prompt,
            agent=dr_agent,
            background=True,
        )
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
          logging.error(f"Failed to submit deep research for row {index}: {e}")
          return index, "", "", "failed", last_error

    if not interaction_id:
      return (
          index,
          "",
          "",
          "failed",
          last_error or "Submission failed: rate limit exceeded.",
      )

  # Poll until completed
  start_time = time.time()
  poll_error_count = 0
  while True:
    if time.time() - start_time > 1800:  # 30 min max
      return (
          index,
          interaction_id,
          "",
          "failed",
          "Deep Research timed out after 30 minutes.",
      )
    try:
      interaction = client.interactions.get(interaction_id)
      poll_error_count = 0
      if interaction.status == "completed":
        output_text = unshorten_deep_research_urls(
            extract_interaction_output_text(interaction)
        )
        return index, interaction_id, output_text, "completed", ""
      elif interaction.status == "failed":
        return (
            index,
            interaction_id,
            "",
            "failed",
            str(interaction.error or "Research failed."),
        )
      elif interaction.status == "cancelled":
        return (
            index,
            interaction_id,
            "",
            "cancelled",
            "Research was cancelled.",
        )
      else:
        time.sleep(15)
    except Exception as e:
      poll_error_count += 1
      if poll_error_count >= MAX_POLL_ERRORS:
        logging.error(
            f"Giving up polling Deep Research row {index} ({interaction_id})"
            f" after {poll_error_count} errors: {e}"
        )
        return index, interaction_id, "", "failed", str(e)
      logging.warning(
          f"Transient error polling Deep Research row {index} ({interaction_id}): {e}"
      )
      time.sleep(15)


async def main(args):
  """Main function to conduct deep research."""
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
  )
  client = genai.Client(api_key=args.gemini_api_key)

  df = pd.read_csv(args.input_csv)

  for col in ["interaction_id", "research_output", "status", "error"]:
    if col not in df.columns:
      df[col] = None
  df["source"] = "deep_research"

  if os.path.exists(args.output_csv):
    try:
      existing_df = pd.read_csv(args.output_csv)
      if len(existing_df) == len(df):
        for col in ["interaction_id", "research_output", "status", "error"]:
          if col in existing_df.columns:
            df[col] = existing_df[col]
    except Exception as e:
      logging.warning(f"Failed to read existing output checkpoint: {e}")

  already_done = 0
  for val in df["research_output"]:
    if pd.notna(val) and str(val).strip():
      already_done += 1
  logging.info(
      f"Starting Deep Research for {len(df)} rows with max concurrency {args.max_concurrency}"
      f" ({already_done}/{len(df)} already completed)..."
  )

  futures = []
  pbar = tqdm(total=len(df), initial=int(already_done), desc="Deep Research Progress")

  with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
    for i, row in df.reset_index(drop=True).iterrows():
      if pd.notna(row.get("research_output")) and str(row.get("research_output")).strip():
        continue
      futures.append(
          executor.submit(
              process_single_deep_research_row,
              i,
              row["research_ideas_prompt"],
              row.get("interaction_id"),
              client,
              args.dr_agent,
          )
      )

    for future in concurrent.futures.as_completed(futures):
      idx, interaction_id, output_text, status, error = future.result()
      df.at[idx, "interaction_id"] = interaction_id
      df.at[idx, "research_output"] = output_text
      df.at[idx, "status"] = status
      df.at[idx, "error"] = error
      pbar.update(1)
      df.to_csv(args.output_csv, index=False)

  pbar.close()
  logging.info(f"Successfully processed. Output written to {args.output_csv}")


def get_args():
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description="Run deep research on topic and opinion pairs."
  )
  parser.add_argument(
      "--input_csv",
      required=True,
      help="Path to the input CSV file.",
  )
  parser.add_argument(
      "--output_csv", required=True, help="Path to save the output CSV file."
  )
  parser.add_argument(
      "--gemini_api_key",
      required=False,
      help=(
          "Google AI Studio API Key. If not provided, uses GEMINI_API_KEY env"
          " var."
      ),
  )
  parser.add_argument(
      "--dr_agent",
      type=str,
      default="deep-research-preview-04-2026",
      help=(
          "The name of the Gemini Deep Research Agent. Default:"
          " deep-research-preview-04-2026."
      ),
  )
  parser.add_argument(
      "--max_concurrency",
      type=int,
      default=10,
      help=(
          "Maximum number of concurrent workers for Deep Research tasks."
          " Default: 10."
      ),
  )
  return parser.parse_args()


if __name__ == "__main__":
  asyncio.run(main(get_args()))
