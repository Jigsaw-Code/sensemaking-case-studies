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
Creates prompts for research tasks from a CSV containing quotes categorized
by topic and proposal.

Example usage:
python3 -m panoramic_insights.phase_1_generate_research_ideas_prompts \
  --input_csv categorized_without_other_filtered.csv \
  --output_csv <output_csv> \
  --gemini_api_key <gemini_api_key> 

input_csv is expected to be the output of categorization_runner,
containing topic, proposal, and quote columns.
"""

import argparse
import asyncio
import json
import pandas as pd

from src.models import genai_model
from panoramic_insights import insights_prompts


def validate_proposal_column(df: pd.DataFrame) -> None:
  """Validates that proposal column exists in the DataFrame."""
  if "proposal" not in df.columns:
    raise ValueError("Input CSV must contain a 'proposal' column.")


async def generate_research_ideas_prompts(
    topic_proposal_df: pd.DataFrame,
    model: genai_model.GenaiModel,
    place_name: str = "",
    peer_cities: list[str] | None = None,
    max_concurrent_calls: int | None = None,
) -> pd.Series:
  """Generates research ideas prompts for grouped topics and proposals."""
  prompt_objs = []
  for _, row in topic_proposal_df.iterrows():
    topic = row["topic"]
    proposal = row["proposal"]
    quotes_val = row.get("quotes", "[]")
    quotes = json.loads(quotes_val) if isinstance(quotes_val, str) else quotes_val
    prompt_objs.append({
        "prompt": insights_prompts.get_research_ideas_prompt(
            topic=topic,
            proposal=proposal,
            quotes=quotes,
            place_name=place_name,
            peer_cities=peer_cities,
        ),
        "topic": topic,
        "proposal": proposal,
    })
  response_df = await insights_prompts.generate_text_in_parallel(
      model, prompt_objs, max_concurrent_calls=max_concurrent_calls
  )
  return response_df["result"]


async def main(args):
  model = genai_model.GenaiModel(
      model_name=args.model_name,
      gemini_api_key=args.gemini_api_key,
  )

  # Input CSV can have 1 row per quote or 1 row per proposal
  all_quotes_df = pd.read_csv(args.input_csv)
  validate_proposal_column(all_quotes_df)

  # Group quotes by topic then proposal, aggregating quotes if available.
  if "quote" in all_quotes_df.columns:
    topic_proposal_df = (
        all_quotes_df.groupby(["topic", "proposal"])
        .agg({"quote": lambda s: json.dumps(list(s))})
        .reset_index()
        .rename(columns={"quote": "quotes"})
    )
  else:
    topic_proposal_df = (
        all_quotes_df[["topic", "proposal"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    topic_proposal_df["quotes"] = "[]"

  peer_cities = None
  if args.peer_cities:
    peer_cities = []
    for c in args.peer_cities.split(","):
      c_clean = c.strip()
      if c_clean:
        peer_cities.append(c_clean)
  topic_proposal_df["research_ideas_prompt"] = (
      await generate_research_ideas_prompts(
          topic_proposal_df,
          model,
          place_name=args.place_name,
          peer_cities=peer_cities,
          max_concurrent_calls=args.concurrency,
      )
  )

  topic_proposal_df.to_csv(args.output_csv, index=False)
  print(f"Successfully processed. Output written to {args.output_csv}")


def get_args():
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description="Generates research prompts for a CSV of categorized quotes."
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
      "--place_name",
      type=str,
      default="",
      help="Target municipality or region name.",
  )
  parser.add_argument(
      "--peer_cities",
      type=str,
      default="",
      help="Comma-separated list of peer municipalities.",
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
      "--model_name",
      type=str,
      default="gemini-3.8-flash",
      help="The name of the AI model to use. Default: gemini-3.8-flash.",
  )
  parser.add_argument(
      "--concurrency",
      type=int,
      default=None,
      help="Maximum number of concurrent API calls.",
  )
  return parser.parse_args()


if __name__ == "__main__":
  asyncio.run(main(get_args()))
