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
Generates structured insights from research outputs.

Example usage:
python3 -m panoramic_insights.phase_3_generate_structured_insights \
  --deep_research_csv <deep_research_csv> \
  --data_commons_csv <data_commons_csv> \
  --file_agent_csv <file_agent_csv> \
  --output_csv <output_csv> \
  --gemini_api_key <gemini_api_key>

Research CSVs are expected to contain columns: topic, proposal,
research_output, research_ideas_prompt. Each research CSV is passed via its
explicit source flag (`--deep_research_csv`, `--data_commons_csv`,
`--file_agent_csv`).
"""

import argparse
import asyncio
import collections
import json
import logging
import os
import re
import urllib.parse
import pandas as pd

from panoramic_insights import insights_prompts
from src.models import genai_model


INSIGHTS_DF_COLUMNS = [
    "topic",
    "proposal",
    "full_insight",
    "citation_text",
    "citation_link",
    "category",
    "source",
]

# One row per policy proposal. `report_json` holds the structured JSON object:
# `{"policy", "problem_definition": {"tldr", "insights": [{"category",
# "text_with_citations"}]}, "recommendation": {"tldr", "insights": [...]},
# "quotes": [...], "references": [{"id", "title", "url"}]}`.
REPORTS_DF_COLUMNS = [
    "topic",
    "proposal",
    "report_json",
]

_LOCAL_LINK_RE = re.compile(r"/data/\S*[^\s.,;:)\]}>\"'`]")

# Research sources produced by Phase 2, in the order the flags are listed.
RESEARCH_SOURCES = ("deep_research", "data_commons", "file_agent")


def resolve_dataset_links(text: str) -> tuple[str, list[str]]:
  """Rewrites local dataset links to source URLs and reports what did not match.

  The registry in config.json is the single source of truth: each entry already
  carries the `/data/...` path the model is told to cite and the `source_url` it
  should be published as. Anything still pointing at /data/ afterwards is a path
  the model invented, and is returned rather than repaired.

  Args:
    text: Arbitrary text that may contain local dataset links.

  Returns:
    A tuple of the rewritten text and the sorted list of unresolved links.
  """
  if not text:
    return text, []

  registry = getattr(insights_prompts, "LOCAL_DATASETS_REGISTRY", {})
  for info in registry.values():
    path = info.get("path", "")
    source_url = info.get("source_url", "")
    if not path or not source_url:
      continue
    text = text.replace(path, source_url)

  unresolved = sorted(set(_LOCAL_LINK_RE.findall(text)))
  return text, unresolved


# Model calls in this phase are large and run concurrently, so they are the
# first to be shed when Gemini is overloaded (503 "Preempted out of decode
# queue"). Backoff is per call, so one success elsewhere does not reset it.
GENERATION_MAX_ATTEMPTS = 8
GENERATION_INITIAL_DELAY_SECONDS = 5.0
GENERATION_MAX_DELAY_SECONDS = 120.0


class GenerationFailedError(RuntimeError):
  """Raised when a model call returns no usable text after every retry."""


async def _generate_text_with_retry(
    model: genai_model.GenaiModel,
    prompt: str,
    run_name: str,
    response_schema: dict | None,
    label: str,
) -> str:
  """Calls the model until it returns non-empty text, backing off on failure.

  `GenaiModel.generate_content` reports API failures (503, 429, timeouts) as
  an `error` entry in the returned dict rather than raising, so both the dict
  and exceptions are inspected here.

  Args:
    model: The Gemini wrapper used to generate content.
    prompt: The prompt text.
    run_name: Label passed through to the model for tracing.
    response_schema: Optional JSON schema for structured output.
    label: Human-readable identifier for this call, used in logs.

  Returns:
    The non-empty response text.

  Raises:
    GenerationFailedError: If every attempt failed or returned empty text.
  """
  delay = GENERATION_INITIAL_DELAY_SECONDS
  last_error = ""
  attempt = 0
  while attempt < GENERATION_MAX_ATTEMPTS:
    attempt += 1
    try:
      resp = await model.generate_content(
          prompt=prompt,
          run_name=run_name,
          response_schema=response_schema,
      )
      text = resp.get("text", "") if isinstance(resp, dict) else ""
      if isinstance(text, str) and text.strip():
        return text
      if isinstance(resp, dict) and resp.get("error"):
        last_error = repr(resp["error"])
      else:
        last_error = "empty response text"
    except Exception as e:  # pylint: disable=broad-exception-caught
      last_error = repr(e)

    if attempt < GENERATION_MAX_ATTEMPTS:
      logging.warning(
          "%s failed (attempt %d/%d): %s. Retrying in %.0fs.",
          label,
          attempt,
          GENERATION_MAX_ATTEMPTS,
          last_error[:300],
          delay,
      )
      await asyncio.sleep(delay)
      delay = min(delay * 2, GENERATION_MAX_DELAY_SECONDS)

  raise GenerationFailedError(
      f"{label} failed after {GENERATION_MAX_ATTEMPTS} attempts. Last error:"
      f" {last_error[:500]}"
  )


def _raise_if_any_failed(desc: str, failed_keys: list, total: int) -> None:
  """Fails the phase when any model call produced no output.

  Successful calls are already in the checkpoint, so a rerun only repeats the
  failed ones. Continuing would silently publish a partial report.

  Args:
    desc: Name of the batch, used in the error message.
    failed_keys: Cache keys of the calls that produced no output.
    total: Number of calls attempted in this batch.

  Raises:
    RuntimeError: If `failed_keys` is not empty.
  """
  if not failed_keys:
    return
  shown = []
  for key in failed_keys[:10]:
    shown.append(str(key)[:200])
  raise RuntimeError(
      f"{desc}: {len(failed_keys)} of {total} model calls failed after"
      f" retries; refusing to write partial output. Failed: {shown}"
  )


async def generate_structured_insights(
    df: pd.DataFrame,
    model: genai_model.GenaiModel,
    place_name: str = "",
    max_concurrent_calls: int | None = None,
    checkpoint_file: str | None = None,
) -> pd.Series:
  prompt_objs = []
  for _, row in df.iterrows():
    proposal_val = row["proposal"]
    for category in insights_prompts.get_criteria_categories():
      prompt_objs.append({
          "prompt": insights_prompts.get_structured_insights_prompt(
              target_category=category,
              research_output=row["research_output"],
              place_name=place_name,
              research_ideas_prompt=row["research_ideas_prompt"],
          ),
          "topic": row["topic"],
          "proposal": proposal_val,
          "target_category": category,
          "response_schema": (
              insights_prompts.InsightOutputStructure.model_json_schema()
          ),
      })

  cached_map = await _run_prompts_with_checkpoint(
      prompt_objs,
      model,
      run_name="structured_insights",
      key_fields=["topic", "proposal", "target_category"],
      desc="Generating insights",
      max_concurrent_calls=max_concurrent_calls,
      checkpoint_file=checkpoint_file,
  )

  results_by_topic_proposal = collections.defaultdict(list)
  for p in prompt_objs:
    key = (p["topic"], p["proposal"])
    prompt_key = (p["topic"], p["proposal"], p["target_category"])
    raw_result = cached_map.get(prompt_key, "")
    try:
      obj = json.loads(raw_result)
      results_by_topic_proposal[key].extend(obj.get("insights", []))
    except Exception as e:
      logging.warning(
          f"Failed to parse structured insights JSON for {prompt_key}: {e}. Raw: {str(raw_result)[:200]}"
      )

  combined_results = []
  for _, row in df.iterrows():
    key = (row["topic"], row["proposal"])
    combined_results.append(
        json.dumps({"insights": results_by_topic_proposal.get(key, [])})
    )
  return pd.Series(combined_results, index=df.index)


def create_insights_dataframe(df: pd.DataFrame) -> pd.DataFrame:
  """Reads a dataframe with 1 row per topic and proposal, outputs 1 row per valid insight.

  Insights citing a `/data/` path that is not in the registry are dropped. Such
  a citation names a dataset that does not exist, so the numbers it supports
  cannot be checked by anyone.

  Args:
    df: One row per topic and proposal, with an `insights` JSON column.

  Returns:
    One row per valid insight.
  """
  new_rows = []
  rejected_count = 0
  for _, row in df.iterrows():
    insights_json_str = row.get("insights", "{}")
    try:
      insights_obj = json.loads(insights_json_str)
    except Exception:
      insights_obj = {}

    insights_list = insights_obj.get("insights", [])
    proposal_val = row["proposal"]
    for insight in insights_list:
      full_text = insight.get("full_insight", "")
      if not full_text or not isinstance(full_text, str) or not full_text.strip():
        continue

      full_text, unresolved_full = resolve_dataset_links(full_text.strip())
      citation_link, unresolved_citation = resolve_dataset_links(
          insight.get("citation_link", "")
      )

      unresolved = sorted(set(unresolved_full) | set(unresolved_citation))
      if unresolved:
        rejected_count += 1
        logging.error(
            "[CitationRegistry] Dropping insight for topic '%s' category '%s':"
            " cites %d dataset path(s) absent from the registry: %s",
            row["topic"],
            insight.get("category", ""),
            len(unresolved),
            ", ".join(unresolved),
        )
        continue

      new_rows.append({
          "topic": row["topic"],
          "proposal": proposal_val,
          "full_insight": full_text,
          "citation_text": insight.get("citation_text", ""),
          "citation_link": citation_link,
          "category": insight.get("category", ""),
          "source": insight.get("source", ""),
      })

  if rejected_count:
    logging.error(
        "[CitationRegistry] Dropped %d insight(s) citing unregistered datasets.",
        rejected_count,
    )
  if not new_rows:
    return pd.DataFrame(columns=INSIGHTS_DF_COLUMNS)
  return pd.DataFrame(new_rows)


async def _run_prompts_with_checkpoint(
    prompt_objs: list[dict],
    model: genai_model.GenaiModel,
    run_name: str,
    key_fields: list[str],
    desc: str,
    max_concurrent_calls: int | None = None,
    checkpoint_file: str | None = None,
) -> dict:
  """Runs prompts concurrently, resuming from and appending to a checkpoint.

  Args:
    prompt_objs: Dicts each carrying a `prompt` plus every field in
      `key_fields`, which together identify the call.
    model: The Gemini wrapper used to generate content.
    run_name: Label passed through to the model for tracing.
    key_fields: Field names whose values form the cache key.
    desc: Progress bar description.
    max_concurrent_calls: Ceiling on in-flight calls. Defaults to 10.
    checkpoint_file: Optional JSONL file to resume from and append to.

  Returns:
    A mapping from the tuple of key field values to the raw response text.
  """
  cached_map = {}
  if checkpoint_file and os.path.exists(checkpoint_file):
    try:
      with open(checkpoint_file, "r", encoding="utf-8") as f:
        for line in f:
          line_str = line.strip()
          if not line_str:
            continue
          rec = json.loads(line_str)
          res_val = rec.get("result", "")
          if isinstance(res_val, str) and res_val.strip():
            key_values = []
            for field in key_fields:
              key_values.append(rec.get(field, ""))
            cached_map[tuple(key_values)] = res_val
      print(f"Loaded {len(cached_map)} cached items from {checkpoint_file}")
    except Exception as e:
      logging.warning(f"Error loading checkpoint {checkpoint_file}: {e}")

  missing_prompts = []
  for obj in prompt_objs:
    key_values = []
    for field in key_fields:
      key_values.append(obj.get(field, ""))
    if tuple(key_values) not in cached_map:
      missing_prompts.append(obj)

  if not missing_prompts:
    return cached_map

  print(f"{desc}: {len(missing_prompts)} calls ({len(cached_map)} from cache)...")
  semaphore = asyncio.Semaphore(max_concurrent_calls or 10)
  lock = asyncio.Lock()
  import tqdm.asyncio
  pbar = tqdm.asyncio.tqdm(total=len(missing_prompts), desc=desc)

  async def run_one(obj):
    async with semaphore:
      key_values = []
      for field in key_fields:
        key_values.append(obj.get(field, ""))
      label = f"{run_name} {tuple(key_values)}"
      try:
        raw_result = await _generate_text_with_retry(
            model, obj["prompt"], run_name, obj.get("response_schema"), label
        )
      except GenerationFailedError as e:
        logging.error(str(e))
        raw_result = ""

      rec = {"result": raw_result}
      for field in key_fields:
        rec[field] = obj.get(field, "")
      if checkpoint_file and raw_result.strip():
        async with lock:
          with open(checkpoint_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
      pbar.update(1)
      return rec

  tasks = [asyncio.create_task(run_one(p)) for p in missing_prompts]
  completed_recs = await asyncio.gather(*tasks)
  pbar.close()
  failed_keys = []
  for rec in completed_recs:
    key_values = []
    for field in key_fields:
      key_values.append(rec.get(field, ""))
    cached_map[tuple(key_values)] = rec["result"]
    if not rec["result"].strip():
      failed_keys.append(tuple(key_values))
  _raise_if_any_failed(desc, failed_keys, len(missing_prompts))
  return cached_map


def _warn_on_unresolved_links(label: str, text: str) -> None:
  """Logs an error when composed report text still carries a local file path.

  Insights reaching this point already had their `/data/` paths swapped for
  Drive URLs, so anything left is something the report model introduced.

  Args:
    label: Human-readable identifier for the text, used in the log line.
    text: The composed report text to inspect.
  """
  _, unresolved = resolve_dataset_links(text)
  if unresolved:
    logging.error(
        "Report %s contains %d local path(s) not present in the source"
        " insights: %s",
        label,
        len(unresolved),
        ", ".join(unresolved),
    )


def parse_quotes(value: object) -> list[str]:
  """Reads a proposal's participant quotes from a `quotes` cell.

  Args:
    value: A JSON-encoded list of quotes (as phase 1 writes it), a list, or a
      missing value.

  Returns:
    The non-blank quotes, in their original order. Empty when there are none
    or the cell cannot be parsed.
  """
  if isinstance(value, list):
    items = value
  elif isinstance(value, str):
    if not value.strip():
      return []
    try:
      items = json.loads(value)
    except json.JSONDecodeError:
      logging.error("Could not parse quotes cell as JSON: %s", value[:200])
      return []
  else:
    return []
  if not isinstance(items, list):
    logging.error("Quotes cell is not a JSON list: %s", str(value)[:200])
    return []
  quotes = []
  for item in items:
    if not isinstance(item, str):
      continue
    text = item.strip()
    if text and text.lower() != "nan":
      quotes.append(text)
  return quotes


def select_quotes(
    quote_ids: object, quotes: list[str], proposal: str
) -> list[str]:
  """Maps the quote numbers the report model picked back to verbatim quotes.

  Args:
    quote_ids: The model's `participant_quote_ids`: 1-based positions in
      `quotes`.
    quotes: The quotes shown to the model, in the order they were numbered.
    proposal: The proposal the report is about, used in log lines.

  Returns:
    The picked quotes, verbatim and in the order picked, without duplicates and
    capped at `insights_prompts.MAX_PARTICIPANT_QUOTES`.
  """
  if quote_ids is None:
    return []
  if not isinstance(quote_ids, list):
    logging.error(
        "participant_quote_ids for proposal %r is not a list: %r",
        proposal[:80],
        quote_ids,
    )
    return []
  selected = []
  seen = set()
  for quote_id in quote_ids:
    is_number = isinstance(quote_id, int) and not isinstance(quote_id, bool)
    if not is_number or not 1 <= quote_id <= len(quotes):
      logging.error(
          "Report for proposal %r picked quote %r, which is not one of the %d"
          " numbered quotes.",
          proposal[:80],
          quote_id,
          len(quotes),
      )
      continue
    if quote_id in seen:
      continue
    seen.add(quote_id)
    selected.append(quotes[quote_id - 1])
  if len(selected) > insights_prompts.MAX_PARTICIPANT_QUOTES:
    logging.warning(
        "Report for proposal %r picked %d quotes; keeping the first %d.",
        proposal[:80],
        len(selected),
        insights_prompts.MAX_PARTICIPANT_QUOTES,
    )
    selected = selected[: insights_prompts.MAX_PARTICIPANT_QUOTES]
  return selected


def _extract_report_block(
    raw_block: object,
    default_categories: list[str],
    block_name: str,
    proposal: str,
) -> dict | None:
  """Validates and normalizes one structured report block (`tldr` + `insights`).

  Args:
    raw_block: Parsed JSON value for `problem_definition` or `recommendation`.
    default_categories: Default category order from `REPORT_FIELDS` used when
      an item omits `category`.
    block_name: Block name for logging.
    proposal: Policy proposal text for logging.

  Returns:
    Normalized `{"tldr": str, "insights": [{"category", "text_with_citations"}]}`
    or None if required content is missing.
  """
  if not isinstance(raw_block, dict):
    return None
  tldr = str(raw_block.get("tldr", "")).strip()
  raw_items = raw_block.get("insights", [])
  if not isinstance(raw_items, list):
    return None
  normalized_items = []
  for idx in range(len(raw_items)):
    item = raw_items[idx]
    default_cat = (
        default_categories[idx]
        if idx < len(default_categories)
        else (default_categories[-1] if default_categories else "")
    )
    if not isinstance(item, dict):
      continue
    cat = str(item.get("category", "")).strip() or default_cat
    text = str(item.get("text_with_citations", "")).strip()
    if text.startswith("- ") or text.startswith("* "):
      text = text[2:].strip()
    if not text:
      continue
    _warn_on_unresolved_links(
        f"{block_name}.insights[{idx}] ({cat}) for {proposal[:60]!r}", text
    )
    normalized_items.append({
        "category": cat,
        "text_with_citations": text,
    })
  if not tldr or not normalized_items:
    return None
  _warn_on_unresolved_links(f"{block_name}.tldr for {proposal[:60]!r}", tldr)
  return {
      "tldr": tldr,
      "insights": normalized_items,
  }


async def generate_policy_reports(
    insights_df: pd.DataFrame,
    model: genai_model.GenaiModel,
    place_name: str = "",
    max_concurrent_calls: int | None = None,
    checkpoint_file: str | None = None,
    quotes_by_proposal: dict[tuple[str, str], list[str]] | None = None,
) -> pd.DataFrame:
  """Composes each proposal's insights into a structured policy report in one step.

  Args:
    insights_df: One row per insight, carrying `topic`, `proposal`, `category`
      and `full_insight`.
    model: The Gemini wrapper used to generate content.
    place_name: Target municipality, used to keep the writing locally framed.
    max_concurrent_calls: Ceiling on in-flight calls.
    checkpoint_file: Optional JSONL file to resume from and append to.
    quotes_by_proposal: Optional participant quotes keyed on `(topic,
      proposal)`. The report picks the ones it shows from these.

  Returns:
    One row per proposal with `topic`, `proposal`, and `report_json`. Empty
    frame when there is nothing to compose.
  """
  if insights_df.empty:
    return pd.DataFrame(columns=REPORTS_DF_COLUMNS)
  if quotes_by_proposal is None:
    quotes_by_proposal = {}

  grouped = collections.defaultdict(list)
  for _, row in insights_df.iterrows():
    text = str(row.get("full_insight", "")).strip()
    if not text:
      continue
    key = (str(row.get("topic", "")), str(row.get("proposal", "")))
    grouped[key].append({
        "category": str(row.get("category", "")),
        "full_insight": text,
    })

  prompt_objs = []
  for (topic, proposal), insights in grouped.items():
    prompt_objs.append({
        "prompt": insights_prompts.get_policy_report_prompt(
            proposal=proposal,
            insights=insights,
            place_name=place_name,
            topic=topic,
            quotes=quotes_by_proposal.get((topic, proposal), []),
        ),
        "topic": topic,
        "proposal": proposal,
        "response_schema": insights_prompts.PolicyReport.model_json_schema(),
    })

  if not prompt_objs:
    return pd.DataFrame(columns=REPORTS_DF_COLUMNS)

  cached_map = await _run_prompts_with_checkpoint(
      prompt_objs,
      model,
      run_name="policy_report",
      key_fields=["topic", "proposal"],
      desc="Composing policy reports",
      max_concurrent_calls=max_concurrent_calls,
      checkpoint_file=checkpoint_file,
  )

  problem_cats = []
  for pt in insights_prompts.REPORT_FIELDS["problem_statement"]["points"]:
    problem_cats.append(pt.category)
  rec_cats = []
  for pt in insights_prompts.REPORT_FIELDS["recommendation_path"]["points"]:
    rec_cats.append(pt.category)

  rows = []
  failed_count = 0
  for obj in prompt_objs:
    topic = obj["topic"]
    proposal = obj["proposal"]
    raw_result = cached_map.get((topic, proposal), "")
    try:
      parsed = json.loads(raw_result)
    except Exception as e:
      failed_count += 1
      logging.error(
          "Failed to parse policy report JSON for proposal %r: %s. Raw: %s",
          proposal[:80],
          e,
          str(raw_result)[:200],
      )
      continue
    if not isinstance(parsed, dict):
      failed_count += 1
      logging.error(
          "Policy report JSON for proposal %r is not a dict: %r",
          proposal[:80],
          parsed,
      )
      continue

    problem_block = _extract_report_block(
        parsed.get("problem_definition"),
        problem_cats,
        "problem_definition",
        proposal,
    )
    recommendation_block = _extract_report_block(
        parsed.get("recommendation"),
        rec_cats,
        "recommendation",
        proposal,
    )
    if not problem_block or not recommendation_block:
      failed_count += 1
      logging.error(
          "Policy report for proposal %r is missing a required block"
          " (problem_definition valid: %s, recommendation valid: %s).",
          proposal[:80],
          bool(problem_block),
          bool(recommendation_block),
      )
      continue

    selected_quotes = select_quotes(
        parsed.get("participant_quote_ids", []),
        quotes_by_proposal.get((topic, proposal), []),
        proposal,
    )
    report_obj = {
        "policy": proposal,
        "problem_definition": problem_block,
        "recommendation": recommendation_block,
        "quotes": selected_quotes,
        "references": [],
    }
    rows.append({
        "topic": topic,
        "proposal": proposal,
        "report_json": json.dumps(report_obj, ensure_ascii=False),
    })

  if failed_count:
    logging.error("Dropped %d policy report(s) that could not be composed.", failed_count)
  if not rows:
    return pd.DataFrame(columns=REPORTS_DF_COLUMNS)
  return pd.DataFrame(rows, columns=REPORTS_DF_COLUMNS)


# Citation links the report model writes, plus the two backtick variants it
# sometimes wraps them in: `[text](url)` around the whole link, and [`text`]
# around the label only. Plain-text source descriptions resolved from
# config.json (which may contain spaces) are also matched inside `(...)`.
_WRAPPED_LINK_RE = re.compile(r"`(\[[^\]\n]+\]\([^)\n]+\))`")
_WRAPPED_LABEL_RE = re.compile(r"\[`([^`\]\n]+)`\](?=\()")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]\n]+)\]\(([^)\n]+)\)")
# A sentence ends at terminal punctuation (optionally followed by closing
# quotes or brackets) plus whitespace, or at a line break. Over-splitting, e.g.
# after "U.S.", only costs an extra footnote marker, never a lost one.
_SENTENCE_BREAK_RE = re.compile(r"[.!?][\"'\u201d\u2019)\]]*\s+|\n")
_DRIVE_FILE_ID_RE = re.compile(r"drive\.google\.com/file/d/([A-Za-z0-9_-]+)")
_DRIVE_OPEN_ID_RE = re.compile(r"drive\.google\.com/open\?(?:.*&)?id=([A-Za-z0-9_-]+)")


def _source_key(url: str) -> str:
  """Returns the identity of a cited source, so variants of one URL collapse.

  Drive links are keyed on the file ID, which ignores `/view`, `/edit` and
  `?usp=drivesdk` differences. Other URLs drop their fragment, the `usp`
  tracking parameter and any trailing slash, and compare the scheme and host
  case-insensitively. Plain-text source descriptions (without a URL scheme)
  normalize whitespace and case.

  Args:
    url: A cited URL or plain-text source description.

  Returns:
    A string that is equal for citations pointing at the same source.
  """
  for pattern in (_DRIVE_FILE_ID_RE, _DRIVE_OPEN_ID_RE):
    match = pattern.search(url)
    if match:
      return f"drive:{match.group(1)}"
  parsed = urllib.parse.urlsplit(url.strip())
  if not parsed.scheme:
    return " ".join(url.split()).lower()
  kept_params = []
  for name, value in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True):
    if name != "usp":
      kept_params.append((name, value))
  path = parsed.path.rstrip("/")
  return urllib.parse.urlunsplit((
      parsed.scheme.lower(),
      parsed.netloc.lower(),
      path,
      urllib.parse.urlencode(kept_params),
      "",
  ))


def _source_title(url: str) -> str:
  """Returns a human-readable name for a cited source.

  Plain-text source descriptions (without an HTTP/HTTPS scheme) are returned as
  written. Registered datasets with a URL are named by their registry filename.
  Other URLs are named by host and, when present, the last path segment (e.g. a
  PDF name).

  Args:
    url: A cited URL or plain-text source description.

  Returns:
    A short display title.
  """
  parsed = urllib.parse.urlsplit(url.strip())
  if not parsed.scheme:
    return url.strip()
  key = _source_key(url)
  registry = getattr(insights_prompts, "LOCAL_DATASETS_REGISTRY", {})
  for name, info in registry.items():
    source_url = info.get("source_url", "")
    if source_url and _source_key(source_url) == key:
      return name
  host = parsed.netloc.lower()
  if host.startswith("www."):
    host = host[4:]
  last_segment = urllib.parse.unquote(parsed.path.rstrip("/").split("/")[-1])
  if last_segment and host:
    return f"{host} \u2014 {last_segment}"
  return host or url


def _normalize_citation_links(text: str) -> str:
  """Removes the backtick wrappers the report model sometimes adds to links."""
  text = _WRAPPED_LINK_RE.sub(r"\1", text)
  return _WRAPPED_LABEL_RE.sub(r"[\1]", text)


def _sentence_index_by_offset(text: str, links: list[re.Match]) -> list[int]:
  """Returns, for each link, the index of the sentence it sits in.

  Sentence breaks are searched with the links masked out, so punctuation
  inside a link label or URL never splits a sentence.

  Args:
    text: The text the links were found in.
    links: Link matches in `text`, in order.

  Returns:
    One sentence index per link.
  """
  masked = list(text)
  for link in links:
    for pos in range(link.start(), link.end()):
      masked[pos] = "x"
  break_offsets = []
  for match in _SENTENCE_BREAK_RE.finditer("".join(masked)):
    break_offsets.append(match.start())
  indices = []
  for link in links:
    sentence_index = 0
    for offset in break_offsets:
      if offset < link.start():
        sentence_index += 1
    indices.append(sentence_index)
  return indices


def number_citations(texts: list[str]) -> tuple[list[str], list[dict]]:
  """Replaces inline citation links with numbered footnote markers.

  Each distinct source gets one number, assigned in reading order across all
  the texts. Within one sentence a source is marked once, after the last
  phrase that cites it; the other phrases citing it in that sentence keep
  their words but lose the link. Links to in-page anchors (`#...`) are left
  untouched. Text that has no inline links, including text already processed
  by this function, comes back unchanged.

  Args:
    texts: The texts of one report, in the order a reader sees them.

  Returns:
    A tuple of the rewritten texts, in the input order, and the reference list
    `[{"id": int, "title": str, "url": str}, ...]` ordered by id.
  """
  number_by_key = {}
  references = []
  rewritten_texts = []
  for raw_text in texts:
    text = _normalize_citation_links(raw_text or "")
    links = []
    for link in _MARKDOWN_LINK_RE.finditer(text):
      if not link.group(2).startswith("#"):
        links.append(link)
    if not links:
      rewritten_texts.append(text)
      continue

    sentence_indices = _sentence_index_by_offset(text, links)
    last_link_for_source_in_sentence = {}
    for position in range(len(links)):
      group_key = (sentence_indices[position], _source_key(links[position].group(2)))
      last_link_for_source_in_sentence[group_key] = position

    pieces = []
    cursor = 0
    for position in range(len(links)):
      link = links[position]
      label = link.group(1)
      url = link.group(2)
      source = _source_key(url)
      pieces.append(text[cursor:link.start()])
      pieces.append(label)
      group_key = (sentence_indices[position], source)
      if last_link_for_source_in_sentence[group_key] == position:
        if source not in number_by_key:
          number_by_key[source] = len(references) + 1
          references.append({
              "id": number_by_key[source],
              "title": _source_title(url),
              "url": url,
          })
        pieces.append(f"[^{number_by_key[source]}]")
      cursor = link.end()
    pieces.append(text[cursor:])
    rewritten_texts.append("".join(pieces))
  return rewritten_texts, references


def _apply_footnotes_to_report_obj(report_obj: dict) -> dict:
  """Rewrites inline citation links inside a structured report dict into footnotes."""
  updated = json.loads(json.dumps(report_obj))
  texts = []
  slots = []

  for block_key in ("problem_definition", "recommendation"):
    block = updated.get(block_key)
    if not isinstance(block, dict):
      continue
    tldr_val = block.get("tldr", "")
    texts.append(tldr_val if isinstance(tldr_val, str) else "")
    slots.append((block_key, "tldr", -1))
    insights_list = block.get("insights", [])
    if isinstance(insights_list, list):
      for idx in range(len(insights_list)):
        item = insights_list[idx]
        if isinstance(item, dict):
          txt = item.get("text_with_citations", "")
          texts.append(txt if isinstance(txt, str) else "")
          slots.append((block_key, "insight", idx))

  rewritten, references = number_citations(texts)
  for pos in range(len(slots)):
    block_key, kind, idx = slots[pos]
    if kind == "tldr":
      updated[block_key]["tldr"] = rewritten[pos]
    else:
      updated[block_key]["insights"][idx]["text_with_citations"] = rewritten[pos]

  existing_refs = updated.get("references", [])
  if not references and isinstance(existing_refs, list) and existing_refs:
    updated["references"] = existing_refs
  else:
    updated["references"] = references
  return updated


def apply_citation_footnotes(reports_df: pd.DataFrame) -> pd.DataFrame:
  """Rewrites every report's inline citations as per-report numbered footnotes.

  Numbering restarts at 1 for each report and follows the display order of the
  report (`problem_definition.tldr` -> `problem_definition.insights` ->
  `recommendation.tldr` -> `recommendation.insights`). When `report_json` is
  present, the footnoted texts and deduplicated `references` list are written
  directly into `report_json`.

  Args:
    reports_df: Reports frame carrying `report_json`.

  Returns:
    A copy of the frame with footnoted report content.
  """
  reports_df = reports_df.copy()
  if reports_df.empty:
    return reports_df

  new_json_values = []
  for _, row in reports_df.iterrows():
    raw_val = row.get("report_json", "")
    if isinstance(raw_val, dict):
      report_obj = raw_val
    elif isinstance(raw_val, str) and raw_val.strip():
      report_obj = json.loads(raw_val)
    else:
      new_json_values.append(raw_val)
      continue
    if not isinstance(report_obj, dict):
      new_json_values.append(raw_val)
      continue
    updated_obj = _apply_footnotes_to_report_obj(report_obj)
    new_json_values.append(json.dumps(updated_obj, ensure_ascii=False))
  reports_df["report_json"] = new_json_values
  return reports_df


FACT_CHECK_FAIL = "FAIL"


def filter_fact_checked_insights(
    insights_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, int]]]:
  """Drops insights that failed fact checking and counts statuses per proposal.

  PASS and SKIP insights are kept: SKIP means the check could not run, not
  that the insight is wrong. Rows without a `verification_status` (fact check
  not run) are kept and counted as unchecked.

  Args:
    insights_df: Insights frame, optionally carrying `verification_status`.

  Returns:
    A tuple of (kept insights frame, counts keyed on `(topic, proposal)`).
    Each counts dict has `passed`, `skipped`, `failed_dropped` and
    `unchecked` keys.
  """
  has_status = "verification_status" in insights_df.columns
  counts = {}
  keep_mask = []
  for _, row in insights_df.iterrows():
    key = (str(row.get("topic", "")), str(row.get("proposal", "")))
    if key not in counts:
      counts[key] = {
          "passed": 0,
          "skipped": 0,
          "failed_dropped": 0,
          "unchecked": 0,
      }
    status = ""
    if has_status and pd.notna(row.get("verification_status")):
      status = str(row.get("verification_status")).strip().upper()
    if status == FACT_CHECK_FAIL:
      counts[key]["failed_dropped"] += 1
      keep_mask.append(False)
      continue
    if status == "PASS":
      counts[key]["passed"] += 1
    elif status == "SKIP":
      counts[key]["skipped"] += 1
    else:
      counts[key]["unchecked"] += 1
    keep_mask.append(True)

  kept_df = insights_df[keep_mask].reset_index(drop=True)
  return kept_df, counts


def attach_fact_check_counts(
    reports_df: pd.DataFrame,
    counts: dict[tuple[str, str], dict[str, int]],
) -> pd.DataFrame:
  """Adds each proposal's fact check counts to its `report_json` object.

  Args:
    reports_df: Reports frame with `topic`, `proposal` and `report_json`.
    counts: Fact check counts keyed on `(topic, proposal)`.

  Returns:
    A copy of the frame whose `report_json` objects carry a `fact_check` key.
  """
  reports_df = reports_df.copy()
  if reports_df.empty or "report_json" not in reports_df.columns:
    return reports_df
  new_values = []
  for _, row in reports_df.iterrows():
    raw_val = row.get("report_json", "")
    try:
      report_obj = json.loads(raw_val) if isinstance(raw_val, str) else raw_val
    except Exception:
      new_values.append(raw_val)
      continue
    if not isinstance(report_obj, dict):
      new_values.append(raw_val)
      continue
    key = (str(row.get("topic", "")), str(row.get("proposal", "")))
    report_obj["fact_check"] = counts.get(
        key,
        {"passed": 0, "skipped": 0, "failed_dropped": 0, "unchecked": 0},
    )
    new_values.append(json.dumps(report_obj, ensure_ascii=False))
  reports_df["report_json"] = new_values
  return reports_df


def load_research_inputs(
    input_csvs: list[tuple[str, str]],
) -> pd.DataFrame:
  """Loads the Phase 2 research CSVs and merges them into one row per proposal.

  Args:
    input_csvs: `(path, source)` tuples where `source` is one of
      `RESEARCH_SOURCES`.

  Returns:
    One row per `(topic, proposal)` with combined `research_output`,
    `research_ideas_prompt` and `quotes` columns.

  Raises:
    ValueError: If a required column is missing or an explicit source is not a
      known research source.
  """
  required_cols = [
      "topic",
      "proposal",
      "research_output",
      "research_ideas_prompt",
  ]
  dfs = []
  for item in input_csvs:
    if not isinstance(item, tuple) or len(item) != 2:
      raise ValueError(
          f"Expected a (path, source) tuple for research input, got {item!r}."
      )
    path, explicit_source = item
    if explicit_source not in RESEARCH_SOURCES:
      raise ValueError(
          f"Unknown research source {explicit_source!r} for {path}; expected"
          f" one of {list(RESEARCH_SOURCES)}."
      )

    temp_df = pd.read_csv(path)
    for col in required_cols:
      if col not in temp_df.columns:
        raise ValueError(f"Required column '{col}' missing in input file {path}")

    temp_df["source_origin"] = explicit_source

    # Participant quotes are optional: phase 1 carries them through as a JSON
    # list, but inputs without resident quotes have none.
    if "quotes" not in temp_df.columns:
      temp_df["quotes"] = "[]"
    dfs.append(temp_df[required_cols + ["quotes", "source_origin"]])

  df = pd.concat(dfs, ignore_index=True)

  # Helper function to format combined research output with clear source headers
  def format_combined_research(group):
    parts = []
    for _, row in group.iterrows():
      source = row["source_origin"]
      raw_content = row.get("research_output", "")
      if pd.isna(raw_content) or not str(raw_content).strip():
        continue
      content = str(raw_content).strip()
      if source == "data_commons":
        header = "### Data Commons Research Output"
      elif source == "deep_research":
        header = "### Deep Research Output"
      elif source == "file_agent":
        header = "### File Agent Research Output"
      else:
        raise ValueError(f"Unexpected research source: {source!r}")
      parts.append(f"{header}:\n{content}")
    return "\n\n".join(parts)

  def first_quotes(group):
    # Every research source copies the same phase 1 quotes, so the first
    # non-empty list is the proposal's list.
    for value in group["quotes"]:
      quotes = parse_quotes(value)
      if quotes:
        return json.dumps(quotes, ensure_ascii=False)
    return "[]"

  def format_group_row(group):
    return pd.Series({
        "research_output": format_combined_research(group),
        "research_ideas_prompt": group["research_ideas_prompt"].iloc[0],
        "quotes": first_quotes(group),
    })

  df = (
      df.groupby(["topic", "proposal"])
      .apply(format_group_row, include_groups=False)
      .reset_index()
  )
  return df


def build_quotes_by_proposal(
    research_df: pd.DataFrame,
) -> dict[tuple[str, str], list[str]]:
  """Maps each `(topic, proposal)` to its participant quotes.

  Args:
    research_df: Output of `load_research_inputs`.

  Returns:
    Parsed quote lists keyed on `(topic, proposal)`.
  """
  quotes_by_proposal = {}
  for _, row in research_df.iterrows():
    key = (str(row["topic"]), str(row["proposal"]))
    quotes_by_proposal[key] = parse_quotes(row["quotes"])
  return quotes_by_proposal


async def run_insights_stage(
    args, model: genai_model.GenaiModel, research_df: pd.DataFrame
) -> pd.DataFrame:
  """Generates insights from the research and writes them to `output_csv`.

  Args:
    args: Parsed command-line arguments.
    model: The Gemini wrapper used to generate content.
    research_df: Output of `load_research_inputs`.

  Returns:
    The insights frame that was written.
  """
  stage1_checkpoint = f"{args.output_csv}.stage1_checkpoint.jsonl"
  research_df = research_df.copy()
  research_df["insights"] = await generate_structured_insights(
      research_df,
      model,
      place_name=args.place_name,
      max_concurrent_calls=args.concurrency,
      checkpoint_file=stage1_checkpoint,
  )
  insights_df = create_insights_dataframe(research_df)
  insights_df.to_csv(args.output_csv, index=False)
  print(f"Wrote {len(insights_df)} insights to {args.output_csv}")
  return insights_df


async def run_reports_stage(
    args,
    model: genai_model.GenaiModel,
    research_df: pd.DataFrame,
    insights_df: pd.DataFrame,
) -> pd.DataFrame:
  """Composes policy reports from insights that did not fail fact checking.

  Args:
    args: Parsed command-line arguments.
    model: The Gemini wrapper used to generate content.
    research_df: Output of `load_research_inputs`, used for quotes.
    insights_df: Insights frame, optionally carrying `verification_status`.

  Returns:
    The reports frame that was written.
  """
  report_checkpoint = f"{args.output_csv}.report_v2_checkpoint.jsonl"
  kept_df, fact_check_counts = filter_fact_checked_insights(insights_df)
  for (_, proposal), counts in fact_check_counts.items():
    print(
        f"Fact check for {proposal[:60]!r}: passed={counts['passed']}"
        f" skipped={counts['skipped']}"
        f" failed_dropped={counts['failed_dropped']}"
        f" unchecked={counts['unchecked']}"
    )

  print("Composing structured policy reports from insights...")
  reports_df = await generate_policy_reports(
      kept_df,
      model,
      place_name=args.place_name,
      max_concurrent_calls=args.concurrency,
      checkpoint_file=report_checkpoint,
      quotes_by_proposal=build_quotes_by_proposal(research_df),
  )

  reports_df = apply_citation_footnotes(reports_df)
  reports_df = attach_fact_check_counts(reports_df, fact_check_counts)

  reports_csv = args.reports_csv
  if not reports_csv:
    reports_csv = f"{os.path.splitext(args.output_csv)[0]}_reports.csv"
  reports_df.to_csv(reports_csv, index=False)
  print(f"Wrote {len(reports_df)} policy reports to {reports_csv}")
  return reports_df


def _split_csv_paths(values: list[str] | None) -> list[str]:
  """Flattens repeated and comma-separated path flags into a list of paths.

  Args:
    values: Raw argparse values, one entry per flag occurrence, or None.

  Returns:
    The non-empty paths in command-line order.
  """
  paths = []
  if not values:
    return paths
  for value in values:
    for part in value.split(","):
      part = part.strip()
      if part:
        paths.append(part)
  return paths


def collect_research_inputs(args) -> list[tuple[str, str]]:
  """Pairs every research CSV on the command line with its declared source.

  Args:
    args: Parsed command-line arguments.

  Returns:
    `(path, source)` tuples for `load_research_inputs`.

  Raises:
    ValueError: If no research CSV was given.
  """
  inputs = []
  for path in _split_csv_paths(args.deep_research_csv):
    inputs.append((path, "deep_research"))
  for path in _split_csv_paths(args.data_commons_csv):
    inputs.append((path, "data_commons"))
  for path in _split_csv_paths(args.file_agent_csv):
    inputs.append((path, "file_agent"))
  if not inputs:
    raise ValueError(
        "No research CSV given. Pass --deep_research_csv, --data_commons_csv,"
        " or --file_agent_csv."
    )
  return inputs


async def main(args):
  model = genai_model.GenaiModel(
      model_name=args.model_name,
      gemini_api_key=args.gemini_api_key,
  )
  research_df = load_research_inputs(collect_research_inputs(args))

  if args.stage in ("all", "insights"):
    insights_df = await run_insights_stage(args, model, research_df)
  else:
    insights_df = pd.read_csv(args.output_csv)
    print(f"Loaded {len(insights_df)} insights from {args.output_csv}")

  if args.stage in ("all", "reports"):
    await run_reports_stage(args, model, research_df, insights_df)


def get_args(argv: list[str] | None = None):
  """Parses command-line arguments.

  Args:
    argv: Arguments to parse; defaults to `sys.argv[1:]`.

  Returns:
    The parsed arguments.
  """
  parser = argparse.ArgumentParser(
      description="Generates structured insights and policy reports from Phase 2 research CSVs."
  )
  parser.add_argument(
      "--deep_research_csv",
      action="append",
      default=[],
      help=(
          "Deep Research CSV(s). Repeat the flag or separate paths with"
          " commas."
      ),
  )
  parser.add_argument(
      "--data_commons_csv",
      action="append",
      default=[],
      help=(
          "Data Commons CSV(s). Repeat the flag or separate paths with commas."
      ),
  )
  parser.add_argument(
      "--file_agent_csv",
      action="append",
      default=[],
      help="File Agent CSV(s). Repeat the flag or separate paths with commas.",
  )
  parser.add_argument(
      "--output_csv", required=True, help="Path to save the output CSV file."
  )
  parser.add_argument(
      "--reports_csv",
      type=str,
      default="",
      help=(
          "Path to save the composed policy reports. Defaults to the"
          " --output_csv path with a '_reports' suffix."
      ),
  )
  parser.add_argument(
      "--place_name",
      type=str,
      default="",
      help="Target municipality or region name.",
  )
  parser.add_argument(
      "--stage",
      choices=["all", "insights", "reports"],
      default="all",
      help=(
          "Which part to run. 'insights' writes --output_csv only. 'reports'"
          " reads --output_csv (after fact checking), drops FAIL insights and"
          " writes --reports_csv. 'all' runs both without fact checking."
      ),
  )
  parser.add_argument(
      "--gemini_api_key",
      required=False,
      default=os.environ.get("GEMINI_API_KEY", ""),
      help="Google AI Studio API Key.",
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
      default=10,
      help="Maximum number of concurrent LLM API calls. Default: 10.",
  )
  return parser.parse_args(argv)


if __name__ == "__main__":
  asyncio.run(main(get_args()))
