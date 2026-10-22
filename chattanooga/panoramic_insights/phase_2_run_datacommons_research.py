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
Runs quantitative civic research using the Data Commons MCP (Model Context Protocol) server.

This script implements an autonomous Research Agent in Phase 2 of the insights pipeline.
It acts as the structured, quantitative counterpart to the qualitative Deep Research agent.
The agent uses Gemini function-calling bound to Python-wrapped Data Commons MCP tools
to compile highly grounded, verified statistical dossiers for a target location.

Core Design Patterns & Concepts:
1. **2-Step Drill-Down Protocol**:
   - Step 1 (Discovery): The agent calls `search_indicators` with human-readable place names
     (e.g., ["<place_name>"]) to find relevant statistical variable DCIDs available for that place.
   - Step 2 (Retrieval): The agent calls `get_observations` with the exact target place DCID
     (e.g., "<place_dcid>") and the discovered variable DCID to retrieve the actual values.
2. **JSON-RPC over HTTP**:
   - The Data Commons MCP endpoint (`https://api.datacommons.org/mcp`) answers each
     `tools/call` request with a single JSON-RPC 2.0 JSON body. Tool output is read
     from `result.structuredContent` (or the text in `result.content`), and errors
     from `error` / `result.isError`.
3. **Deterministic Grounding**:
   - Unlike qualitative web searches, this agent queries a structured database. It does not
     "hallucinate" statistics; it either retrieves the exact census/economic figures or reports them as missing.
   - If no tool call succeeds for a row, the row's dossier is left empty instead of
     keeping an answer the model wrote without any Data Commons data.

Example usage:
python3 -m panoramic_insights.phase_2_run_datacommons_research \
  --input_csv <input_csv> \
  --output_csv <output_csv> \
  --place_name "<place_name>" \
  --place_dcid "<place_dcid>"
"""

import argparse
import asyncio
import json
import logging
import os
import traceback
from google import genai
from google.genai import types
import pandas as pd
import requests
from tqdm import tqdm

# Load Data Commons API Key strictly from the DC_API_KEY environment variable.
# This dedicated key is required to authenticate requests to the Data Commons MCP endpoint.
DC_API_KEY = os.environ.get("DC_API_KEY", "")


MCP_URL = "https://api.datacommons.org/mcp"
MCP_TIMEOUT_SECONDS = 120
MAX_AGENT_ITERATIONS = 12
# Upper bound on indicators the agent is asked to retrieve per proposal.
MAX_INDICATORS = 8
# Sent with the last tool results once the round budget is spent.
FINAL_ANSWER_REQUEST = (
    "The tool budget is used up. Do not call any more tools. Write the final"
    " markdown dossier now, using only the Data Commons tool results above."
    " For metrics you did not retrieve, write that they were not retrieved."
)


def build_user_prompt(row: pd.Series) -> str:
  """Builds the Data Commons agent's request from one input row.

  The agent gets the proposal itself, not the Phase 1 research ideas: those
  cover costs, laws, local files, and peer cities, none of which Data Commons
  holds, and chasing them exhausts the tool budget.

  Args:
    row: An input row with `topic` and `proposal` columns.

  Returns:
    The user message for the agent.

  Raises:
    ValueError: If `proposal` is missing or empty.
  """
  proposal = row.get("proposal")
  topic = row.get("topic")
  if pd.isna(proposal) or not str(proposal).strip():
    raise ValueError("Row is missing a non-empty 'proposal' value.")
  lines = []
  if pd.notna(topic) and str(topic).strip():
    lines.append(f'Topic: "{topic}"')
  lines.append(f'Citizen proposal: "{proposal}"')
  lines.append(
      "Find Data Commons statistics that size the problem this proposal"
      " addresses and describe the residents it would affect."
  )
  return "\n".join(lines)


# Retry policy for Gemini calls: transient 429/5xx (e.g., 503 "model is
# experiencing high demand") are retried with exponential backoff instead of
# failing the row on the first burst.
GEMINI_RETRY_OPTIONS = types.HttpRetryOptions(
    attempts=8,
    initial_delay=5.0,
    max_delay=120.0,
    exp_base=2.0,
    http_status_codes=[429, 500, 502, 503, 504],
)


def _is_tool_success(result_json: str) -> bool:
  """Returns True if a serialized tool result is not an error payload.

  Args:
    result_json: JSON string returned by `search_indicators` or
      `get_observations`.
  """
  try:
    parsed = json.loads(result_json)
  except (TypeError, json.JSONDecodeError):
    return False
  if isinstance(parsed, dict) and "error" in parsed:
    return False
  return True


def connect_and_run(tool_name, tool_args):
  """Calls one Data Commons MCP tool and returns its parsed output.

  Sends a JSON-RPC 2.0 `tools/call` request and parses the JSON response body.

  Args:
    tool_name: MCP tool name (e.g., 'search_indicators').
    tool_args: Dict of tool arguments.

  Returns:
    The tool output (`result.structuredContent`, or the parsed text content),
    or a dict with an "error" key if the call failed.
  """
  if not DC_API_KEY:
    return {"error": "DC_API_KEY environment variable is required but not set."}

  rpc_payload = {
      "jsonrpc": "2.0",
      "id": 1,
      "method": "tools/call",
      "params": {"name": tool_name, "arguments": tool_args},
  }
  headers = {
      "X-API-Key": DC_API_KEY,
      "Content-Type": "application/json",
      "Accept": "application/json",
  }
  try:
    resp = requests.post(
        MCP_URL, headers=headers, json=rpc_payload, timeout=MCP_TIMEOUT_SECONDS
    )
  except requests.RequestException as e:
    return {"error": f"Data Commons MCP request failed: {e}"}
  if resp.status_code != 200:
    return {
        "error": f"Data Commons MCP HTTP {resp.status_code}: {resp.text[:500]}"
    }
  try:
    body = resp.json()
  except ValueError:
    return {"error": f"Data Commons MCP returned non-JSON: {resp.text[:500]}"}

  if body.get("error"):
    error = body["error"]
    message = error.get("message") if isinstance(error, dict) else str(error)
    return {"error": f"Data Commons MCP error: {message}"}
  result = body.get("result") or {}
  if result.get("isError"):
    return {"error": f"Data Commons tool error: {result.get('content')}"}
  if result.get("structuredContent") is not None:
    return result["structuredContent"]
  content = result.get("content") or []
  if content and content[0].get("type") == "text":
    text = content[0].get("text", "")
    try:
      return json.loads(text)
    except json.JSONDecodeError:
      return text
  return {"error": f"Data Commons MCP returned no content: {body}"}


def search_indicators(query: str, places: list[str] = None) -> str:
  """Search Data Commons for statistical variables (indicators) matching a query.

  Args:
      query: The search term (e.g., 'unemployment', 'transportation').
      places: Optional list of fully qualified place names (e.g., ['City, State']) to check availability.
  """
  args = {"query": query}
  if places:
    args["places"] = places
  res = connect_and_run("search_indicators", args)
  return json.dumps(res)


def get_observations(
    variable_dcid: str,
    place_dcid: str,
    date: str = None,
    date_range_start: str = None,
    date_range_end: str = None,
) -> str:
  """Fetch observation data for a specific variable and place from Data Commons.

  Args:
      variable_dcid: The unique DCID of the variable (e.g., 'Count_Person').
      place_dcid: The unique DCID of the place (e.g., 'geoId/4714000').
      date: Specific date or 'latest' or 'range' or 'all'.
      date_range_start: Start year (e.g., '2010') if date='range'.
      date_range_end: End year (e.g., '2020') if date='range'.
  """
  args = {"variable_dcid": variable_dcid, "place_dcid": place_dcid}
  if date:
    args["date"] = date
  if date_range_start:
    args["date_range_start"] = date_range_start
  if date_range_end:
    args["date_range_end"] = date_range_end
  res = connect_and_run("get_observations", args)
  return json.dumps(res)




async def run_datacommons_agent(
    prompt: str,
    client: genai.Client,
    model_id: str,
    place_name: str,
    place_dcid: str,
) -> str:
  """Runs a single agentic research loop for a prompt, grounded in Data Commons.

  This function implements a classic **ReAct (Reasoning and Acting) loop** using
  Gemini's native function-calling capabilities.

  Key Features of the Loop:
  1. **System Instruction Steering**: The agent is heavily steered to follow the
     2-step drill-down protocol (Search -> Fetch) and to focus strictly on the
     target place.
  2. **Tool Binding**: Python functions (`search_indicators` and
  `get_observations`) are
     passed directly as tools. The SDK automatically generates JSON schemas from
     their docstrings.
  3. **State Management**: The conversational history (user prompts, model
  thoughts,
     tool calls, and tool responses) is accumulated in the `messages` list,
     preserving
     context across multiple function-calling rounds.
  4. **Loop Prevention**: A hard ceiling of 12 iterations is enforced. If the
  model
     gets stuck in a recursive tool-calling loop, it will gracefully terminate
     and return
     what it has gathered, preventing runaway API costs.
  5. **Dynamic Argument Binding**: If the model forgets to pass the required
  `place_dcid`
     when calling `get_observations`, the loop interceptor automatically binds
     the target
     `place_dcid` in-flight, ensuring robustness.
  """
  system_instruction = f"""You are an expert Civic Research Assistant specializing in quantitative data retrieval.
Your goal is to find Data Commons statistics that give quantitative context for the citizen proposal in the user's message.

The target geographic area of interest is: {place_name} (DCID: {place_dcid}).
Always prioritize finding and reporting data for this specific place.

Scope:
- Pick at most {MAX_INDICATORS} indicators that are relevant to the proposal and that Data Commons is likely to hold, such as population and age groups, income and poverty, commuting mode, vehicle availability, housing, school enrollment, or health.
- Do not try to research everything about the proposal. Costs, budgets, laws, local incident logs, and peer city programs are covered by other agents; skip them.
- You may request several tool calls in the same turn.

Methodology:
1. **Search**: Use the `search_indicators` tool to discover relevant statistical variables (indicator DCIDs).
   - When searching, pass the qualified place name `["{place_name}"]` in the `places` argument to ensure the variables are available for this place.
2. **Fetch**: Once you have identified relevant variable DCIDs (e.g., `Count_Person`, `Median_Age_Person`), call `get_observations` to fetch the actual values for the target place `{place_dcid}`.
   - Always request recent data (e.g., specify a date or date range if appropriate, or default to latest).
3. **Report**: Compile your findings into a short structured markdown report (dossier).
   - Report ONLY values that a `get_observations` call returned in this conversation. Do not add facts, figures, or claims from general knowledge, other sources, or local files.
   - Present the exact numbers, dates, variable DCIDs, and sources clearly.
   - For every retrieved metric, include the direct Data Commons Timelines Explorer deep-link URL:
     `https://datacommons.org/tools/timeline#place=<PLACE_DCID_ENCODED>&statsVar=<VARIABLE_DCID>`
     (e.g., `https://datacommons.org/tools/timeline#place=geoId%2F4714000&statsVar=Median_Income_Person`).
     When comparing multiple metrics, join variable DCIDs with double underscores `__` (e.g., `statsVar=VAR1__VAR2`).
   - If a metric you searched for is not available in Data Commons, state "Metric [Name] not found in Data Commons" rather than guessing.
   - Focus strictly on facts and quantitative metrics. Avoid speculative or flowery language.
"""
  # Register our python-wrapped MCP wrappers as tools. Automatic function
  # calling is disabled so this loop alone executes tools and can count which
  # calls actually returned Data Commons data.
  tools = [search_indicators, get_observations]
  config = types.GenerateContentConfig(
      system_instruction=system_instruction,
      tools=tools,
      temperature=0.1,  # Low temperature for deterministic, analytical output
      automatic_function_calling=types.AutomaticFunctionCallingConfig(
          disable=True
      ),
  )
  # Same as `config` but with no tools declared, so the model can only answer
  # in text.
  final_config = config.model_copy(
      update={"tools": None, "automatic_function_calling": None}
  )

  # Initialize the conversation history with the user's original research prompt
  messages = [
      types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
  ]

  # Initial inference call to kick off the reasoning phase
  response = await client.aio.models.generate_content(
      model=model_id, contents=messages, config=config
  )

  # Execute the ReAct loop
  iterations = 0
  successful_tool_calls = 0
  while response.function_calls and iterations < MAX_AGENT_ITERATIONS:
    # Append the model's thoughts and tool requests to history
    messages.append(response.candidates[0].content)
    tool_responses = []

    # Execute all function calls requested by the model
    for function_call in response.function_calls:
      name = function_call.name
      args = dict(function_call.args or {})
      call_id = function_call.id

      # Dynamic Interception: Ensure place_dcid is bound if the model omitted it
      if name == "get_observations" and not args.get("place_dcid"):
        args["place_dcid"] = place_dcid

      # Dispatch tool execution off the event loop (requests is blocking)
      if name == "search_indicators":
        result = await asyncio.to_thread(search_indicators, **args)
      elif name == "get_observations":
        result = await asyncio.to_thread(get_observations, **args)
      else:
        result = json.dumps({"error": f"Unknown function: {name}"})

      if _is_tool_success(result):
        successful_tool_calls += 1
      else:
        logging.warning("Data Commons tool %s failed: %s", name, result[:300])

      tool_responses.append(
          types.Part(
              function_response=types.FunctionResponse(
                  name=name, response={"result": result}, id=call_id
              )
          )
      )

    iterations += 1

    # On the last allowed round the model must write the dossier from what it
    # has; otherwise the loop could end on a bare function call with no text.
    # Tools are removed from the request entirely because gemini-3.6-flash
    # still emits function calls under FunctionCallingConfigMode.NONE.
    next_config = config
    if iterations >= MAX_AGENT_ITERATIONS:
      next_config = final_config
      tool_responses.append(types.Part.from_text(text=FINAL_ANSWER_REQUEST))

    # Function responses are sent back in a "user" turn (the Gemini API
    # rejects a "tool" role).
    messages.append(types.Content(role="user", parts=tool_responses))
    response = await client.aio.models.generate_content(
        model=model_id, contents=messages, config=next_config
    )

  logging.info(
      "Data Commons agent finished after %d tool rounds with %d successful"
      " tool calls.",
      iterations,
      successful_tool_calls,
  )
  if successful_tool_calls == 0:
    logging.warning(
        "Data Commons agent made no successful tool call; discarding the"
        " model's answer so it is not treated as Data Commons evidence."
    )
    return ""

  # Once no more functions are called, the final response contains the compiled
  # markdown dossier
  text = response.text or ""
  if not text.strip():
    logging.error(
        "Data Commons agent got %d successful tool results but its final"
        " answer has no text.",
        successful_tool_calls,
    )
  return text


async def _process_single_row(
    idx: int,
    prompt: str,
    client: genai.Client,
    model_id: str,
    place_name: str,
    place_dcid: str,
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
) -> tuple[int, str]:
  """Runs Data Commons research for a single row under semaphore concurrency.

  Returns an empty dossier if the research fails, so that downstream phases
  skip the row rather than treating an error message as evidence.
  """
  async with semaphore:
    try:
      dossier = await run_datacommons_agent(
          prompt=prompt,
          client=client,
          model_id=model_id,
          place_name=place_name,
          place_dcid=place_dcid,
      )
    except Exception:
      logging.error("Data Commons research failed for row %d:", idx)
      traceback.print_exc()
      dossier = ""
    pbar.update(1)
    return idx, dossier


async def main(args):
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
  )
  global DC_API_KEY
  if not DC_API_KEY:
    DC_API_KEY = os.environ.get("DC_API_KEY", "")
  if not DC_API_KEY:
    raise ValueError(
        "DC_API_KEY environment variable is required but not set. "
        "Please export DC_API_KEY with your Data Commons API key."
    )

  client = genai.Client(
      api_key=args.gemini_api_key,
      http_options=types.HttpOptions(retry_options=GEMINI_RETRY_OPTIONS),
  )

  df = pd.read_csv(args.input_csv)

  if os.path.exists(args.output_csv):
    try:
      existing_df = pd.read_csv(args.output_csv)
      if len(existing_df) == len(df):
        if "research_output" in existing_df.columns:
          df["research_output"] = existing_df["research_output"]
        if "interaction_id" in existing_df.columns:
          df["interaction_id"] = existing_df["interaction_id"]
    except Exception as e:
      logging.warning(f"Failed to read existing output checkpoint: {e}")

  already_done = df["research_output"].notna().sum() if "research_output" in df.columns else 0
  logging.info(
      f"Starting Data Commons research for {len(df)} rows on place"
      f" '{args.place_name}' with max concurrency {args.max_concurrency}"
      f" ({already_done}/{len(df)} already completed)..."
  )

  if "research_output" not in df.columns:
    df["research_output"] = None
  if "interaction_id" not in df.columns:
    df["interaction_id"] = "datacommons_agent"
  df["source"] = "data_commons"

  semaphore = asyncio.Semaphore(args.max_concurrency)
  pbar = tqdm(total=len(df), initial=int(already_done), desc="Data Commons Research")

  tasks = []
  for idx, row in df.reset_index(drop=True).iterrows():
    if pd.notna(row.get("research_output")) and str(row.get("research_output")).strip():
      continue
    prompt = build_user_prompt(row)
    task = _process_single_row(
        idx=idx,
        prompt=prompt,
        client=client,
        model_id=args.model_name,
        place_name=args.place_name,
        place_dcid=args.place_dcid,
        semaphore=semaphore,
        pbar=pbar,
    )
    tasks.append(task)

  for coro in asyncio.as_completed(tasks):
    idx, dossier = await coro
    df.at[idx, "research_output"] = dossier
    df.to_csv(args.output_csv, index=False)

  pbar.close()
  logging.info(f"Successfully processed. Output written to {args.output_csv}")


def get_args():
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description=(
          "Run Data Commons agentic research on topic and opinion pairs."
      )
  )
  parser.add_argument(
      "--input_csv",
      required=True,
      help="Path to the input CSV file (from Phase 1).",
  )
  parser.add_argument(
      "--output_csv", required=True, help="Path to save the output CSV file."
  )
  parser.add_argument(
      "--place_name",
      default="",
      help="Human-readable name of the target place.",
  )
  parser.add_argument(
      "--place_dcid",
      default="",
      help="Data Commons DCID of the target place.",
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
      default="gemini-3-flash-preview",
      help="The name of the Gemini Model. Default: gemini-3-flash-preview.",
  )
  parser.add_argument(
      "--max_concurrency",
      type=int,
      default=5,
      help="Maximum number of concurrent Data Commons research tasks. Default: 5.",
  )
  return parser.parse_args()


if __name__ == "__main__":
  asyncio.run(main(get_args()))
