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

import json
import os
import typing
import pandas as pd
import pydantic
from src.models import genai_model


_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def load_config(config_path: str = _CONFIG_PATH) -> dict:
  """Loads configuration from a JSON file."""
  if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found: {config_path}")
  with open(config_path, "r", encoding="utf-8") as f:
    return json.load(f)


_CONFIG = load_config()
LOCAL_DATASETS_REGISTRY = _CONFIG.get("file_sources", {})
TOOLS_REGISTRY = _CONFIG.get("tools", {})


class Category(pydantic.BaseModel):
  """Represents an insight evaluation category."""
  name: str
  display_name: str
  definition: str
  criteria: str
  eval_question: str
  eval_scale: str  # e.g., "Yes/No", "1-5 Scale"


class Insight(pydantic.BaseModel):
  """A single evidence-backed finding for one category of one policy proposal.

  An insight is a building block, not a deliverable. The report layer composes
  insights into the problem statement and recommendation path, so no fixed
  shape (lead sentence, bullet count) is imposed here; the category's own
  standard decides how much structure the text needs.
  """

  full_insight: str = pydantic.Field(
      description=(
          "The finding written as natural language prose, shaped only by what"
          " the target category's standard asks for. Do not impose a fixed"
          " template, lead sentence, or bullet count. Every empirical"
          " statistic, report finding, count, percentage, dollar amount, or"
          " quantitative data point used as argument basis MUST be hyperlinked"
          " in Markdown format immediately where it appears:"
          " `[metric](source_url_or_filepath)`. Proper nouns, administrative"
          " service hotlines (e.g. 311), route numbers (e.g. Highway 153), and"
          " legal code sections (e.g. City Code Chapter 32) are exempt from"
          " linking."
      )
  )
  source: str = pydantic.Field(
      description=(
          "Source pathway: 'file_agent', 'deep_research', or 'data_commons'."
      )
  )
  category: str = pydantic.Field(
      description="Canonical category name matching the target category under evaluation."
  )
  citation_text: str = pydantic.Field(
      description="Verbatim excerpt, calculation trace, or factual basis from the source evidence."
  )
  citation_link: str = pydantic.Field(
      description="Exact file path (/data/...) or primary web URL (https://...) supporting the insight."
  )


class InsightOutputStructure(pydantic.BaseModel):
  insights: list[Insight]


class ReportInsightItem(pydantic.BaseModel):
  """One supporting point inside a report block."""

  category: str = pydantic.Field(
      description=(
          "Canonical category name of this supporting point, matching the"
          " report template item in order (e.g. 'problem_sizing',"
          " 'reference_point', 'resident_impact', 'implementation_path',"
          " 'cost_estimate')."
      )
  )
  text_with_citations: str = pydantic.Field(
      description=(
          "The supporting point prose (without a leading '- ' bullet prefix),"
          " where every metric is immediately followed by a Markdown link to"
          " its source `[metric](url)` carried over unchanged from the source"
          " insight."
      )
  )


class ReportBlock(pydantic.BaseModel):
  """One composed block of a policy report (supporting points + opening tldr)."""

  insights: list[ReportInsightItem] = pydantic.Field(
      description=(
          "The supporting points of this block, in the exact order the block"
          " specification lists them. Written before `tldr` so that every"
          " metric referenced in `tldr` is already grounded here."
      )
  )
  tldr: str = pydantic.Field(
      description=(
          "The 2-3 sentence opening statement (TL;DR) of this block, written"
          " from the supporting points above and the block's opening-statement"
          " category insights. Every metric mentioned here must already appear"
          " in `insights`. Do NOT include Markdown links in `tldr`."
      )
  )


class PolicyReport(pydantic.BaseModel):
  """The composed deliverable for one policy proposal.

  Built by the report layer in a single structured call from all of a
  proposal's insights. Each block (`problem_definition` and `recommendation`)
  contains its supporting `insights` (each tagged with `category` and
  `text_with_citations`) and its 2-3 sentence `tldr`.
  """

  problem_definition: ReportBlock = pydantic.Field(
      description=(
          "The Problem Definition (The Hook) block, containing supporting"
          " `insights` (problem_sizing, resident_impact) and the opening"
          " `tldr`."
      )
  )
  recommendation: ReportBlock = pydantic.Field(
      description=(
          "The Recommendation (the solution) block, containing supporting"
          " `insights` (implementation_path, reference_point, cost_estimate)"
          " and the opening `tldr`."
      )
  )
  participant_quote_ids: list[int] = pydantic.Field(
      description=(
          "The bracketed numbers of the participant quotes picked for the"
          " report. An empty list when no quotes were provided."
      )
  )


class ReportItem(typing.NamedTuple):
  """One item of the report template.

  Attributes:
    category: The insight category whose insights feed the item.
    marker: The template's marker for the item (`required` or `optional`), or
      empty where the template gives none.
    requirement: What the item must contain, verbatim from the template.
  """

  category: str
  marker: str
  requirement: str

  @property
  def tag(self) -> str:
    """The item's tag as the template writes it, e.g. `<cost_estimate>`."""
    if self.marker:
      return f"<{self.category}: {self.marker}>"
    return f"<{self.category}>"


def format_local_dataset_descriptions(
    file_list: list[str] | None = None,
    registry: dict | None = None,
) -> str:
  """Formats dataset descriptions for prompts using canonical citation tokens.

  Each dataset is listed by the exact `<path>` string recorded in the
  registry. That string is what agents are required to cite and what phase 3
  resolves to `source_url`, so the prompt, the citation, and the resolver all
  refer to a dataset by the same characters.

  Args:
    file_list: Optional subset of datasets, given as paths or filenames.
    registry: Optional registry override; defaults to the configured one.

  Returns:
    A newline-separated numbered list of datasets and their descriptions.
  """
  active_registry = registry if registry is not None else LOCAL_DATASETS_REGISTRY
  if not file_list:
    file_list = list(active_registry.keys())
  formatted = []
  for i, key in enumerate(file_list):
    basename = os.path.basename(key)
    entry = active_registry.get(basename, {})
    desc = entry.get("description", "Unknown municipal dataset.")
    path = entry.get("path", f"/data/{basename}")
    line = f"{i+1}. `{path}` - {desc}"
    line += _format_dataset_guidance(entry)
    formatted.append(line)
  return "\n".join(formatted)


def _format_dataset_guidance(entry: dict) -> str:
  """Renders the optional per-dataset guidance fields of a registry entry.

  These fields keep dataset-specific editorial decisions in `config.json`
  rather than in prompt prose, so the prompts stay general:

  * `evidence_use`: `"evidence"` (default) or `"context_only"`. Context-only
    files (for example the participants' own posts) may inform research but
    must never be cited as proof that the problem exists.
  * `describe_as`: the exact phrase to use when the file is named in prose.
  * `do_not_caveat`: known limitations the city has already explained; the
    agents must not add caveats about them.
  * `superseded_by`: a newer file to prefer for figures both files contain.

  Args:
    entry: One registry entry.

  Returns:
    Guidance text to append to the dataset's line, or an empty string.
  """
  notes = []
  if entry.get("evidence_use") == "context_only":
    notes.append(
        "Context only: use it to understand what residents raise, but never"
        " cite it as evidence that the problem exists or how large it is."
    )
  if entry.get("describe_as"):
    notes.append(f"Refer to it in prose as \"{entry['describe_as']}\".")
  if entry.get("do_not_caveat"):
    notes.append(
        "Treat it as authoritative and do not add caveats about:"
        f" {entry['do_not_caveat']}."
    )
  if entry.get("superseded_by"):
    notes.append(
        f"For figures that also appear in `{entry['superseded_by']}`, use that"
        " newer file instead."
    )
  if not notes:
    return ""
  return " " + " ".join(notes)


def format_tools_descriptions(
    tool_list: list[str] | None = None,
    registry: dict | None = None,
) -> str:
  """Formats tool descriptions for prompts using tool names, API URLs, and descriptions."""
  active_registry = registry if registry is not None else TOOLS_REGISTRY
  formatted = []
  target_keys = tool_list if tool_list is not None else list(active_registry.keys())
  for i, name in enumerate(target_keys):
    tool_info = active_registry.get(name, {})
    api = tool_info.get("api", "")
    desc = tool_info.get("description", "No description provided.")
    if api:
      formatted.append(f"{i+1}. `{name}` (API: `{api}`)\n   Description: {desc}")
    else:
      formatted.append(f"{i+1}. `{name}`\n   Description: {desc}")
  return "\n".join(formatted)


# ==============================================================================
# CATEGORY DEFINITION
# ==============================================================================
# The canonical insight taxonomy. Each definition below is copied verbatim from
# the written category specification; do not paraphrase or "improve" it here.
# The generation prompt shows the model this text followed by the category's
# report usage, which is derived from `REPORT_FIELDS` below. `eval_question` is
# derived from the definition for the offline rubric helper and is not part of
# the written specification.
_PROBLEM_DEFINITION = (
    "Clearly defines the who, what, where, when, and why of the problem being"
    " addressed (i.e. location, impacted districts and groups, ec.) by the"
    " policy proposal."
)

_PROBLEM_SIZING = (
    "Describes the prevalence, severity, concentrations, etc. of the problem"
    " using nominal (e.g. number of\u2026 ) and relative (e.g. percentage"
    " of\u2026) indicators alongside a benchmark to gauge if the sizing is"
    " within expectations."
)

_RESIDENT_IMPACT = (
    "Defines the direct, everyday toll a civic problem takes on the average"
    " citizen. It establishes a tangible human connection by illustrating"
    " exactly how the proposed solution will resolve this burden and improve"
    " their daily life."
)

_RECOMMENDATION = (
    "A high-level summary that translates the proposed solution into a single"
    " actionable directive, clearly defining exactly what the city must do, who"
    " will do it, and how it will be funded."
)

_IMPLEMENTATION_PATH = (
    "Defines and describes the precise administrative, legislative, or"
    " operational mechanism and vehicle required to turn the policy proposal"
    " into reality that is the most direct and efficient."
)

_REFERENCE_POINT = (
    "A well documented, real-world precedent from a peer city or past pilot"
    " program that illustrates how a similar challenge was successfully"
    " managed. It provides a concrete blueprint for action, helping to validate"
    " feasibility, identify funding mechanisms, and anticipate implementation"
    " risks."
)

_COST_ESTIMATE = (
    "A cost estimate is a projected financial requirement grounded in"
    " real-world benchmarks from previous feasibility studies, pilot programs,"
    " or peer city initiatives. It provides a concrete, defensible budget"
    " figure based on proven precedent rather than theoretical calculations."
)

# The report template, verbatim from the written requirements. This is the one
# place that says how the insight categories are used to build the report: the
# report prompts below render their blocks from it, and each category's usage
# note is derived from it. Each block has an opening statement (`tldr`) and its
# supporting points, in order. Every item names the category whose insights
# feed it and carries the template's own marker (`required`, or empty where the
# template gives none); only items marked `optional` may be left out.
REPORT_FIELDS = {
    "problem_statement": {
        "label": "Problem Definition (The Hook)",
        "tldr": ReportItem(
            "problem_definition",
            "required",
            "anchoring on the proposed policy above, clearly define what the"
            " problem is in 2-3 sentences. This is the hook that should be"
            " easily understood in isolation that intuitively connects the"
            " problem to the policy proposal. Do not define more than one clear"
            " problem - it introduces complexity. Describe the condition the"
            " proposal tries to fix, not the absence of the proposed program,"
            " and do not describe the solution here. Do not introduce metrics"
            " here that aren't explained in the sections below.",
        ),
        "points": [
            ReportItem(
                "problem_sizing",
                "required",
                "State the baseline metric that describes the size of the"
                " problem today, framed as need against capacity wherever the"
                " evidence allows (e.g., lane-miles that need repaving each"
                " year against lane-miles the budget covers, share of"
                " residents without a park within a 10-minute walk against a"
                " benchmark). Every count needs a denominator, a trend or a"
                " benchmark so the reader can tell whether it is large. When"
                " the problem is concentrated in a particular location (i.e."
                " street, district), call out the street or district."
                " Prioritize city audits, plans, budgets, code and studies;"
                " draw on several of them rather than building the section on"
                " one dataset. Use service request logs only when the"
                " proposal is about service responsiveness, and never present"
                " a request status as the cause of a wait. Describe the"
                " problem, not the absence of the proposed program. Every"
                " metric must be followed immediately by a citation linking"
                " to the source of truth.",
            ),
            ReportItem(
                "resident_impact",
                "required",
                "Clearly articulate how the exact mechanism of this policy"
                " (e.g., providing grading assistance and gardening safety"
                " templates) removes a direct administrative, physical, or"
                " financial burden for the average resident. Semantic Scoping"
                " Constraint: You must interpret words like \"safety,\""
                " \"health,\" or \"equity\" strictly within the immediate"
                " operational context of the policy (e.g., \"safety\" in a"
                " gardening policy means physical soil/tool safety, not"
                " neighborhood crime). Data Scope Constraint: You are strictly"
                " forbidden from citing macro-level neighborhood statistics,"
                " violent crime rates, or overarching poverty metrics to"
                " inflate the problem. Anchor the impact strictly to the"
                " localized, household-level friction the policy resolves. Do"
                " not forcefully stretch to make systemic connections.",
            ),
        ],
    },
    "recommendation_path": {
        "label": "Recommendation (the solution)",
        "tldr": ReportItem(
            "recommendation",
            "",
            "anchoring on the bullets listed below, in exactly 2-3 sentences,"
            " describe the recommended solution and the path to implementing it"
            " to actualize the policy proposal. It must read as a complete,"
            " persuasive thought if the user never clicks to read the rest. It"
            " captures the What and the Why. Draw a direct connection between"
            " how the implementation path bridges the proposal to the problem."
            " Do not introduce metrics here that aren't explained in the"
            " sections below.",
        ),
        "points": [
            ReportItem(
                "implementation_path",
                "required",
                "State the exact administrative, operational, regulatory, or"
                " legislative instrument required (e.g., Departmental Rule"
                " Change, RFP/Procurement, Administrative Guideline, or Council"
                " Ordinance) and the executing city department. Only propose 1"
                " option. Be specific and direct. Do not default to \"passing a"
                " law\" if an administrative/operational lever is faster. Cite"
                " any dependencies or risks (e.g., the specific City Code,"
                " Charter section, or State Statute) required to execute it."
                " Explicitly name or call out any risks.",
            ),
            ReportItem(
                "reference_point",
                "required",
                "anchor on the problem definition and implementation path above"
                " and identify precedents, case studies, or examples of when"
                " other cities and states have tried to address problems using"
                " similar implementation paths. If benchmarks are not"
                " available, try using historical data. Provide the context on"
                " what made the situations similar. Focus on what the other"
                " city did that this city could draw from, include the"
                " measured outcome when a source documents one (e.g.,"
                " \"Knoxville reduced crashes by 40%\"), and call out"
                " comparable risks. Every metric must be followed immediately"
                " by a citation linking to the source of truth. Make sure that"
                " the data/evidence is directly relevant to the problem.",
            ),
            ReportItem(
                "cost_estimate",
                "",
                "Lead with the cost. If the proposal states its own cost,"
                " revenue or quantity, use that figure as given and do not"
                " reinterpret its units. Otherwise cite previous local"
                " feasibility studies, RFPs, contracts, budgets, or the"
                " reference point above as the financial benchmark (e.g.,"
                " \"Hoboken installed this for $12k per intersection\") while"
                " clearly citing sources. State the type of funding required"
                " (e.g., Federal grant, Capital Bond, or reallocation of"
                " existing operating budget) and whether the cost is a"
                " one-time capital expense or a recurring operational cost."
                " Simple arithmetic on cited inputs is allowed and expected"
                " when it helps the reader (a multi-year total, unit cost"
                " times quantity, a ratio to an existing budget line): cite"
                " both inputs and call the result an estimate. Do NOT build"
                " models, amortizations or projections from assumptions."
                " Mention savings only when a source quantifies them for this"
                " kind of work, and keep expense and savings in separate"
                " sentences. Do not repeat the reference point's figures."
                " Every metric must be followed immediately by a citation"
                " linking to the source of truth.",
            ),
        ],
    },
}

# The participant quotes section of the report template, verbatim.
MAX_PARTICIPANT_QUOTES = 3
_PARTICIPANT_QUOTES_REQUIREMENT = (
    f"Pick up to {MAX_PARTICIPANT_QUOTES} quotes from residents that showcase"
    " the nature of the problem on a humanistic level."
)

# ==============================================================================
# PUBLICATION STANDARD
# ==============================================================================
# One set of writing rules shared by the insight stage and the report stage,
# so the two cannot drift apart. Every rule below answers a recurring reviewer
# complaint on published reports; keep them together and keep them short.
PUBLICATION_STANDARD = """PUBLICATION STANDARD (applies to every sentence you write):
1. No meta or defensive writing. Never mention sources as sources, data gaps, what could not be analyzed, what a reader should verify, or the drafting process. Write only what the evidence supports and omit unsupported claims without commentary.
2. Problem before solution & single-issue focus. Problem sections describe the condition the proposal tries to fix, not the absence of the proposed program and not how the proposal works. Solution content and solution cost benchmarks belong in the recommendation sections. When a proposal touches on multiple clauses, define and size only the single primary problem addressed by the core mechanism.
3. Size the problem as need against capacity. Give every count a denominator, a trend or a benchmark; prefer "X needed against Y delivered" over raw counts. Focus `problem_sizing` on citywide scale and system capacity rather than multi-step unit-conversion formulas, and state the continuous time range given in the dataset description whenever citing service request logs.
4. Evidence diversity & scope alignment. Do not build a section on one dataset when the research offers others (audits, plans, budgets, code, studies). Use service request logs only when the proposal is about service responsiveness, and never treat a request status as the cause of a wait. Never link broad citywide statistics to a targeted proposal unless the source shows they apply to the exact same locations or program scope. Never cite residents' own posts as evidence that a problem exists.
5. Tone. Refer to external municipal examples as "other cities". Describe recommended actions, legal steps, and projected outcomes with conditional modal verbs (such as could, would, or can) rather than prescriptive directives or declarative present-tense verbs for unacted proposals. Frame outcomes from other cities as indicative evidence (such as show, indicate, or suggest) rather than dispositive proof, and use clear, direct sentence connectors (such as "and", "whereas", or starting a new sentence) rather than concessive clauses. Define every acronym, technical index, or specialized rating scale in plain everyday words in each section where it appears, prefer everyday words over archaic statutory terms, and name every actor explicitly. Avoid unquoted editorial adjectives. The recommendation opening must explain how the mechanism solves the problem rather than restating the policy as a directive.
6. Source quality. Never cite social media, video platforms, user forums, document-sharing sites, wikis, or automated news aggregators; discard any figure supported only by such domains. Prefer official city, state and federal documents, audits, plans, budgets, code and peer-reviewed studies.
7. Editions and labels. When a figure exists in several editions of a document, use only the latest edition (omitting superseded editions) and name the document and section precisely.
8. Legal claims & administrative path. State a legal conclusion no more strongly than the cited source does, name a needed legal review at most once in the implementation path, and state the responsible department and governing instrument concisely without listing procedural day-count calendars.
9. Consistency, baselines, and attribution. Every comparison you state (difference, ratio, "less than half", "largest") must follow from the numbers you give and explicitly state what it is compared against. Never attribute a figure from another city's case study to the target city, and keep `reference_point` focused on external examples that share the proposal's mechanism rather than criticizing the target city's past performance.
10. No repetition. Each figure appears at most once across the supporting points (do not repeat `reference_point` figures in `cost_estimate`, do not repeat a specific household example word-for-word between the opening statement and `problem_sizing`, and state combined program budget totals rather than listing sub-line splits); an opening statement may restate a supporting point's figure in everyday terms without a citation.
11. Proposal figures. When the proposal states a cost, revenue, quantity or threshold, use it as given; do not reinterpret its units or second-guess it.
"""


def _report_usage(category_name: str) -> str:
  """Lists, from the report template, the report items a category feeds.

  Args:
    category_name: Canonical name of an insight category.

  Returns:
    A usage note naming each template item tagged with the category, with the
    item's requirement verbatim.

  Raises:
    ValueError: If no template item is tagged with `category_name`, because
      such a category would be generated and then never used.
  """
  lines = []
  for field in REPORT_FIELDS.values():
    tldr = field["tldr"]
    if tldr.category == category_name:
      lines.append(
          f"- The opening statement of the \"{field['label']}\" block"
          f" {tldr.tag}: {tldr.requirement}"
      )
    for point in field["points"]:
      if point.category == category_name:
        lines.append(
            f"- A supporting point of the \"{field['label']}\" block"
            f" {point.tag}: {point.requirement}"
        )
  if not lines:
    raise ValueError(
        f"No report template item is tagged with category {category_name!r}."
    )
  return (
      "Report usage: the published report uses insights of this category for"
      " the items below.\n" + "\n".join(lines)
  )


def _make_category(
    name: str, display_name: str, definition: str, eval_question: str
) -> Category:
  """Builds a category whose standard is its definition plus its report usage.

  Args:
    name: Canonical category name, as used in the report template tags.
    display_name: Human-readable category name.
    definition: The category definition, verbatim from the specification.
    eval_question: Yes/No rubric question derived from the definition.

  Returns:
    The assembled category. `criteria` repeats `definition` so that the two
    prompt slots can never drift apart.
  """
  standard = definition + "\n\n" + _report_usage(name)
  return Category(
      name=name,
      display_name=display_name,
      definition=standard,
      criteria=standard,
      eval_question=eval_question,
      eval_scale="Yes/No",
  )


INSIGHT_CATEGORIES = [
    _make_category(
        name="problem_definition",
        display_name="Problem definition",
        definition=_PROBLEM_DEFINITION,
        eval_question=(
            "Does the insight clearly define the who, what, where, when, and"
            " why of the problem the policy proposal addresses, such as the"
            " location and the impacted districts and groups?"
        ),
    ),
    _make_category(
        name="problem_sizing",
        display_name="Problem sizing",
        definition=_PROBLEM_SIZING,
        eval_question=(
            "Does the insight describe the prevalence, severity, or"
            " concentration of the problem with nominal indicators (such as"
            " counts) and relative indicators (such as percentages), alongside"
            " a benchmark that shows whether the size is within expectations?"
        ),
    ),
    _make_category(
        name="resident_impact",
        display_name="Resident impact",
        definition=_RESIDENT_IMPACT,
        eval_question=(
            "Does the insight describe the direct, everyday toll the problem"
            " takes on the average resident, and illustrate how the proposed"
            " solution would resolve that burden and improve their daily life?"
        ),
    ),
    _make_category(
        name="recommendation",
        display_name="Recommendation",
        definition=_RECOMMENDATION,
        eval_question=(
            "Does the insight translate the proposed solution into a single"
            " actionable directive that states exactly what the city must do,"
            " who will do it, and how it will be funded?"
        ),
    ),
    _make_category(
        name="implementation_path",
        display_name="Implementation path",
        definition=_IMPLEMENTATION_PATH,
        eval_question=(
            "Does the insight name the precise administrative, legislative, or"
            " operational mechanism and vehicle that turns the policy proposal"
            " into reality most directly and efficiently?"
        ),
    ),
    _make_category(
        name="reference_point",
        display_name="Reference point",
        definition=_REFERENCE_POINT,
        eval_question=(
            "Does the insight cite a well-documented, real-world precedent"
            " from a peer city or past pilot program that tackled a similar"
            " challenge, and use it to validate feasibility, identify funding"
            " mechanisms, or anticipate implementation risks?"
        ),
    ),
    _make_category(
        name="cost_estimate",
        display_name="Cost estimate",
        definition=_COST_ESTIMATE,
        eval_question=(
            "Does the insight give a projected cost grounded in a cited"
            " real-world benchmark from a previous feasibility study, pilot"
            " program, or peer city initiative, rather than a theoretical"
            " calculation?"
        ),
    ),
]


def get_criteria_categories() -> list[str]:
  """Returns the list of criteria category names."""
  return [c.name for c in INSIGHT_CATEGORIES]


def get_criteria_guidelines(target_category: str | None = None) -> str:
  """Returns generation guidelines for categories."""
  if target_category:
    for cat in INSIGHT_CATEGORIES:
      if cat.name == target_category:
        return f"- {cat.display_name}: {cat.criteria}"
  return "\n".join([f"- {c.display_name}: {c.criteria}" for c in INSIGHT_CATEGORIES])


def get_insight_evaluation_questions() -> dict[str, str]:
  """Returns evaluation questions for each category."""
  return {c.name: c.eval_question for c in INSIGHT_CATEGORIES}


# ==============================================================================
# RESEARCH_IDEA (#RESEARCH_IDEA)
# ==============================================================================
def get_research_ideas_prompt(
    topic: str,
    proposal: str = "",
    quotes: list[str] | None = None,
    place_name: str = "",
    peer_cities: list[str] | None = None,
    datasets_registry: dict | None = None,
    tools_registry: dict | list | None = None,
) -> str:
  """Generates Stage 1 prompt for research ideas across municipal topics and citizen proposals."""
  formatted_quotes = ""
  if quotes:
    quote_lines = []
    for q in quotes:
      quote_lines.append(f"- {q}")
    formatted_quotes = "\n".join(quote_lines)
    formatted_quotes = f"Participant Context Quotes:\n{formatted_quotes}\n"

  datasets_desc = format_local_dataset_descriptions(registry=datasets_registry)
  tools_desc = format_tools_descriptions(registry=tools_registry)
  tools_section = ""
  if tools_desc:
    tools_section = f"""
You also have the following external APIs / tools available:
{tools_desc}
"""

  peer_cities_str = ", ".join(peer_cities) if peer_cities else "comparable cities"
  place_header = f"for {place_name}" if place_name else "for the target municipality"

  return f"""You are a senior municipal policy researcher and data analyst {place_header}.
We are investigating community input on the topic "{topic}".
Specifically, we are analyzing the citizen proposal (phrased as a solution/initiative):
"{proposal}"

{formatted_quotes}
To help ground your research, you have the following local municipal datasets and documents available in `/data/`:
{datasets_desc}
{tools_section}
The research ideas and potential insights should address one or more of the following canonical policy categories:
{get_criteria_guidelines()}

INSTRUCTIONS:
Generate a comprehensive list of specific, actionable research questions and investigative directions to explore whether empirical data, legal authority, and precedents from other cities support or challenge this proposal.
0. Primary Focus on Central Mechanism (Sprawling / Multi-Part Proposals): If the citizen proposal touches on multiple topics, sub-clauses, or secondary conditions, identify and anchor your research questions primarily on the SINGLE most central problem and primary operational mechanism of the proposal. Do not let secondary or procedural sub-clauses splinter the research focus.
1. Problem Before Solution: First ask what condition the proposal is trying to fix and how large it is, framed as need against current capacity (for example, how much work is needed each year against how much the current budget delivers). The problem is never "the proposed program does not exist yet".
2. Local Data & Tool Investigation: Propose targeted analyses across the local files in `/data/` and configured API tools (such as querying GIS REST endpoints for neighborhood and district boundaries), selecting the datasets that most directly measure the proposal's core subject (such as municipal research briefs, budget and capital plans, audit reports, governing code sections, collision logs, or service request logs). Draw on several sources rather than one. Only propose 311 service request analyses when the proposal directly addresses municipal maintenance responsiveness or service complaints; do not force 311 metrics onto policies where service complaints are tangential.
3. Precedents in Other Cities: Propose investigations into operational playbooks, service level agreements (SLAs), and post-implementation outcomes from comparable cities (such as {peer_cities_str}).
4. Statutory & Fiscal Feasibility: Propose queries to identify the governing local municipal code / charter authority, and to find the specific budget lines, contracts or studies that fund or price this mechanism today (not the city's total budget).
5. Equity & Demographic Distribution: Only when the proposal targets particular groups or areas, or when its benefits or burdens would clearly fall unevenly, propose questions on whether lower-income or historically underserved neighborhoods face disproportionate impacts.

Do not return final research conclusions now. Instead, generate at least 6 distinct, structured research ideas to guide our research investigations.
"""


def _research_method_instructions(evidence_sources: str) -> str:
  """Returns the hypothesis-driven research method shared by Phase 2 agents.

  Deep Research and the File Agent follow the same method and differ only in
  where they look for evidence.

  Args:
    evidence_sources: Where this agent finds evidence, phrased to complete
      "the sequence of evidence needed to test it ...".

  Returns:
    The instruction block.
  """
  return f"""RESEARCH METHOD (plan the reasoning before collecting evidence):
The research ideas above are starting points, not a checklist. Anchor on the SINGLE most central problem the proposal seeks to solve and its primary operational mechanism, then build a short hypothesis-driven plan before you collect anything. For each research direction you pursue:
- Hypothesis & Rationale: State the claim or assumption being tested and how it connects the problem to the proposed mechanism.
- Verification Path: Lay out the sequence of evidence needed to test it {evidence_sources}, moving from baseline scope to causal or spatial alignment.
- Scope Guardrail: Define what evidence actually measures the proposal's target, and do not substitute loosely related citywide metrics (such as general 311 complaint logs or department-wide budgets) that do not.
Then execute the plan. Keep only evidence that directly supports or refutes a hypothesis, and omit untested hypotheses and missing data without commentary.
"""


# ==============================================================================
# FILE_AGENT (#FILE_AGENT)
# ==============================================================================
def get_file_agent_prompt(
    topic: str,
    proposal: str = "",
    sandbox_work_dir: str = "/workspace/runs",
    file_list: list[str] | None = None,
    place_name: str = "",
    research_ideas_prompt: str = "",
    tool_list: list[str] | None = None,
    tools_registry: dict | list | None = None,
) -> str:
  """Generates Stage 2 prompt for File Agent to analyze local municipal CSV/PDF/Markdown files in /data/ and call configured API tools."""
  if not file_list:
    file_list = []
    for fname in LOCAL_DATASETS_REGISTRY.keys():
      file_list.append(f"/data/{fname}")

  proposal_context = ""
  if proposal:
    proposal_context = f"""Specifically, we are analyzing the citizen proposal (initiative/solution):
"{proposal}"
"""

  research_context = ""
  if research_ideas_prompt:
    research_context = f"""
Here are research ideas generated in Phase 1:
{research_ideas_prompt}
"""

  files_str = format_local_dataset_descriptions(file_list)
  tools_str = format_tools_descriptions(tool_list, registry=tools_registry)
  place_header = f"for the City/Region of {place_name}" if place_name else "for the local municipality"

  tools_section = ""
  if tools_str:
    tools_section = f"""
In addition to the local data files, you have access to the following external APIs / tools:
{tools_str}

When relevant to your research, you may write and execute custom Python scripts using `urllib.request` or `requests` to query these APIs and combine them with local datasets in `/data/` (for example, using `shapely` for point-in-polygon spatial lookups).
"""

  method = _research_method_instructions(
      "in the local files in `/data/` and the configured API tools"
  )

  return f"""You are a senior data analyst {place_header}.
We are investigating the topic "{topic}".
{proposal_context}
{research_context}
You have the following data files available in the shared directory `/data/`:
{files_str}
{tools_section}
{method}
Write and execute custom Python scripts using pandas, PyMuPDF (fitz) / pypdf, requests / urllib, or standard file I/O to test your hypotheses and extract documented facts from these datasets and tools.
CRITICAL: The 'code_execution' tool in this environment only supports Python (language='python'). You must perform all operations programmatically by writing and running standard Python code.

INVESTIGATIVE LOGIC & SELECTIVE EVIDENCE CURATION:
- Hypothesis-Driven Inquiry (Not Keyword Harvesting): Do not indiscriminately scrape every number that matches a keyword. Follow your verification paths to test each hypothesis, and retain only metrics that directly prove or refute the core policy argument.
- Scope & Attribution Verification: Before reporting a statistic or budget line item, verify what it actually measures. Do NOT connect two separate metrics (such as a specific asset inventory and general citywide 311 complaints, or a narrow pilot and an entire departmental budget) unless your analysis confirms they apply to the exact same locations or program scope. If a dataset only offers loosely related citywide numbers that would mislead downstream synthesis, leave them out of the report.
- Evidence Diversity: Cover the proposal from several files (audits, plans, budgets, code, studies) rather than mining one dataset. Files marked "Context only" in the dataset list may inform what to look for but must never be cited as evidence of the problem.
- Editions: When two files carry different editions of the same figure (for example two budget years), use the newer one and name the document precisely.

DIRECT EXTRACTION & CALCULATION GUIDELINES:
- Financial & Budget Data:
  Every cost, budget allocation, tax rate, revenue figure or dollar amount MUST come verbatim from the source files. Simple arithmetic on two cited figures is allowed (a multi-year total, unit cost times quantity, a ratio to a budget line) when you show both inputs; label the result an estimate. You are STRICTLY FORBIDDEN from financial modelling, amortization schedules, synthetic revenue projections or multi-step extrapolations.
- Non-Financial Data & Metrics (CALCULATIONS PERMITTED & ENCOURAGED):
  For non-financial datasets that directly measure the proposal's subject (such as infrastructure coverage rates, asset inventories, collision logs, demographic breakdowns, corridor concentrations, or service request logs when the proposal directly concerns maintenance responsiveness), you are permitted and encouraged to write Python scripts to calculate exact sums, percentages, averages, and geographic aggregations directly from the underlying files.
- External APIs & GIS Tools:
  When relevant to your investigation, you may query configured external APIs (such as ArcGIS REST endpoints) using Python (`urllib.request` or `requests`). For geospatial analysis, you can parse GeoJSON responses and use `shapely` (e.g. `shapely.geometry.shape(geom).contains(point)`) to translate address or coordinate references across local files into official neighborhood associations and council districts. Cache or store intermediate query results inside your sandbox working directory.
- Efficient File Inspection:
  * Tabular CSVs: Inspect column headers first (`df.columns`) before filtering with case-insensitive matching (`str.contains(..., case=False, na=False)`).
  * Large PDFs (Budgets / Studies / Audits): Write a Python script using pypdf or PyMuPDF to search all pages for target keywords first, then extract verbatim text and tables from matching pages.
  * Municipal Code, Charter & Research Briefs: Search and quote exact sections, legal authorities, and documented baselines using UTF-8 Python scripts.

MANDATORY CITATION FORMAT:
Every figure you report MUST be written as a Markdown link, with the figure as the link text and its source as the link target:
  `[21% sidewalk coverage](/data/chattanooga_research_brief_06_pedestrian_and_bicycle_networks.md)`
- The link target MUST be one of the `/data/...` paths listed in the dataset list above, copied character for character. Do NOT shorten it, rename it, or alter the extension.
- Before you use a path in a link, confirm the file exists with `os.path.exists`. If it does not, the path is wrong: find the real one in the dataset list above and use that instead. Never write a path from memory.
- For a figure obtained from an API tool, use the exact API endpoint URL as the link target.
- Do NOT report any figure you cannot link to a listed dataset path or API endpoint.
- Mentioning a dataset name in the surrounding text does not satisfy this requirement. The link must be attached to the figure itself.

IMPORTANT GROUNDING CONSTRAINT:
You must ONLY report findings, statistics, or case details that you extract directly from the local files in `/data/` or the configured external API tools. Do NOT include or synthesize statistics about other cities or general web facts unless they are explicitly mentioned inside the provided local files in `/data/` or returned by the configured API tools.

CRITICAL CONCURRENCY INSTRUCTION:
You are operating in a shared container environment. To prevent naming conflicts with concurrent tasks, you MUST perform all your work inside a dedicated isolated directory:
{sandbox_work_dir}

Step 1: Recursively create the directory `{sandbox_work_dir}` (using `os.makedirs('{sandbox_work_dir}', exist_ok=True)`).
Step 2: Write and execute your custom Python scripts EXCLUSIVELY inside `{sandbox_work_dir}/`.
Step 3: Store any intermediate files exclusively inside `{sandbox_work_dir}/`.

Do not write or modify any files outside of `{sandbox_work_dir}`.

When finished, return the COMPLETE structured Markdown research report as your final message, organized around the core hypotheses tested: state each finding clearly, present only the verified, directly relevant figures that support it, and explain the logical connection between the evidence and the proposal. You may also save the report to `{sandbox_work_dir}/final_report.md`, but saving it never replaces returning the full text; a summary or a pointer to the file is not an acceptable final message.
Your Markdown report MUST end with a section titled '## Approach' containing a brief, 1-2 sentence explanation of your exact data retrieval methodology.
"""


# ==============================================================================
# DEEP_RESEARCH (#DEEP_RESEARCH)
# ==============================================================================
def get_deep_research_prompt(research_ideas_prompt: str) -> str:
  """Generates Stage 2 prompt for Deep Research subagent, steering open-web research towards public open data sources."""
  method = _research_method_instructions(
      "in public documents, municipal reports, open data, and evaluations"
      " from other cities"
  )
  return f"""{research_ideas_prompt}

{method}
CRITICAL INSTRUCTIONS FOR DEEP RESEARCH:
1. Evidence Focus: Do not gather loosely related trivia; focus on evidence that directly answers how the proposed mechanism works in practice, what it costs, and what outcomes or risks resulted.
2. Open Data Steering: If the research ideas mention local files or private filesystem paths, do NOT search for private local paths. Instead, investigate public open data sources, case studies from other cities, and official reports on the same civic mechanism.
3. Source Reliability & Specific Citations: Never cite social media, video platforms, user forums, document-sharing sites, wikis, or automated news aggregators. Prefer official city, state and federal documents, audits, court decisions, and peer-reviewed studies; when a news article is the only source, prefer established local or national outlets. Capture and cite exact document titles, authors, and direct source URLs for every empirical figure and case study outcome, and cite the most recent edition when a document is updated yearly.
4. Legal Claims: State a legal conclusion no more strongly than the source does, and distinguish statutes and court decisions from commentary.
5. Cohesive Synthesis: Organize your findings into a logical Markdown research dossier structured around the core policy thesis—covering problem context, comparable implementations in other cities and their measured outcomes, legal/operational mechanisms, and concrete financial benchmarks.
"""


def get_structured_insights_prompt(
    target_category: str,
    research_output: str,
    place_name: str,
    research_ideas_prompt: str = "",
) -> str:
  """Builds the stage 3 prompt that turns research findings into insights.

  The category's own standard drives the content and the shape of the output.
  No fixed template is imposed here: composition into the problem statement and
  the recommendation path happens later, in the report layer.

  Args:
    target_category: Canonical name of the category to generate insights for.
    research_output: Combined research findings to synthesize from.
    place_name: Target municipality, used to keep the writing locally framed.
    research_ideas_prompt: Optional stage 1 directive to align the insights to.

  Returns:
    The fully rendered prompt string.

  Raises:
    ValueError: If `target_category` is empty.
  """
  if not target_category:
    raise ValueError("target_category should not be empty.")

  category_obj = None
  for candidate in INSIGHT_CATEGORIES:
    if candidate.name == target_category:
      category_obj = candidate
      break
  if category_obj is None:
    raise ValueError(
        f"Unknown target_category {target_category!r}. Expected one of:"
        f" {get_criteria_categories()}"
    )
  display_name = category_obj.display_name
  definition = category_obj.definition

  ideas_instruction = ""
  if research_ideas_prompt:
    ideas_instruction = (
        "CRITICAL STAGE 1 RESEARCH ALIGNMENT DIRECTIVE:\n"
        "You MUST align your extracted insights directly with the targeted research ideas and categories generated in Stage 1 below.\n"
        f"Stage 1 Research Directive:\n{research_ideas_prompt}\n"
    )

  datasets_guidance = format_local_dataset_descriptions()

  return f"""Please synthesize the following research findings into high-quality insights for the category `{target_category}`.

TARGET CATEGORY UNDER EVALUATION:
Category Name: `{target_category}` ({display_name})
Standard: {definition}

The Standard above is the only description of what this category must contain. Satisfy it exactly: do not add requirements it does not state, and do not skip requirements it does state.

{ideas_instruction}
LOCAL DATASET GUIDANCE (follow these exact naming, time-range, and supersession rules whenever citing local files):
{datasets_guidance}

{PUBLICATION_STANDARD}
ARGUMENTATION & WRITING SHAPE:
- Lead with the Insight, Then Select Evidence: First identify the single most important conclusion for `{target_category}` regarding the proposal's core mechanism. Then select ONLY the most direct, relevant figures or precedents from the research output to prove that conclusion—actively discard tangential or loosely connected statistics, secondary sub-clauses, weak/banned domains, and superseded editions rather than packing every available number into the paragraph.
- Topic Sentence & Logical Flow: Write each `full_insight` as natural language prose. Open with a clear topic sentence stating the main takeaway and how it connects to the proposal, then present the selected evidence in short, digestible sentences that explicitly explain how numbers relate to one another (such as baseline vs. current volume, or upfront cost vs. funding source vs. long-term return). Use clear, direct sentence connectors (such as "and", "whereas", or starting a new sentence) rather than concessive clauses.
- No Fixed Scaffold: Do NOT use a fixed template, a lead-sentence-plus-bullets scaffold, a fixed number of bullet points, prefix labels ('Pillar 1:', 'Point 1:'), or abstract placeholders ('Scenario A'). Each insight must stand on its own.

CORE QUALITY & GROUNDING DIRECTIVES:
1. STRICT GROUNDING & SCOPE ALIGNMENT (NEVER FABRICATE OR FALSIFY ATTRIBUTION):
   - Every number, statistic, dollar amount, percentage, count, or date in `full_insight` MUST be directly supported by the provided research findings. Never guess or fabricate figures.
   - Do NOT misrepresent administrative identifiers or legal section numbers (e.g. City Code §15.6, Charter 3.1) as empirical statistics.
   - Do NOT stitch together unrelated or broad citywide metrics (such as general 311 complaint backlogs or entire departmental maintenance budgets) onto a narrower policy target unless the source evidence explicitly proves they share the same locations or program scope.
2. CIVIC PLAIN ENGLISH DIRECTIVE (8th-Grade Reading Level / Town Hall Read-Aloud Standard):
   - Write at an 8th-grade reading level for regular citizens in {place_name}. Every sentence must be clear enough to be understood on first hearing when read aloud at a public town hall meeting; keep sentences concise and focused on one idea at a time.
   - Avoid consultant jargon and inflated academic phrasing, and do not echo prompt instructions in the prose (for example, write 'Past examples of similar programs show...' rather than restating the category standard).
   - Use active, concrete verbs ('fund', 'build', 'pave', 'inspect', 'contract', 'hire', 'enforce') and modal verbs ('would', 'could', 'can') for proposed actions, legal steps, and projected outcomes; explain exact City Code sections, department names, technical units/indices, and budget figures in plain everyday words.
   - Stay administratively neutral and follow the tone rules of the Publication Standard above.
3. MANDATORY IN-TEXT CITATIONS FOR QUANTITATIVE DATA (EMPIRICAL METRICS VS. PROPER NOUN IDENTIFIERS):
   - MUST CITE EVERY INDIVIDUAL METRIC (Empirical Quantities & Statistics): Enclose every individual empirical data point, count, percentage, dollar amount, physical measurement, or study year in `full_insight` in its own inline Markdown link (`[metric](source_url_or_filepath)`). If one sentence mentions two empirical numbers from the same source, link BOTH numbers individually. When citing local files in `/data/`, use the exact canonical `/data/<exact_filename>.<extension>` path.
   - STRICTLY EXEMPT FROM CITATION (Proper Nouns & Identifiers): Do NOT hyperlink numbers that function purely as names or identifiers, such as service hotlines ("311 service requests"), route numbers ("Highway 153"), code sections ("City Code Chapter 32"), or district names ("Council District 7").
   - In `citation_text`, provide a checkable, reproducible trace:
     * For `file_agent`: Explain how the figure was extracted or calculated from `/data/` (e.g. "Extract the documented baseline figure from Section 2 of the research brief, or filter the relevant tabular dataset by category and council district to compute the exact count and percentage").
     * For `data_commons`: Specify the exact place DCIDs (e.g. `geoId/4714000`) and statistical variable names.
     * For `deep_research`: Explain how the URL(s) in `citation_link` support the cited figures.
   - In `citation_link`, record the exact primary source path (`/data/<exact_filename>.<extension>`) or direct URL (`https://...`), using Data Commons Explorer deep links (`https://datacommons.org/tools/timeline#...` or `https://datacommons.org/tools/map#...`) for `data_commons`.

COVERAGE & STRICT OMISSION RULE:
- If research results contain grounded factual evidence for `{target_category}`, generate 1 or more high-quality `Insight` objects.
- If research results contain ZERO empirical data or baseline records for `{target_category}`, you MUST return an empty list: `{{"insights": []}}`. Do NOT generate placeholder disclaimers (such as "No data found for..." or "While records are unavailable...").

OUTPUT FORMAT: Return a JSON object matching this schema:
{{
  "insights": [
    {{
      "full_insight": "The finding as natural language prose shaped by the category Standard above, where every empirical statistic, report finding, count, percentage, dollar amount, or quantitative data point used as argument basis is hyperlinked in Markdown format: `[metric](source_url_or_filepath)` (proper nouns like 311 or Highway 153 and legal code sections are exempt from linking).",
      "source": "file_agent | deep_research | data_commons",
      "category": "{target_category}",
      "citation_text": "Exact reproducible step-by-step calculation or quote trace for all cited numbers.",
      "citation_link": "/data/<exact_filename>.<extension> or https://<source_url>"
    }}
  ]
}}

RESEARCH FINDINGS TO SYNTHESIZE:
{research_output}
"""


def get_verification_sandbox_prompt(
    source: str,
    link: str,
    trace: str,
    insight: str,
    sandbox_work_dir: str = "",
) -> str:
  """Returns the verification sandbox prompt for a given insight source."""
  if source == "deep_research":
    return f"""You are a factual verification auditor running in a clean verification sandbox (`fact_checking_sandbox`).
Target Source URL: "{link}"
Claim/Data Trace to Verify: "{trace}"
Raw Insight: "{insight}"

Step 1: Write and execute a Python script (using urllib.request or requests) to fetch the webpage/PDF content at target URL `{link}`.
Step 2: Inspect and search the fetched text/content to locate evidence confirming or refuting the claim or statistics in "{trace}" and "{insight}".
Step 3: Print the exact relevant text excerpts, figures, or facts found in the document that confirm or refute the claim.

Only return your final findings and print statements from your code execution. Do not explain your thought process.
"""
  elif source == "data_commons":
    return f"""You are a statistical verification auditor running in a clean verification sandbox (`fact_checking_sandbox`).
Target Data Source / API URL: "{link}"
Metric/Data Trace to Verify: "{trace}"
Raw Insight: "{insight}"

Step 1: Write and execute a Python script (using urllib.request or requests to query Data Commons REST API or fetch Data Commons webpage content at `{link}`).
Step 2: Inspect the returned data points to verify if the metric or statistics claimed in "{trace}" and "{insight}" are supported by Data Commons.
Step 3: Print the retrieved values, metrics, and data points that confirm or refute the claim.

Only return your final findings and print statements from your code execution. Do not explain your thought process.
"""
  else:
    # file_agent or composite sources
    return f"""You are a replication auditor.
You are running in the exact sandbox environment where the File Agent analyzed this proposal.
The File Agent executed code in: `{sandbox_work_dir}` (or `/workspace`).

We have the following claim/data trace to verify:
- Data Trace: "{trace}"
- Insight: "{insight}"

Step 1: Check `{sandbox_work_dir}` and `/workspace` for python scripts, notebooks, or output files written during Phase 2.
Step 2: If matching scripts exist in `{sandbox_work_dir}` or `/workspace`, execute them to verify the exact numbers/claims.
Step 3: If no matching pre-written script exists in `{sandbox_work_dir}`, write and execute a short Python script to query the raw datasets in `/data/` (e.g. research briefs, budget PDFs, City Code, or tabular CSVs) to calculate and verify whether the statistics in "{trace}" and "{insight}" are supported or refuted.
Step 4: Print the exact resulting values, counts, page excerpts, or lines retrieved.

Only return your final findings and print statements from your code execution. Do not explain your thought process.
"""


def format_insights_for_report(insights: list[dict]) -> str:
  """Renders a proposal's insights as a category-labelled list for the prompt.

  Args:
    insights: Dicts each carrying at least a `category` and a `full_insight`.

  Returns:
    One labelled entry per insight, separated by blank lines. Empty string when
    there is nothing to render.
  """
  entries = []
  position = 0
  for insight in insights:
    text = str(insight.get("full_insight", "")).strip()
    if not text:
      continue
    position += 1
    category = str(insight.get("category", "")).strip() or "uncategorized"
    entries.append(f"[{position}] category=`{category}`\n{text}")
  return "\n\n".join(entries)


def _format_report_points(field_name: str) -> str:
  """Renders a report block's supporting points as tagged bullets.

  Args:
    field_name: A key of `REPORT_FIELDS`.

  Returns:
    One `- <tag> requirement` line per supporting point, in template order.
  """
  lines = []
  for point in REPORT_FIELDS[field_name]["points"]:
    lines.append(f"- {point.tag} {point.requirement}")
  return "\n".join(lines)


def format_quotes_for_report(quotes: list[str] | None) -> str:
  """Numbers participant quotes for the report prompt.

  Args:
    quotes: Participant quotes, numbered in the order given. Callers drop blank
      quotes beforehand, because the numbers are how the model refers back to a
      quote.

  Returns:
    One `[n] "quote"` line per quote, or an empty string when there are none.
  """
  lines = []
  position = 0
  for quote in quotes or []:
    position += 1
    lines.append(f"[{position}] \"{quote}\"")
  return "\n".join(lines)


# ==============================================================================
# REPORT GENERATION
# ==============================================================================
def get_policy_report_prompt(
    proposal: str,
    insights: list[dict],
    place_name: str = "",
    topic: str = "",
    quotes: list[str] | None = None,
) -> str:
  """Builds the prompt that composes one proposal's insights into a report.

  Args:
    proposal: The citizen policy proposal the report is written about.
    insights: All insights generated for that proposal, each a dict with a
      `category` and a `full_insight`.
    place_name: Target municipality, used to keep the writing locally framed.
    topic: Optional topic heading the proposal sits under.
    quotes: Optional participant quotes about the proposal. The model picks the
      ones the report shows by their 1-based position in this list.

  Returns:
    The fully rendered prompt string.

  Raises:
    ValueError: If `proposal` is empty or no insight carries any text.
  """
  if not proposal or not proposal.strip():
    raise ValueError("proposal should not be empty.")
  insights_block = format_insights_for_report(insights)
  if not insights_block:
    raise ValueError("insights must contain at least one non-empty full_insight.")

  topic_line = f"Topic: {topic}\n" if topic else ""
  place_line = place_name or "the target municipality"
  problem_tldr = REPORT_FIELDS["problem_statement"]["tldr"]
  problem_points = _format_report_points("problem_statement")
  recommendation_tldr = REPORT_FIELDS["recommendation_path"]["tldr"]
  recommendation_points = _format_report_points("recommendation_path")

  quotes_block = format_quotes_for_report(quotes)
  if quotes_block:
    quotes_section = (
        f"PARTICIPANT QUOTES:\n{quotes_block}\n\n"
        "BLOCK 3 - `participant_quote_ids`."
        f" {_PARTICIPANT_QUOTES_REQUIREMENT} Return the bracketed numbers of"
        " the quotes you pick."
    )
  else:
    quotes_section = (
        "BLOCK 3 - `participant_quote_ids`. No participant quotes were"
        " provided, so return an empty list."
    )

  return f"""You are composing the published report for one citizen policy proposal in {place_line}.

{topic_line}Policy proposal:
"{proposal}"

You are given every insight that was researched for this proposal. Each is tagged with the category whose standard it was written against. Compose them into the structured blocks below in a single pass.

SOURCE INSIGHTS:
{insights_block}

BLOCK 1 - `problem_definition` ({REPORT_FIELDS["problem_statement"]["label"]}).
First write `insights` (the supporting points, in this exact order):
{problem_points}
Then write `tldr` (the opening statement, drawing on `{problem_tldr.category}` insights and the problem facts in the supporting points above—do NOT include any solution claims, proposed policy actions, or solution benefits from `resident_impact` in `problem_definition.tldr`):
- {problem_tldr.tag} {problem_tldr.requirement}

BLOCK 2 - `recommendation` ({REPORT_FIELDS["recommendation_path"]["label"]}).
First write `insights` (the supporting points, in this exact order):
{recommendation_points}
Then write `tldr` (the opening statement, drawing on `{recommendation_tldr.category}` insights and the supporting points above):
- {recommendation_tldr.tag} {recommendation_tldr.requirement}

{quotes_section}

FINAL RENDERED LAYOUT & COHERENCE REQUIREMENT:
In the published report and evaluation UI, each block is assembled and displayed in this exact reading order, with each segment's label shown in front:
  1. Problem Definition (The Hook):
     [Problem Definition] <problem_definition.tldr>
     [Problem sizing] <problem_definition.insights[0].text_with_citations>
     [Resident impact] <problem_definition.insights[1].text_with_citations>
  2. Recommendation (The Solution):
     [TL;DR] <recommendation.tldr>
     [Implementation path] <recommendation.insights[0].text_with_citations>
     [Reference point] <recommendation.insights[1].text_with_citations>
     [Cost estimate] <recommendation.insights[2].text_with_citations>

When concatenated in this reading order (`tldr` followed by each supporting `insights` item), each block MUST read as a smooth, coherent, connected narrative:
- Anchor the entire report on the SINGLE most central problem and primary operational mechanism of the proposal. If the source insights also contain metrics for a secondary sub-clause or downstream use of funds, drop those secondary metrics so `problem_definition` defines and sizes only one problem.
- In `tldr`, state the block's overarching takeaway in plain everyday terms. In `problem_definition.tldr`, state ONLY what the existing problem is, how large it is at a citywide level, and who it burdens today—never mention the proposed policy or how the solution works, and do not repeat a specific household example word-for-word from the supporting points. In `recommendation.tldr`, do not merely repeat the citizen proposal word-for-word; explain how the specific implementation mechanism turns the proposal into action to solve the problem, and never attribute a benchmark figure from another city to {place_line}.
- Lead every supporting `insights` paragraph with a clear topic sentence stating the point's takeaway and how it connects to the proposal, then select ONLY the most direct, relevant figures or case studies from the source insights to prove that takeaway. Do NOT dump every available statistic from the source insights into the paragraph, do NOT cite a superseded edition of a document when a newer edition is present, and do NOT stitch together loosely related citywide metrics that do not directly measure the proposal's scope.
- Keep each section strictly within its role and complementary to adjacent sections:
  * `problem_sizing` quantifies the citywide scale of the problem and current system capacity relative to need, without walking through multi-step unit-conversion formulas and without using external unit construction costs to price the solution.
  * `resident_impact` focuses on the concrete everyday household burden of the problem and ends with a brief modal clause ("can", "could", or "would") explaining how resolving the problem improves daily life, without introducing solution revenue projections or program budgets.
  * `implementation_path` states the responsible department and governing legal or administrative instrument concisely without listing procedural day-count calendars.
  * `reference_point` focuses on constructive examples from specific other cities or past pilots that used a comparable mechanism and their documented outcomes or risks, without inserting sentences that contrast or criticize {place_line}'s past performance.
  * `cost_estimate` focuses on financial requirements and funding sources using modal verbs ("would", "could", "can"), states combined program budget totals rather than sub-line item splits, and does not repeat figures already cited in `reference_point`.

{PUBLICATION_STANDARD}
HOW TO COMPOSE:
1. Each supporting point in `insights` is an object `{{"category": "<category_name>", "text_with_citations": "<prose>"}}`. Use the exact canonical category name from the `<category>` tag (`problem_sizing`, `resident_impact`, `implementation_path`, `reference_point`, `cost_estimate`), and do NOT include a leading `- ` bullet prefix or `<category>` tag inside `text_with_citations`.
2. Write every supporting point in the exact order listed unless it is marked `optional`.
3. Write `insights` before `tldr` in each block. Every number or metric mentioned in `tldr` MUST already appear in that block's `insights`. Do NOT include Markdown links or citations in `tldr`, and convert complex statistics in `tldr` into relatable everyday terms (e.g. 'more than 4 out of 5 complaints').
4. Every claim must come from the source insights. You are STRICTLY FORBIDDEN from adding facts, numbers, other cities, or policy mechanisms that do not appear above. If you name another city, name it explicitly and refer to external municipal examples as "other cities", and never attribute another city's figure to {place_line}.
5. In `text_with_citations`, carry over the exact Markdown citation link `[metric](url)` for EVERY individual empirical number you choose to include. If one sentence includes two numbers from the same source, link BOTH numbers individually. Never drop a link from a metric you include, never invent one, and never move a link onto a different number.
6. If the source insights cannot fully satisfy a point, write only what the evidence does support, and keep the point shorter rather than padding it. Do NOT invent filler, and do NOT comment on what is missing.
7. Before returning, re-read each block as a whole and check: every comparison follows from the numbers it cites and states its baseline, no figure is repeated across sections, every technical index or scale is defined in plain words in each section where it appears, every proposed action or requirement uses conditional modal verbs (`would`, `could`, `can`) rather than prescriptive directives or declarative present tense, outcomes from other cities are framed as indicative evidence rather than dispositive proof, and sentences use direct connectors (`and`, `whereas`, or a new sentence).

WRITING STANDARD (8th-grade reading level across BOTH `tldr` and `insights` / town hall read-aloud test):
- Both the `tldr` and every supporting `insights` paragraph must be written at an 8th-grade reading level for regular citizens in {place_line}. Every sentence must be easy to understand on first hearing when read aloud at a public town hall meeting—use short, clear sentences that explain one relationship at a time rather than packing multiple statistics into long run-on clauses.
- Ban consultant jargon and inflated phrasing, and do not echo prompt instructions in the prose (for example, write 'Past examples of similar programs show...' rather than restating the category standard).
- Use active, concrete verbs ('fund', 'build', 'pave', 'inspect', 'contract', 'hire', 'enforce') and explain exact City Code sections, department names, and budget numbers in plain everyday words.
- Stay administratively neutral and follow the tone rules of the Publication Standard above.

OUTPUT FORMAT: Return a JSON object matching this schema:
{{
  "problem_definition": {{
    "insights": [
      {{"category": "problem_sizing", "text_with_citations": "..."}},
      {{"category": "resident_impact", "text_with_citations": "..."}}
    ],
    "tldr": "..."
  }},
  "recommendation": {{
    "insights": [
      {{"category": "implementation_path", "text_with_citations": "..."}},
      {{"category": "reference_point", "text_with_citations": "..."}},
      {{"category": "cost_estimate", "text_with_citations": "..."}}
    ],
    "tldr": "..."
  }},
  "participant_quote_ids": [1, 3]
}}
"""


async def generate_text_in_parallel(
    model: genai_model.GenaiModel,
    prompt_obj_list: list[dict],
    max_concurrent_calls: int | None = None,
) -> pd.DataFrame:
  """Utility function to simplify calls to process_prompts_concurrently."""
  response_df, _, _, _ = await model.process_prompts_concurrently(
      prompt_obj_list,
      lambda x, _: x["text"],
      max_concurrent_calls=max_concurrent_calls,
  )
  return response_df
