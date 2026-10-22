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

from unittest import mock

from panoramic_insights import phase_2_run_deep_research
import requests


def test_resolve_redirect_url_success():
  mock_response = mock.MagicMock()
  mock_response.status_code = 302
  mock_response.headers = {"Location": "https://example.com/target_article"}

  with mock.patch("requests.get", return_value=mock_response):
    resolved = phase_2_run_deep_research.resolve_redirect_url(
        "https://vertexaisearch.cloud.google.com/grounding-api-redirect/123"
    )
    assert resolved == "https://example.com/target_article"


def test_resolve_redirect_url_failure():
  with mock.patch(
      "requests.get", side_effect=requests.RequestException("Connection error")
  ):
    url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/123"
    resolved = phase_2_run_deep_research.resolve_redirect_url(url)
    assert resolved == url


def test_unshorten_deep_research_urls():
  sample_text = (
      "1."
      " [6amcity.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc12345)\n2."
      " [paylesspavingtn.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/def67890)"
  )

  def mock_resolver(url, _timeout=5):
    if "abc12345" in url:
      return "https://6amcity.com/article/1"
    if "def67890" in url:
      return "https://paylesspavingtn.com/services"
    return url

  with mock.patch(
      "panoramic_insights.phase_2_run_deep_research.resolve_redirect_url",
      side_effect=mock_resolver,
  ):
    result = phase_2_run_deep_research.unshorten_deep_research_urls(sample_text)
    assert "1. https://6amcity.com/article/1" in result
    assert "2. https://paylesspavingtn.com/services" in result


def test_get_args_concurrency_flags():
  with mock.patch(
      "sys.argv",
      [
          "phase_2_run_deep_research.py",
          "--input_csv",
          "in.csv",
          "--output_csv",
          "out.csv",
          "--max_concurrency",
          "7",
      ],
  ):
    args = phase_2_run_deep_research.get_args()
    assert args.max_concurrency == 7



def _make_interaction(steps, output_text=""):
  """Builds a mock interaction whose to_dict() returns the given steps."""
  interaction = mock.MagicMock()
  interaction.to_dict.return_value = {"steps": steps}
  interaction.output_text = output_text
  return interaction


def test_extract_interaction_output_text_joins_all_model_output_steps():
  """A report split across model_output steps must be reassembled in full.

  Deep Research streams long reports back as several `model_output` steps.
  `interaction.output_text` exposes only the final step, which drops the
  earlier sections of the report.
  """
  steps = [
      {"type": "user_input", "content": [{"text": "the research prompt"}]},
      {"type": "model_output", "content": [{"text": "# Section 1\n"}]},
      {"type": "model_output", "content": [{"text": "## Section 2\n"}]},
      {"type": "model_output", "content": [{"text": "## Section 3\n"}]},
  ]
  interaction = _make_interaction(steps, output_text="## Section 3\n")

  result = phase_2_run_deep_research.extract_interaction_output_text(
      interaction
  )

  assert result == "# Section 1\n## Section 2\n## Section 3\n"


def test_extract_interaction_output_text_ignores_non_text_content():
  """Generated charts and other non-text parts must not break extraction."""
  steps = [
      {
          "type": "model_output",
          "content": [
              {"type": "image", "data": "iVBORw0KGgo="},
              {"text": "## Section 4\n"},
          ],
      },
  ]
  interaction = _make_interaction(steps)

  result = phase_2_run_deep_research.extract_interaction_output_text(
      interaction
  )

  assert result == "## Section 4\n"


def test_extract_interaction_output_text_excludes_user_input():
  """The prompt echoed back as a user_input step must not enter the report."""
  steps = [
      {"type": "user_input", "content": [{"text": "PROMPT TEXT"}]},
      {"type": "model_output", "content": [{"text": "REPORT TEXT"}]},
  ]
  interaction = _make_interaction(steps)

  result = phase_2_run_deep_research.extract_interaction_output_text(
      interaction
  )

  assert result == "REPORT TEXT"


def test_extract_interaction_output_text_falls_back_to_output_text():
  """With no textual model_output step, fall back to the SDK property."""
  interaction = _make_interaction([], output_text="fallback report")

  result = phase_2_run_deep_research.extract_interaction_output_text(
      interaction
  )

  assert result == "fallback report"
