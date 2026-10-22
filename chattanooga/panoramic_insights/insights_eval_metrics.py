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

from panoramic_insights import insights_prompts
import pandas as pd
from src.evals.eval_metrics import EvaluationMetrics, PointwiseEvaluationMetric

insights_criteria = insights_prompts.get_insight_evaluation_questions()


class InsightsPointwiseEvaluationMetric(PointwiseEvaluationMetric):
  """Custom pointwise metric to evaluate exploded insights grouped by opinion."""

  def get_evaluation_data(self, input_data: pd.DataFrame) -> pd.DataFrame:
    # Build a formatted string for each individual insight
    temp_df = input_data.copy()
    if "proposal" not in temp_df.columns:
      raise ValueError("Input CSV must contain a 'proposal' column.")

    temp_df["formatted_insight"] = temp_df.apply(
        lambda row: (
            f"Category: {row.get('category', 'N/A')}\nFull Insight:"
            f" {row.get('full_insight', 'N/A')}\n"
            f"Citation: {row.get('citation_text', 'N/A')}"
            f" ({row.get('citation_link', 'N/A')})"
        ),
        axis=1,
    )

    # Group by topic and proposal, and concatenate all formatted insights
    grouped = (
        temp_df.groupby(["topic", "proposal"])["formatted_insight"]
        .apply(lambda x: "\n\n---\n\n".join(x))
        .reset_index(name="response")
    )

    # Create the final evaluation dataset
    evaluation_data = pd.DataFrame()
    evaluation_data["proposal"] = grouped["proposal"]
    evaluation_data["topic"] = grouped["topic"]
    evaluation_data["response"] = grouped["response"]
    return evaluation_data


class InsightsEvaluationMetrics(EvaluationMetrics):
  """Custom evaluation metrics for exploded insights."""

  def __init__(self):
    name = "Insights"
    additional_input_variables = ["proposal", "topic"]
    # Using 'insight' as response_name as it exists in processed data to bypass validation
    response_name = "insight"

    custom_pointwise_metric = InsightsPointwiseEvaluationMetric(
        name=name,
        criteria=insights_criteria,
        input_variables=additional_input_variables,
        response_name=response_name,
    )

    super().__init__(
        name=name,
        criteria=insights_criteria,
        additional_input_variables=additional_input_variables,
        response_name=response_name,
        group_by_config=None,
        pointwise_metric=custom_pointwise_metric,
    )
    self.pairwise_metric = None


INSIGHTS_METRICS = InsightsEvaluationMetrics()
