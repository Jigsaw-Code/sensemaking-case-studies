# **Sensemaker Library (What Could BG Be? Archive)**

> **Note:** This library is **archival, unmaintained, and read-only**, preserved as part of the **What Could BG Be?** case study. For the full case study walkthrough and methodology overview, see [../README.md](../README.md) and [../summarization.md](../summarization.md). For Jigsaw's active Sensemaking AI tools, visit [Jigsaw Sensemaking Tools](https://github.com/Jigsaw-Code/sensemaking-tools).

## **Setup & GCloud Authentication**

First make sure you have `npm` installed, then install dependencies from `wcbgb/` (or this directory):

```shell
npm install
```

A Google Cloud project is required to control quota and access when using the default Vertex AI models ([gcloud CLI install guide](https://cloud.google.com/sdk/docs/install-sdk#deb)):

```shell
gcloud config set project <your project name here>
gcloud auth application-default login
```

## **Programmatic Usage**

Core entry points and types can be inspected directly in source:
* [`src/sensemaker.ts`](./src/sensemaker.ts) — main `Sensemaker` class (`learnTopics`, `categorizeComments`, `summarize`)
* [`src/types.ts`](./src/types.ts) — comment, topic, vote tally, and summary schemas/types
* [`src/models/model.ts`](./src/models/model.ts) — `Model` abstraction for integrating custom LLMs

Example summarizing Seattle’s $15 Minimum Wage conversation:

```ts
// Set up Sensemaker to use the default Vertex model and authentication info.
const mySensemaker = new Sensemaker({
  defaultModel: new VertexModel(
    "myGoogleCloudProject123",
    "us-central1",
  ),
});

// Note: getCommentsFromCsv is illustrative; see runner-cli/runner_utils.ts for CSV helpers.
// CSV data source: https://github.com/compdemocracy/openData/tree/master/15-per-hour-seattle
const comments: Comment[] = getCommentsFromCsv("./comments.csv");

// Learn what topics were discussed and print them out.
const topics = await mySensemaker.learnTopics(
  comments,
  // Should include subtopics:
  true,
  // No pre-existing topics:
  undefined,
  // Additional context:
  "This is from a conversation on a $15 minimum wage in Seattle"
);
console.log(topics);

// Summarize the conversation and print the result as Markdown.
const summary = await mySensemaker.summarize(
  comments,
  SummarizationType.AGGREGATE_VOTE,
  topics,
  // Additional context:
  "This is from a conversation on a $15 minimum wage in Seattle"
);
console.log(summary.getText("MARKDOWN"));
```

## **CLI Runners**

Three CLI scripts are included under [`./runner-cli/`](./runner-cli/):

* [`./runner-cli/runner.ts`](./runner-cli/runner.ts): takes a CSV representing a conversation and outputs Markdown, grounded CSV, JSON, and HTML summary files (with hover-over comment/vote citations in HTML).
* [`./runner-cli/categorization_runner.ts`](./runner-cli/categorization_runner.ts): takes a CSV representing a conversation and outputs a CSV with comments categorized into topics and subtopics.
* [`./runner-cli/advanced_runner.ts`](./runner-cli/advanced_runner.ts): takes a categorized CSV and outputs three JSON files (`topic-stats`, `comments-with-scores`, and `summary`) used by the interactive web report.

Input CSV files must contain `comment_text` and `comment-id` columns. For deliberations without group breakdowns, vote counts should be set in `agrees`, `disagrees`, and `passes` (or `0` if unavailable). For deliberations with group breakdowns, provide `{group_name}-agree-count`, `{group_name}-disagree-count`, and `{group_name}-pass-count`.

## **Testing**

From this directory (`wcbgb/library/`), unit tests can be run with:
* `npm test` (single run)
* `npm run test-watch` (watch mode)

## **Cloud Vertex Terms of Use**

This library is designed to leverage Vertex AI, and usage is subject to the [Cloud Vertex Terms of Service](https://cloud.google.com/terms/service-terms) and the [Generative AI Prohibited Use Policy](https://policies.google.com/terms/generative-ai/use-policy).
