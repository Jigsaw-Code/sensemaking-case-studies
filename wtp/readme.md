# **We The People**

[We The People](https://wethepeople-250.org/) was a national public opinion research initiative, combining the depth of focus groups with the breadth of polls with an aim to identify areas of broad public agreement on America’s founding ahead of the nation’s 250th anniversary. The product of a partnership between the [Napolitan Institute](https://napolitaninstitute.org) and [Jigsaw](https://jigsaw.google/), 

We The People invited thousands of Americans from every corner of the country to share their beliefs, opinions, and experiences on freedom and equality.

The conversation occurred in three phases:

* **Individual Expression**: In the first round, participants shared personal reflections to open-ended questions. Gemini was used to generate tailored AI follow-up questions to help participants go a level deeper in a personalized dialogue.  
* **Collective Reflection**: In the second round, participants explored opinions that our Sensemaking AI identified from the responses in round one. They were invited to respond to selected quotes and rank these opinions. This round enabled them to see themselves in the national opinion landscape, and helped draw out greater nuance in their thinking.  
* **Conversation Validation**: In the third round, participants voted on [AI-generated declarative statements that were likely to receive broad agreement](https://medium.com/jigsaw/research-note-can-ai-help-thousands-of-people-discover-shared-values-2611366ccdf4), based on participants previous responses.

   
The tools here illustrate methods for:

* **Data Preparation**: reformats and moderates raw exports from Qualtrics into a consistent structure for analysis.  
* **Topic Modeling and Quote Extraction**: identifies discussion topics and extracts representative quotes from the participant dialogue using Gemini.  
* **Quote Ranking**: applies classifiers to score quotes on attributes like reasoning and curiosity, helping identify high-quality contributions.  
* **Interactive Visualization**: automatically creates an interactive opinion map for participants to explore responses from Round 1\.  
* **Predictive Agreement**: generates and ranks propositions using a Simulated Jury technique to identify consensus.

You can [learn more about We the People on Jigsaw’s website](https://jigsaw.google/our-work/scaling-traditional-focus-groups/).

## **The Process**

This walkthrough documents the scripts used and steps taken for data processing and analysis for We the People. The code here is provided for transparency purposes only and may not work with newer models.

### **Setup**

First make sure you have `python3`, `pip3`, Node.js, and `npm` installed. You will also need a Gemini API Key.

Install the Python dependencies by running:

```shell
pip3 install -r requirements.txt
```

### **Phase 1: Individual Expression (Post-Round 1 Processing)**

After Round 1 is complete, run the following script to transform the raw survey export, score data quality, and apply Perspective API scores for moderation.

```shell
bash post_round_1_processing.sh \
  --input_csv <PATH_TO_R1_RAW_EXPORT.csv> \
  --output_dir <OUTPUT_DIR> \
  --api_key "$PERSPECTIVE_API_KEY" \
  --round_1_question_response_text "Q1,Q35,Q36" \
  --round_1_follow_up_questions "Q1FU,Q2FU,Q3FU" \
  --round_1_follow_up_question_response_text "Q23,Q37,Q39"
```

### **Phase 2: Collective Reflection (Pre- and Post-Round 2 Processing)**

Before Round 2 begins, first run the categorization script to identify topics and deduplicate quotes:

```shell
python3 categorization_runner.py \
  --model_name gemini-2.5-pro \
  --vertex_location global \
  --vertex_project "conversation-ai-experiments" \
  --subject "freedom and equality" \
  --output_dir <OUTPUT_DIR>/categorization_outputs \
  --input_file <OUTPUT_DIR>/moderated.csv \
  --topic_depth 2 \
  --topics "Defining Freedom,Defining Equality,Connection to Historical Ideals,Achieving Freedom and Equality" \
  --run_autoraters
```

Then, run the pre-processing script to extract quotes, and run clustering and opinion quality evaluations:

```shell
bash pre_round_2_processing.sh \
  --input_csv <OUTPUT_DIR>/moderated.csv \
  --output_dir <OUTPUT_DIR>/r2_prep \
  --perspective_api_key "$PERSPECTIVE_API_KEY"
```

To create the visualization for Round 2, you can use the `r2_visualization` package. First, you will need to copy the generated data into the visualization's data directory and convert it to JSON format:

```shell
cp <OUTPUT_DIR>/r2_prep/quotes/opinion_counts.csv r2_visualization/src/data/opinions.csv
cp <OUTPUT_DIR>/r2_prep/quotes/quotes_for_visual.csv r2_visualization/src/data/quotes.csv
cd r2_visualization
npm install
node tasks/csv-to-json.js
npm run build
```

After Round 2 concludes, process the results and prepare them for moderation:

```shell
bash post_round_2_processing.sh \
  --input_csv <PATH_TO_R2_RAW_EXPORT.csv> \
  --output_dir <OUTPUT_DIR>/r2_post \
  --perspective_api_key "$PERSPECTIVE_API_KEY"
```

### **Phase 3: Conversation Validation (Pre-Round 3 Processing)**

Before Round 3 begins, generate distinct propositions using data from the first two rounds.

```shell
bash pre_round_3_processing.sh \
  --r1_input_file <OUTPUT_DIR>/moderated.csv \
  --r2_input_file <OUTPUT_DIR>/r2_post/processed.csv \
  --output_dir <OUTPUT_DIR>/r3_prep \
  --api_key "$GEMINI_API_KEY" \
  --proposition_count 5
```

After generating the initial propositions, run the proposition refinement script to perform simulated juries, deduplication, and generate nuanced propositions:

```shell
python3 -m proposition_refinement.main \
  --input_pkl <OUTPUT_DIR>/r3_prep/wtp_s1.pkl \
  --output_pkl <OUTPUT_DIR>/r3_prep/wtp_s1_refinement.pkl \
  --final_propositions_per_topic 4 \
  --propositions_per_opinion 3 \
  --num_nuanced_propositions 15 \
  --verbose \
  --simulated_jury_model_name gemini-2.5-flash-lite \
  --gemini_api_key "$GEMINI_API_KEY" \
  --deduplication_method rank_filling \
  --run_pav_selection \
  --processed_r2_data <OUTPUT_DIR>/r2_post/processed.csv
```

Finally, simplify the refined propositions:

```shell
python3 -m proposition_simplification_runner \
  --input_csv <OUTPUT_DIR>/r3_prep/propositions_wtp_s1.csv \
  --output_csv <OUTPUT_DIR>/r3_prep/simplified_propositions.csv \
  --project conversation-ai-experiments \
  --location global \
  --model_name gemini-2.5-pro
```

## **How it Works**

### **Topic Modeling and Quote Extraction**

The categorization runner uses Gemini to identify discussion topics and extract representative quotes. It provides options to discover topics dynamically or use predefined topics. During We the People, this allowed us to run the pipeline multiple times before settling on a final set of topics to use for round 2\.

### **Constructive Quality Scoring**

Quotes are scored to surface the most constructive contributions, prioritizing attributes like reasoning and curiosity. These help identify the most constructive quotes, which are highlighted first in the interactive report.

### **Interactive Visualization**

The visualization process generates an interactive map of the opinions discovered from the first round, enabling participants in Round 2 to visually explore and understand the scale breadth of responses.

### **Proposition Generation & Simulated Juries**

The [simulated jury technique](https://medium.com/jigsaw/research-note-can-ai-help-thousands-of-people-discover-shared-values-2611366ccdf4) is used to predict agreement across the participant pool, generating nuanced statements that reflect areas of shared agreement within the varied opinion landscape. These statements are rank-ordered to identify those most likely to receive broad agreement.

### **Testing**

Unit tests can be run to ensure all components are functioning correctly. From this directory, run:

```shell
python3 -m pytest .
```