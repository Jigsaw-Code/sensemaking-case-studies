# Sensemaking AI Case Studies

> **Note:** This repository contains the historical, unmaintained code and data for the **What Could BG Be?** and **We the People** case studies, provided solely for transparency and replication purposes. For our active, maintained tools and the latest Sensemaking AI pipeline (including tools for Adaptive Interviewing, Topic Discovery, Quote Extraction, and Interactive Reports), please visit the [Jigsaw Sensemaking Tools](https://github.com/Jigsaw-Code/sensemaking-tools) repository.

## Overview

Jigsaw explored the application of AI to public opinion research through two proofs of concept, [What Could BG Be?](https://jigsaw.google/our-work/reimagining-the-town-hall-meeting/) and [We the People](https://jigsaw.google/our-work/scaling-traditional-focus-groups/). The code within these case studies demonstrates early experimental methods for transforming raw community feedback into clear, digestible insights using Google’s Gemini models. It showcases techniques such as clustering public opinion, extracting representative quotes, and generating interactive web interfaces to explore summarized data.

*   **What Could BG Be?** was a month-long public consultation crowdsourcing ideas to inform the 25-year plan for Warren County, Kentucky. [Visit the case study section of this repository to learn more about the project and explore the code base](case_studies/wcbgb/).

*   **We the People** was a nationally representative public opinion research initiative, combining the depth of focus groups with the breadth of polls with an aim to identify areas of broad public agreement on the topic of freedom and equality in America. This repository contains the official replication code for this project. [Visit the case study section of this repository to learn more about the project and explore the code base](case_studies/wtp/).

---

## Directory Architecture

```text
.
├── case_studies/
│   ├── wcbgb/                 <-- What Could BG Be? case study code
│   └── wtp/                   <-- We the People case study code
├── LICENSE                    <-- Apache-2.0 License
├── CODE_OF_CONDUCT.md         <-- Community Conduct Guidelines
├── CONTRIBUTING.md            <-- Contribution and CLA Guidelines
├── SECURITY.md                <-- Vulnerability Reporting Instructions
└── README.md                  <-- Project Documentation & Overview
```

---

## Running the Historical Code

> **Disclaimer:** The code in this repository is unmaintained and provided for historical reference. Running this code may require older dependency versions, deprecated API endpoints, or manual adjustments. It may not work with newer models and APIs. 

### Prerequisites
Running the historical code across these case studies requires several core dependencies depending on which pipeline steps you are executing (e.g., data processing vs. visualization):

*   **Python 3 & pip3:** Required for data processing, topic modeling, and running the simulated jury scripts.
*   **Node.js & npm:** (Version 18.0.0 or higher) Required for building and viewing the interactive frontend visual reports.
*   **API Access:** You will need a **Gemini API Key** and/or a **Google Cloud Project** setup (with the `gcloud` CLI authenticated) to execute the AI modeling routines.

Please refer to the individual `README.md` files within `case_studies/wcbgb/` and `case_studies/wtp/` for detailed, project-specific setup commands and requirements.

---

## Licensing & Compliance
This project is licensed under the Apache License 2.0. Every source file must include the standard Apache-2.0 license header at the top of the file to comply with automated OSPO `Cross` security and policy scanners.