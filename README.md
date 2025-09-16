# Multi-LLM Research Agent for LGBTQ+ Safety Analysis

This project deploys a team of autonomous AI agents to research, analyze, and synthesize information regarding global LGBTQ+ safety. It uses a multi-agent framework to delegate tasks and generate comprehensive reports based on real-time data.

The system is containerized with Docker for easy, one-command setup and perfect reproducibility of the research.

## Core Features

-   **Modular Agent Design:** Specialized agents for research, analysis, and writing, allowing for complex task decomposition.
-   **Dynamic Tooling:** Agents are equipped with tools like web search to gather up-to-the-minute information.
-   **Centralized Configuration:** All agent parameters, models, and prompts are managed via a `config.yaml` file for easy tuning.
-   **Reproducible Environment:** Docker and Docker Compose ensure the project runs identically everywhere, solving dependency and environment issues.

---

## Getting Started

### Prerequisites

-   Docker and Docker Compose
-   An API key from an LLM provider (e..g, OpenAI, Anthropic, Groq).

### How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/QueerAgent1/iqsf-multi-llm-research.git
    cd iqsf-multi-llm-research
    ```

2.  **Configure your API Key:**
    Create a `.env` file by copying the example. This file is kept private and is ignored by Git.
    ```bash
    cp .env.example .env
    ```
    Now, open the `.env` file with a text editor and add your LLM provider's API key:
    ```
    API_KEY="sk-your-secret-key-here"
    ```

3.  **Build and Run with Docker Compose:**
    This single command will build the Docker image, install all dependencies, and run the entire research process:
    ```bash
    docker-compose up --build
    ```
    To run in the background (detached mode), use `docker-compose up --build -d`.

### Expected Output

The script will run in your terminal, showing the thought process, actions, and observations of each agent. Upon completion, a final report will be saved to the `/data` directory as a Markdown file (e.g., `final_report.md`). A `research_log.log` file will also be generated with detailed execution logs for debugging.

---

## Project Structuredebugging.

---

## Project Structure
iqsf-multi-llm-research/
│
├── .github/ # GitHub Actions workflows (CI/CD)
├── agents/ # Contains the logic for each specialized AI agent
│ └── researcher.py
├── config/ # Centralized configuration
│ └── config.yaml
├── data/ # (Gitignored) Default location for inputs and outputs
│ └── .gitkeep
├── notebooks/ # Jupyter notebooks for experimentation
│ └── research.ipynb
├── prompts/ # Manages prompt templates for all agents
│ └── researcher_prompts.py
├── tools/ # Defines tools agents can use (e.g., web search)
│ └── search_tools.py
├── utils/ # Helper functions and utilities
│ └── file_utils.py
│
├── .dockerignore # Specifies files to ignore in the Docker build
├── .env.example # Example environment file
├── .gitignore # Specifies files for Git to ignore
├── docker-compose.yml # Orchestrates the Docker container
├── Dockerfile # Defines the Docker container environment
├── main.py # Main entry point to run the agentic workflow
├── README.md # This file
└── requirements.txt # Python dependencies
