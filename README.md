# LangGraph Multi-Agent AI Research System

An AI-powered research assistant built using LangGraph, LangChain, OpenRouter, and Streamlit.

## Features

- Planner Agent – Creates a structured research plan.
- Research Agent – Generates research findings.
- Writer Agent – Produces a structured Markdown report.
- Download generated reports as Markdown files.

## Tech Stack

- Python
- LangGraph
- LangChain
- OpenRouter
- Streamlit

## Setup

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file and add your OpenRouter API key:

```env
OPENROUTER_API_KEY=your_api_key_here
```

5. Run the application:

```bash
streamlit run app.py
```

## Security

Never upload your `.env` file or API keys to GitHub.

## Project Type

AI / Generative AI / Multi-Agent Systems