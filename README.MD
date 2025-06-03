# AI Research Assistant

A command-line research assistant built with LangChain and Google Gemini (Gemini‐2.5‐flash). Given a user query, it can:

1. Invoke external tools (e.g., web search, Wikipedia lookup, saving results to file).  
2. Parse the LLM’s output into a structured JSON format.  
3. Display or save a final summary of the research results.

---

## Table of Contents

- [Features](#features)  
- [Prerequisites](#prerequisites)  
- [Environment Setup](#environment-setup)  
- [Installation](#installation)  
- [Project Structure](#project-structure)  
- [Usage](#usage)  
- [Example](#example)  
- [Tools](#tools)  
- [Troubleshooting](#troubleshooting)  
- [License](#license)

---

## Features

- **Google Gemini (Gemini‐2.5‐flash)** for natural-language research queries.  
- **Tool‐calling agent** that can:
  - 🔍 Search the web (`search_tool`)  
  - 📚 Lookup Wikipedia (`wiki_tool`)  
  - 💾 Save results into a text or JSON file (`save_tool`)  
- Structured output wrapped in a Pydantic model (`ResearchResponse`), containing:
  - `topic`  
  - `summary`  
  - `sources`  
  - `tools_used`  

---

## Prerequisites

- Python 3.9+  
- A valid Google Cloud API key with access to Gemini.  
- (Optional) A `.env` file for any additional environment variables.  

---

## Environment Setup

1. **Clone this repository** (or copy the files) into your local machine.  
2. **Create a virtual environment** (strongly recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate         # macOS/Linux
   .\venv\Scripts\activate          # Windows
