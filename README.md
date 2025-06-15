# Automated Book Publication Workflow

This project implements an automated workflow for fetching, processing, and versioning book content, as specified for evaluation purposes. It scrapes content from a web URL, applies AI-driven transformations (mocked), incorporates human-in-the-loop review (simulated), and manages content versions using ChromaDB with an RL-based search algorithm for retrieval.

---

## Project Overview

The system fulfills the following requirements:

* **Scraping & Screenshots:** Fetches text and saves screenshots from `https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_1` using [Playwright](https://playwright.dev/).
* **AI Writing & Review:** Simulates LLM (e.g., Gemini) for spinning (AI Writer), refining (AI Reviewer), and finalizing (AI Editor) content.
* **Human-in-the-Loop:** Supports multiple iterations of simulated human review for writers, reviewers, and editors.
* **Agentic API:** Manages seamless content flow between scraping, AI processing, human review, and versioning.
* **Versioning & Consistency:** Stores versions in [ChromaDB](https://www.trychroma.com/) and retrieves them using a Q-learning-based RL search algorithm.

---

## Files

* `book_publication_workflow.py`: Main Python script implementing the workflow.
* `screenshots/`: Folder containing screenshots of the scraped webpage (e.g., `screenshot_20250615_222145.png`).
* `final_book_content.txt`: Output file with the final processed content.
* `book_workflow_demo.mp4`: Demo video showing the code execution and outputs.

---

## Prerequisites

To run the project, you need:

* Python 3.8 or higher
* Required libraries: `Playwright`, `ChromaDB`
* Playwright browsers installed

---

## How to Run

1.  **Install dependencies:**
    ```bash
    pip install playwright chromadb
    playwright install
    ```
2.  **Run the script:**
    ```bash
    python book_publication_workflow.py
    ```

**Outputs:**
* A screenshot is saved in the `screenshots/` folder.
* The final processed content is saved in `final_book_content.txt`.
* Versions are stored in ChromaDB (accessible via the script).

---

## Demo Video

Watch the demo video here: [`book_workflow_demo.mp4`](book_workflow_demo.mp4)

The video shows:
* The `book_publication_workflow.py` code in VS Code.
* Execution of the script in the terminal.
* Output files (`screenshots/` and `final_book_content.txt`).

---

## Notes

* The AI components (writer, reviewer, editor) are mocked to simulate LLM behavior, as real LLM integration requires API access.
* The human review is simulated but designed to support real human input via a UI in a production setting.
* The RL search algorithm uses Q-learning for ranking versions, with potential for enhancement using advanced state representations.
* This project is for evaluation purposes only, as specified, and is not intended for commercial use.

---

## Author

**GitHub:** [annie-2314](https://github.com/annie-2314)

Created for evaluation, submission deadline: June 22, 2025