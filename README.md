\# GameWise AI 🎮



GameWise AI is an intelligent Steam game recommendation system that converts natural-language requests into grounded and explainable game recommendations.



Users can describe the kind of game they want using budget, platform, review quality, release year, play mode, genre, or mood. GameWise applies structured filters, semantic retrieval, metadata-aware ranking, and optional grounded answer generation to return relevant results.



\---



\## Features



\* Natural-language game search

\* Structured filtering by:



&#x20; \* Price

&#x20; \* Positive review percentage

&#x20; \* Platform

&#x20; \* Release year

&#x20; \* Free-to-play status

&#x20; \* Single-player, co-op, or multiplayer support

\* Semantic retrieval using Sentence Transformers

\* Field-aware concept matching using genres, tags, categories, and descriptions

\* Hybrid ranking with review-quality and play-mode signals

\* Clarification handling for overly broad requests

\* Strict no-result handling without silently weakening user requirements

\* Grounded recommendation summaries

\* Local summary fallback when no OpenAI API key is available

\* Interactive Streamlit interface

\* Search history and game shortlist

\* Recommendation sorting and CSV export

\* Optional developer ranking details



\---



\## Application Screenshots



\### Home



!\[GameWise AI home page](docs/images/home.png)



\### Recommendations



!\[GameWise AI recommendations](docs/images/recommendations.png)



\### Clarification handling



!\[GameWise AI clarification message](docs/images/clarification.png)



\### No-result handling



!\[GameWise AI no-result message](docs/images/no\_results.png)



\### Developer ranking details



!\[GameWise AI developer mode](docs/images/developer\_mode.png)



\---



\## System Architecture



```mermaid

flowchart TD

&#x20;   A\[User natural-language query] --> B\[Query parser]

&#x20;   B --> C\[Structured filters]

&#x20;   C --> D\[Candidate game set]

&#x20;   D --> E\[Sentence Transformer query embedding]

&#x20;   E --> F\[Semantic similarity]

&#x20;   D --> G\[Field-aware concept matching]

&#x20;   D --> H\[Play-mode preference scoring]

&#x20;   D --> I\[Review-quality scoring]

&#x20;   F --> J\[Hybrid ranking]

&#x20;   G --> J

&#x20;   H --> J

&#x20;   I --> J

&#x20;   J --> K\[Top-k Steam games]

&#x20;   K --> L\[Grounded recommendation generator]

&#x20;   L --> M\[Interactive Streamlit interface]

```



\---



\## Dataset



The project uses a local dataset containing 1,495 Steam games.



The cleaned dataset contains information such as:



\* Game name

\* Price

\* Free-to-play status

\* Release year

\* Positive and negative review counts

\* Positive review percentage

\* Genres

\* Steam categories

\* Community tags

\* Supported platforms

\* Game descriptions

\* Steam store URL



The data-cleaning pipeline creates:



```text

data/processed/steam\_games\_cleaned.csv

data/processed/data\_quality\_report.json

```



\---



\## Embeddings



GameWise uses:



```text

sentence-transformers/all-MiniLM-L6-v2

```



The model creates a 384-dimensional embedding for each game.



Generated files:



```text

data/processed/game\_embeddings.npy

data/processed/game\_embedding\_index.csv

```



The embedding model and processed search data are cached so they are loaded only once during each Python process.



\---



\## Hybrid Retrieval



GameWise combines several ranking signals.



\### Semantic similarity



The user query is embedded with the same Sentence Transformer model used for the game records.



Cosine similarity is calculated between the query embedding and candidate game embeddings.



\### Concept relevance



Requested concepts such as survival, relaxing, psychological horror, strategy, farming, RPG, racing, or shooter are matched against:



\* Genres and tags as strong evidence

\* Official Steam categories as supporting evidence

\* Game descriptions as weaker evidence



This prevents incidental description wording from receiving the same score as an explicit genre or tag match.



\### Play-mode preference



Official Steam categories are used to verify:



\* Single-player

\* Co-op

\* Multiplayer

\* PvP-related features



The score can distinguish between pure single-player games and games that also heavily emphasize multiplayer or PvP.



\### Review quality



Positive review percentage is included as a ranking-quality signal.



\### Hybrid score



The weighting changes according to the information contained in the query.



For a query containing both a concept and a play mode:



```text

Hybrid score =

0.55 × normalized semantic similarity

\+ 0.20 × concept relevance

\+ 0.15 × play-mode preference

\+ 0.10 × review quality

```



For a concept-only query:



```text

Hybrid score =

0.65 × normalized semantic similarity

\+ 0.25 × concept relevance

\+ 0.10 × review quality

```



For a play-mode-only query:



```text

Hybrid score =

0.70 × normalized semantic similarity

\+ 0.20 × play-mode preference

\+ 0.10 × review quality

```



For other sufficiently specific queries:



```text

Hybrid score =

0.90 × normalized semantic similarity

\+ 0.10 × review quality

```



\---



\## Grounded Generation



After retrieval, GameWise can generate a concise recommendation summary using only the retrieved Steam records.



The generation layer is instructed not to invent:



\* Prices

\* Review scores

\* Platforms

\* Release years

\* Steam URLs

\* Unsupported game characteristics



When an API key is unavailable or the model request fails, GameWise returns a local grounded summary instead.



\---



\## Evaluation



The retrieval system was evaluated using 11 representative queries covering:



\* Cooperative survival

\* Relaxing single-player games

\* Psychological horror

\* Tactical strategy

\* Story-rich RPGs

\* Racing

\* Farming simulation

\* Recent multiplayer shooters

\* Broad-query clarification

\* Impossible-condition handling



Current evaluation results:



\* Search behavior pass rate: \*\*11/11\*\*

\* Hard-filter accuracy: \*\*100%\*\*

\* Valid-title rate: \*\*100%\*\*

\* Duplicate-free rate: \*\*100%\*\*

\* Metadata-term relevance@5: \*\*100%\*\*



These results apply to the defined evaluation cases and the current 1,495-game dataset. They should not be interpreted as universal accuracy across the complete Steam catalogue.



Detailed results are available in:



```text

evaluation/evaluation\_results.csv

evaluation/evaluation\_summary.md

```



\---



\## Example Queries



```text

a cooperative survival game under $20 with at least 80% positive reviews

```



```text

a relaxing single-player casual game under $15

```



```text

a free psychological horror game

```



```text

a turn-based tactical strategy game under $20 for Linux

```



```text

a story-rich RPG with at least 90% positive reviews

```



\---



\## Installation



\### 1. Clone the repository



```powershell

git clone https://github.com/momo840505/gamewise-ai.git

cd gamewise-ai

```



\### 2. Create a virtual environment



```powershell

python -m venv .venv

```



\### 3. Activate the environment



```powershell

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\\.venv\\Scripts\\Activate.ps1

```



\### 4. Install dependencies



```powershell

python -m pip install --upgrade pip

python -m pip install -r requirements.txt

```



\### 5. Configure optional generation settings



Copy the example environment file:



```powershell

Copy-Item .env.example .env

```



Add an API key only to the local `.env` file:



```text

OPENAI\_API\_KEY=

OPENAI\_MODEL=

```



The `.env` file must not be committed to GitHub.



GameWise works without an API key by using its local grounded-summary fallback.



\---



\## Run the Application



```powershell

python -m streamlit run .\\app.py

```



Open:



```text

http://localhost:8501

```



\---



\## Run the Tests



```powershell

python -m pytest -q

```



Current test result:



```text

16 passed

```



\---



\## Run the Evaluation



Run the evaluation module from the project root:



```powershell

python -m scripts.evaluate\_retrieval

```



Do not run it using:



```text

python scripts/evaluate\_retrieval.py

```



The module form ensures that the project package imports are resolved correctly.



\---



\## Project Structure



```text

gamewise-ai/

├── app.py

├── requirements.txt

├── README.md

├── .env.example

├── data/

│   ├── raw/

│   │   └── steam\_top\_games\_2026.csv

│   └── processed/

│       ├── steam\_games\_cleaned.csv

│       ├── data\_quality\_report.json

│       ├── game\_embeddings.npy

│       └── game\_embedding\_index.csv

├── docs/

│   └── images/

│       ├── home.png

│       ├── recommendations.png

│       ├── clarification.png

│       ├── no\_results.png

│       └── developer\_mode.png

├── evaluation/

│   ├── evaluation\_cases.json

│   ├── evaluation\_results.csv

│   └── evaluation\_summary.md

├── scripts/

│   ├── build\_embeddings.py

│   ├── clean\_dataset.py

│   ├── evaluate\_retrieval.py

│   ├── generate\_answer.py

│   └── hybrid\_search.py

└── tests/

&#x20;   ├── test\_generate\_answer.py

&#x20;   └── test\_hybrid\_search.py

```



\---



\## Limitations



\* The dataset contains only the selected top 1,495 Steam games.

\* Prices and availability represent a dataset snapshot rather than live Steam data.

\* Steam community tags may contain noisy or unexpected labels.

\* Concept rules are manually defined.

\* The recommender does not use personal gameplay history.

\* Evaluation currently uses a limited collection of predefined test cases.

\* The first semantic query may be slower while the embedding model is loaded.



\---



\## Future Improvements



\* Retrieve live Steam prices and availability

\* Add user profiles and personalised recommendation history

\* Add collaborative-filtering signals

\* Expand the evaluation dataset with manual relevance judgements

\* Add multilingual query support

\* Improve query understanding with a structured language-model parser

\* Add more detailed comparison and shortlist tools

\* Add user feedback for recommendation quality

