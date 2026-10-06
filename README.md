\# YatraLink 🚆



> \*\*Find a way, not just a train.\*\*



YatraLink is an intelligent railway journey optimizer for India.



Instead of simply telling users that a direct train has no confirmed availability, YatraLink searches for practical alternatives such as:



\- Direct trains

\- Same-train split journeys

\- Connecting trains

\- Multi-segment journeys

\- Alternative routes



\## Core Idea



If:



`BLR → Delhi` ❌ No confirmed seat



YatraLink searches alternatives such as:



`BLR → Hyderabad → Delhi`



or:



`BLR → Nagpur → Delhi`



and ranks journeys based on:



\- Seat availability

\- Price

\- Journey duration

\- Number of train changes

\- Connection time

\- Connection risk



\## Project Status



🚧 Product-definition and architecture phase.



\## Documentation



\- \[Product Requirements Document](docs/YatraLink\_PRD.pdf)

\- \[Technical Design Document](docs/YatraLink\_Technical\_Design\_Document.pdf)

\- \[Technology Stack](docs/YatraLink\_Tech\_Stack.pdf)



\## Planned Technology



\- Next.js

\- TypeScript

\- Tailwind CSS

\- FastAPI

\- Python

\- PostgreSQL / Supabase

\- Redis

\- AI/LLM layer



\## Core Architecture



```text

User

&#x20;│

&#x20;▼

YatraLink Web App

&#x20;│

&#x20;▼

Search API

&#x20;│

&#x20;├── Train Data

&#x20;├── Availability

&#x20;├── Connection Engine

&#x20;└── Ranking Engine

&#x20;│

&#x20;▼

Best Journey Options


