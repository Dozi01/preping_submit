"""
Task Generation Prompts
"""

COMPLEXITY_LEVEL_DESCRIPTIONS = {
    1:"""- **Single App Focus**: Use **1 app** to generate deep, meaningful tasks
- **Entity Scope**: Use **1-2 entities** within the app
- **Multi-Step Tasks**: Each task should require **2+ API calls** within the app""",
    2:"""- **Two-App Workflows**: Use **2 apps** and transfer information or actions between them
- **Entity Scope**: Use **2-3 entities** across the apps
- **Multi-Step Tasks**: Each task should require **3+ API calls** across the apps""",
    3:"""- **Three-App Workflows**: Use **3 apps** and transfer information or actions across them
- **Entity Scope**: Use **3+ entities** across the apps
- **Multi-Step Tasks**: Each task should require **3+ API calls** across the apps"""
}


# =============================================================================
# TASK GENERATION PROMPT
# =============================================================================

TASK_GENERATION_PROMPT = '''You are a synthetic task generator for an AI agent to construct the memory in the environment.

Your goal is to generate diverse, realistic, and challenging task instructions that an AI assistant would need to complete. These tasks should reflect real-world user scenarios.

## API Documentation
{api_docs}

## Guidelines
1. **Feasibility**: Generate tasks that are actually executable with the given APIs
2. **Naturalness**: Write as a real user would phrase them, not as technical API calls
3. **Verifiability**: Instructions must be precise enough to allow for an objective judgment of success or failure.
   - The agent will submit its answer as a single text string. Success is judged by comparing that string to the expected value.
   - For this to be verifiable, each QUERY must require **exactly one type of answer** (e.g. one number, or one name/title, or one list.)
   - Do not ask for two or more distinct values in one task (e.g. both a count and a list, or both title and artist). Otherwise the expected format of the submitted text is ambiguous and cannot be reliably checked.
4. **Entity Use**:
   - Prefer realistic, natural task phrasing over rigid placeholder patterns.
   - If **environment information is provided**, treat it as optional grounding support:
     - You may use discovered names, titles, folders, contacts, and other entities when they help make a task concrete.
     - You do not need to force those entities into every task.
   - If **environment information is empty or missing**, you may use generic user-centric references such as:
     - "my favorite playlist", "my song library", "my playlists", "my album library", "artists I follow"

## Task Category Guidelines
Generate tasks across the following categories:

### 1. QUERY Tasks (Information Retrieval)
Tasks that require finding and returning specific information. The answer should be objective and verifiable (number, name, comma-separated list, yes/no).
Include concrete details with specific output format such as numbers, dates, names, locations, thresholds, item lists, etc.
Avoid vague terms such as 'check', 'review', 'show' or 'ensure' unless they are accompanied by measurable criteria.

### 2. ACTION Tasks (State Modification)
Tasks that require modifying the state of one or more apps. No explicit answer required - success is determined by state changes.

{dataset_examples_section}

{environment_info_section}
{task_history_section}

{memory_context_section}

## Output Format
Return a JSON array of task instructions:
```json
[
    {{
        "category": "QUERY|ACTION",
        "involved_apps": ["app1", "app2"],
        "involved_apis": ["app.api1", "app.api2"],
        "instruction": "Natural language task instruction"
    }}
]
```

Generate {num_tasks} diverse task instructions:
'''


DATASET_EXAMPLES_SECTION = """**Examples from Dataset:**
- How much money have I sent to my roommates on venmo since 1st Jan of this year?
- Name the artist most recommended to me on Spotify.
- What is the total cost of my electricity bills for this year? The bills are in \\\"~/bills/\\\" directory of my file system.
- What is the title of the most-liked song in my Spotify playlists.
- How long is my longest Spotify playlist, in minutes, rounded to the nearest number?
- Add a comment, \\\"Thank you so much!\\\", to all the venmo payments I received from my roommates in the last 10 days (including today), and like those payments.
- The last Venmo payment request I sent to Brandon was an accident and they approved it. Send them the money back.
- Like all the venmo transactions from yesterday or today involving any of my coworkers on my venmo social feed.
- Christopher has asked for my movie recommendations via phone text message. Reply to them with a list of comma-separated movie titles from my Simple Note account as per their request.
- Give a 5-star rating to all songs in my Spotify playlists which I have liked. If I have already rated it lower, increase it to 5."""


MEMORY_GUIDE_V2 = '''
   - If agent memory is provided, prioritize generating tasks that **fill gaps** in the memory.
   - **Target Missing Reasoning**: Create tasks that require **reasoning logic** currently missing or weak in the agent's memory.
   - **Address Coverage Gaps**: Focus on **under-utilized apps**, **unused workflow combinations**, or specific scenarios where the agent **previously failed**.
   - **Avoid Redundancy**: Do NOT generate tasks for capabilities or patterns that the memory already covers well.
   - If memory is empty or not provided, generate diverse tasks as usual.
'''

# =============================================================================
# SIMPLE TASK GENERATION INSTRUCTIONS
# =============================================================================
"""
- "What is the title of the oldest released song in my Spotify account from across my song, album and playlist libraries?"
- "Name the artist most recommended to me on Spotify."
- "What is the total cost of my electricity bills for this year? The bills are in \"~/bills/\" directory of my file system."
- "How much money have I sent to my roommates on venmo since 1st Jan of this year?"
- "How long is my longest Spotify playlist, in minutes, rounded to the nearest number?"

- "Follow all the classical artists on Spotify that have at least 22 followers."
- "Mark \"Learning to cook a signature dish from scratch\" in my Bucket List Simple Note as done."
- "Like all the venmo transactions from yesterday or today involving any of my coworkers on my venmo social feed."
- "Add all spotify-recommended classical songs released in this year to a new \"Spotify Recommended Songs\" playlist."
- "Add a comment, \"Thank you!\", to all the venmo payments I received from my coworkers in the last 5 days (including today), and like those payments."
"""

ENVIRONMENT_EXTRACTION_PROMPT = '''You are analyzing an AI agent's execution trajectory to extract useful environment information.

The trajectory shows an agent interacting with various APIs (Spotify, Amazon, Venmo, Gmail, SimpleNote, etc.) in a simulated environment.
Your goal is to extract **concrete, specific information** that could be used to generate new realistic tasks.

## CRITICAL: Extract ONLY Pre-existing Environment Entities
**IMPORTANT**: Only extract entities that already existed in the environment BEFORE the agent's actions.
- The environment resets between sessions, so any entities CREATED by the agent will NOT exist in future sessions.
- You must distinguish between:
  - **PRE-EXISTING entities** (returned by READ/GET/SEARCH operations) -> Extract these
  - **AGENT-CREATED entities** (created by POST/CREATE operations) -> Do NOT extract these

## Trajectory
{trajectory}

## What to Extract (ONLY from READ/GET/SEARCH outputs)

### Spotify
- Artist names, album titles, playlist names
- Song titles, genre names
- Follower counts, play counts

### Amazon
- Product names, categories
- Seller names, brand names
- Price ranges, ratings

### Venmo
- Friend names/usernames
- Transaction descriptions
- Balance information

### Gmail
- Contact names/emails
- Label names
- Recurring email patterns (bills, notifications)

### Phone
- Contact names, phone numbers
- Relationship types (roommate, coworker, friend)

### SimpleNote
- Note titles
- Existing content patterns

### File System
- Directory structures
- File naming conventions
- Existing file types

### Relationships
- roommates, coworkers, friends, siblings, parents
- manager, partner, husband, wife

## What NOT to Extract
- User's own info (name, email, phone) - this is already known
- API documentation or schemas
- Generic error messages
- **Entities CREATED by the agent's actions**

## Output Format
```json
{{
    "app_name": {{
        "entity_type": ["list", "of", "discovered", "values"],
        ...
    }},
    "relationships": {{
        "roommates": ["name1", "name2"],
        "coworkers": ["name3"],
        ...
    }}
}}
```

Extract the environment information:
'''




# Prompt for extracting environment information from trajectory
ENVIRONMENT_EXTRACTION_PROMPT = """You are analyzing an AI agent's execution trajectory to extract useful environment information.

The trajectory shows an agent interacting with various APIs (Spotify, Amazon, Venmo, etc.) in a simulated environment.
Your goal is to extract **concrete, specific information** that could be used to generate new realistic tasks.

## CRITICAL: Extract ONLY Pre-existing Environment Entities
**IMPORTANT**: Only extract entities that already existed in the environment BEFORE the agent's actions.
- The environment resets between sessions, so any entities CREATED by the agent during this session will NOT exist in future sessions.
- You must distinguish between:
  - **PRE-EXISTING entities** (returned by READ/GET/SEARCH operations) -> Extract these
  - **AGENT-CREATED entities** (created by POST/CREATE operations) -> Do NOT extract these

## Trajectory
{trajectory}

## What to Extract (ONLY from READ/GET/SEARCH outputs)
Focus on discovering **specific entities and data** that PRE-EXIST in this environment:
- **Spotify**: Artist names, album titles, playlist names, song titles, genre names (from search results, library queries, etc.)
- **Amazon**: Product names, categories, price ranges, seller names (from product listings, search results)
- **Venmo**: Friend names/emails, existing transaction history, balance information
- **Todoist**: Project names, task titles, labels (that already existed)
- **Other apps**: Any concrete entity names, IDs, or values that were DISCOVERED, not CREATED

## What NOT to Extract
- User's own info (name, email, phone) - this is already known
- API documentation or schemas
- Generic error messages without useful info
- **Entities CREATED by the agent's actions (e.g., new playlists, new transactions, new posts, new orders)**
- Any entity that was the RESULT of a create/post/add operation

## Output Format
Return a JSON object with the following structure:
```json
{{
    "app_name": {{
        "entity_type": ["list", "of", "discovered", "values"],
        ...
    }},
    ...
}}
```

Example:
```json
{{
    "spotify": {{
        "artists": ["Taylor Swift", "Ed Sheeran", "BTS"],
        "playlists": ["Rock Classics", "Top 50 Global"],
        "genres": ["pop", "rock", "hip-hop"]
    }},
    "amazon": {{
        "products": ["Sony WH-1000XM5", "Kindle Paperwhite"],
        "categories": ["Electronics", "Books"]
    }}
}}
```

If no useful environment information is found, return an empty object: {{}}

Extract the environment information:
"""


GROUNDED_ENVIRONMENT_SUMMARY_PROMPT = """You are analyzing an AI agent's execution trajectory to extract grounded environment observations for future synthetic task generation.

Your goal is to summarize only the reusable, pre-existing environment facts observed in this trajectory.

Requirements:
- Focus on concrete observations that could ground future tasks.
- Preserve task-local coherence: keep related observations together instead of flattening unrelated names into one loose list.
- Exclude anything created, posted, added, invited, updated, renamed, or otherwise changed by the agent during this task.
- Exclude generic API documentation, schemas, and error messages.
- Keep the summary compact: 2-5 bullet lines total.

Return ONLY JSON:
```json
{{
  "summary": "- observation 1\\n- observation 2"
}}
```

Task instruction:
{task_instruction}

Trajectory:
{trajectory}
"""


# Prompt for generating initial tasks (NO environment info available)
INITIAL_TASK_GENERATION_PROMPT = """You are a task instruction generator for an AI assistant evaluation system.

You have access to API documentation for available apps. Your goal is to generate diverse, realistic task instructions.

## API Documentation
{api_docs}

## Task Category Guidelines
Generate tasks across these abstract categories:

### 1. Query Tasks (Information Retrieval)
Tasks that ask for specific, verifiable information.
- Requires a clear answer: number, name, ID, or comma-separated list
- Example patterns: "How many...", "What is the name of...", "List all..."

### 2. Aggregation / Superlative Tasks
Tasks requiring comparison or filtering across multiple items.
- Find max/min, count unique, rank items, filter by condition
- May be Query (return value) or Action (perform on filtered result)

### 3. Action Tasks (State Modification)
Tasks that modify state in the environment without requiring an answer.
- Just execute the command: create, follow, delete, add, remove, etc.
- Success = state change completed, no return value needed

## Answer Format Requirements
For Query Tasks, the expected answer must be objective and verifiable - such as a number, name, ID, or comma-separated list. Avoid tasks that expect subjective summaries, explanations, or free-form narrative responses.

For Action Tasks, no answer is needed. The task is complete when the requested state change is executed successfully.

## App Scope
You may generate:
- **Single-app tasks**: Using APIs from ONE app (simpler, focused)
- **Cross-app tasks**: Combining APIs from 2-3 apps (more complex, realistic workflows)

## Guidelines
1. **Feasibility**: Generate tasks that are actually executable with the given APIs
2. **Multi-Step Tasks**: Each task should require **2+ API calls**
3. **Naturalness**: Write as a real user would phrase them, not as technical API calls
4. **Use Condition-Based References**: Since we don't know specific entity names, reference entities using conditions that uniquely identify them:
   - GOOD: "the most played song", "the longest playlist", "the oldest order"
   - GOOD: "all songs released after 2020", "products under $50", "any friends with pending requests"
   - BAD: "my favorite playlist" (ambiguous, cannot verify)
   - BAD: "Taylor Swift", "Sony WH-1000XM5" (specific names we don't know exist)
5. **Variety**: Create diverse tasks that agent can explore and given memory much.

## Output Format
Return a JSON array of task instructions:
```json
[
    {{
        "instruction": "Natural language task instruction",
        "involved_apis": ["app.api_name", ...],
        "involved_apps": ["app1", "app2"]
    }}
]
```

Generate {num_tasks} task instructions (mix of single-app and cross-app, using placeholder/generic references for entities):
"""


# Prompt for generating grounded tasks (WITH environment info)
GROUNDED_TASK_GENERATION_PROMPT = """You are a task instruction generator for an AI assistant evaluation system.

You have access to:
1. API documentation for available apps
2. **Discovered environment information** from previous executions (real entities that exist in this environment)

## API Documentation
{api_docs}

## Discovered Environment Information
This is REAL data from the environment - use these exact names/values in your tasks:
{environment_info}

## Task Category Guidelines
Generate tasks across these abstract categories:

### 1. Query Tasks (Information Retrieval)
Tasks that ask for specific, verifiable information.
- Requires a clear answer: number, name, ID, or comma-separated list
- Example patterns: "How many...", "What is the name of...", "List all..."

### 2. Aggregation / Superlative Tasks
Tasks requiring comparison or filtering across multiple items.
- Find max/min, count unique, rank items, filter by condition
- May be Query (return value) or Action (perform on filtered result)

### 3. Action Tasks (State Modification)
Tasks that modify state in the environment without requiring an answer.
- Just execute the command: create, follow, delete, add, remove, etc.
- Success = state change completed, no return value needed

## Answer Format Requirements
For Query Tasks, the expected answer must be objective and verifiable - such as a number, name, ID, or comma-separated list. Avoid tasks that expect subjective summaries, explanations, or free-form narrative responses.

For Action Tasks, no answer is needed. The task is complete when the requested state change is executed successfully.

## App Scope
You may generate:
- **Single-app tasks**: Using APIs from ONE app (simpler, focused)
- **Cross-app tasks**: Combining APIs from 2-3 apps (more complex, realistic workflows)

## Guidelines
1. **Use Real Entities**: Reference the discovered entities by their exact names (artists, products, playlists, etc.)
2. **Feasibility**: Generate tasks that are actually executable with the given APIs
3. **Multi-Step Tasks**: Each task should require **2+ API calls**
4. **Naturalness**: Write as a real user would phrase the request
5. **Variety**: Create diverse task patterns across all categories

## Output Format
Return a JSON array of task instructions:
```json
[
    {{
        "instruction": "Natural language task using real entity names",
        "involved_apis": ["app.api_name", ...],
        "involved_apps": ["app1", "app2"]
    }}
]
```

Generate {num_tasks} grounded task instructions (mix of single-app and cross-app, using the discovered environment information):
"""
