#!/usr/bin/env python3
"""World Cup 2026 Prediction Agent — built to get you to 1st place."""

import json
import math
import os
from pathlib import Path

import anthropic
from duckduckgo_search import DDGS

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096

SCORING_RULES = """
| Stage                        | Exact Score | Correct Result |
|------------------------------|-------------|----------------|
| Group Stage                  | 3 pts       | 1 pt           |
| Round of 32 / Round of 16    | 5 pts       | 2 pts          |
| Quarter Finals               | 8 pts       | 4 pts          |
| Semi Finals + Third Place    | 10 pts      | 5 pts          |
| Final                        | 15 pts      | 8 pts          |
"""

SYSTEM_PROMPT = f"""You are an elite football analyst and World Cup 2026 prediction assistant. \
Your one goal: help the user beat their 11 friends in a score prediction competition. \
The user is currently 3rd place, 2 points behind 1st — every point matters.

Today's date: June 27, 2026. The 2026 FIFA World Cup is currently in progress (Jun 11 – Jul 19).
Format: 12 groups of 4 teams. Top 2 from each group + 8 best 3rd-place finishers → Round of 32.

━━━ COMPETITION SCORING RULES ━━━
{SCORING_RULES}

━━━ PRIMARY DATA SOURCE — OPTA ANALYST ━━━
Your primary source for match stats, previews, and team data is Opta Analyst (theanalyst.com). \
Opta is the gold standard in football analytics — used by professionals worldwide. \
When searching for stats, always try "site:theanalyst.com [query]" first before generic web search. \
Key data Opta provides: shots, possession history, goals scored/conceded patterns, pressing \
intensity, set piece threat, player ratings, head-to-head records, and pre-match previews.

━━━ MANDATORY PRE-MATCH ANALYSIS — DO THIS BEFORE EVERY PREDICTION ━━━
You MUST run all of the following steps in order before giving any score prediction:

STEP 1 — Team A's World Cup form, tactics + player stats
  Search A: "site:theanalyst.com [Team A] World Cup 2026"  ← Opta match stats/preview
  Search B: "[Team A] World Cup 2026 matches goals scorers results"  ← match-by-match results
  Search C: "[Team A] World Cup 2026 tactics formation pressing how they play"  ← tactical shape
  Extract ALL of the following:
  - Last 2-3 WC 2026 matches: exact scores, who scored, which minute, how they conceded
  - Key players in form: top scorers, assist makers, player driving attacks
  - Defensive record: clean sheets, goals conceded, how vulnerable at the back
  - Tactical patterns: formation used, pressing intensity, defensive shape, how they attack \
(through wide areas, through the middle, set pieces), how they transition defence-to-attack
  - Any red cards, suspensions, or key injuries from WC matches

STEP 2 — Team B's World Cup form, tactics + player stats
  Same three searches for Team B. Extract the same detail: match scores, goalscorers, \
  key players in form, defensive record, suspensions, and tactical patterns.

STEP 3 — Tournament context
  Search: "[Group X] World Cup 2026 standings table"
  Extract: current points, GD, what each team needs (must win / draw enough / already through / \
already eliminated), whether 3rd place advancement is relevant and what points threshold is needed.

STEP 4 — Lineups, injuries & Opta match preview
  Search A: "site:theanalyst.com [Team A] vs [Team B] preview 2026"  ← Opta preview
  Search B: "[Team A] vs [Team B] lineup confirmed injuries suspended World Cup 2026"
  Extract: confirmed or expected starting XI, key absences, rotation risks \
(teams already qualified often rest players), set piece threats flagged by Opta.

STEP 5 — Statistical model
  Run run_statistical_model. The output includes:
  - win/draw/loss % — the overall probability of each outcome
  - top_scorelines — all scorelines ranked by raw probability (NOTE: can be misleading, \
    see below)
  - recommended_scoreline — the top scoreline WITHIN the most likely outcome. \
    THIS is your data-driven starting point. Always use this, not the raw top scoreline.
  WHY: The Dixon-Coles model inflates 1-1 and 0-0, making them appear top of the raw list \
  even when draw% is only 20%. The recommended_scoreline corrects for this by finding the \
  best scoreline within the outcome the model actually favours.

STEP 6 — Synthesise and recommend
  Combine all of the above (Opta stats, form, context, model) into a clear verdict:
  - Start from recommended_scoreline as your baseline. Adjust up/down based on form, \
    tactics, lineups, and context — but you need a strong reason to deviate from it.
  - State the predicted score
  - State win/draw/loss % and the recommended_scoreline from the model
  - State confidence (low / medium / high)
  - State 1 alternative scoreline
  - Explain the key factor driving the pick (form, Opta stats, context, or lineup)
  - Flag if a correct-result-only play is smarter than chasing the exact score

━━━ KNOCKOUT STAGE SCORING RULE (CRITICAL) ━━━
Knockout predictions cover the full 120 minutes including extra time — NOT just 90 minutes. \
A draw prediction (e.g. 1-1) means the game goes to penalties. \
Let the data drive the pick — do not artificially avoid draws, but do not default to them either. \
Group stage predictions are 90-minute scores only.

━━━ STRATEGIC PRINCIPLES ━━━
1. EXACT SCORES ARE KING in knockout rounds. Final: exact (15pts) vs correct result (8pts). \
Always push for the exact scoreline in late rounds.
2. Group stage: exact (3pts) is 3× a correct result — worth targeting when confident.
3. NEVER use or reference xG — it is not a reliable predictor for individual match scorelines.
4. 3rd place CAN advance — always check if a team sitting 3rd can still qualify. \
A team with 4pts in 3rd usually advances; 3pts is borderline. \
This changes how desperate teams are to attack vs sit on a result.
5. Rotation alert — teams already through may rest key players. Always check this.
6. Think about what your 11 friends will pick. Contrarian-but-defensible exact scores \
gain ground; everyone picking the same result means zero points gained on rivals.
7. When uncertain on exact score, lean 1-0 or 2-1 — most common WC knockout scorelines historically.

━━━ WHAT YOU CAN HELP WITH ━━━
- Predict scores for upcoming matches (always follow the 6 steps above)
- Discuss tournament bracket and likely upsets
- Review the user's existing picks and suggest adjustments
- Help think through bold vs safe picks given the current points gap
"""

# ── Statistical Model (Elo + Dixon-Coles Bivariate Poisson) ───────────────────
# Ported from https://github.com/Hicruben/world-cup-2026-prediction-model

_RATINGS_FILE = Path(__file__).parent / "world-cup-2026-prediction-model" / "data" / "elo-calibrated.json"
with open(_RATINGS_FILE) as _f:
    _ELO_DATA = json.load(_f)
RATINGS: dict[str, int] = _ELO_DATA["ratings"]

DC_RHO = -0.13  # Dixon-Coles correction (empirically ~-0.13)


def _dc_tau(a: int, b: int, lam: float, mu: float) -> float:
    if a == 0 and b == 0: return 1 - lam * mu * DC_RHO
    if a == 0 and b == 1: return 1 + lam * DC_RHO
    if a == 1 and b == 0: return 1 + mu * DC_RHO
    if a == 1 and b == 1: return 1 - DC_RHO
    return 1.0


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _expected_goals(rating: int, opponent: int, home_bonus: float = 0) -> float:
    diff = (rating + home_bonus) - opponent
    return max(0.3, min(3.5, 1.35 + diff / 400))


def run_statistical_model(team_a: str, team_b: str, home_team: str = "") -> dict:
    """
    Compute win/draw/loss probabilities and top scorelines via
    Elo + Dixon-Coles bivariate Poisson (ported from the GitHub model).
    """
    a_key = team_a.lower().replace(" ", "-")
    b_key = team_b.lower().replace(" ", "-")

    if a_key not in RATINGS:
        close = [k for k in RATINGS if team_a.lower() in k]
        return {"error": f"Unknown team '{team_a}'. Close matches: {close or list(RATINGS)[:10]}"}
    if b_key not in RATINGS:
        close = [k for k in RATINGS if team_b.lower() in k]
        return {"error": f"Unknown team '{team_b}'. Close matches: {close or list(RATINGS)[:10]}"}

    ra, rb = RATINGS[a_key], RATINGS[b_key]

    home_bonus = 0.0
    home_label = "neutral venue"
    if home_team:
        h_key = home_team.lower().replace(" ", "-")
        if h_key == a_key:
            home_bonus = 75.0
            home_label = f"{team_a} at home (+75 Elo)"
        elif h_key == b_key:
            home_bonus = -75.0
            home_label = f"{team_b} at home (+75 Elo)"

    lam = _expected_goals(ra, rb, home_bonus)
    mu  = _expected_goals(rb, ra, -home_bonus / 2)

    # Build full scoreline distribution (0-0 … 8-8)
    scorelines: list[tuple[int, int, float]] = []
    win_a = draw = win_b = 0.0

    for a in range(9):
        pa = _poisson_pmf(a, lam)
        for b in range(9):
            tau = _dc_tau(a, b, lam, mu)
            p = pa * _poisson_pmf(b, mu) * tau
            scorelines.append((a, b, p))
            if a > b:   win_a += p
            elif a < b: win_b += p
            else:       draw  += p

    total = win_a + draw + win_b
    win_a /= total; draw /= total; win_b /= total
    scorelines = [(a, b, p / total) for a, b, p in scorelines]
    scorelines.sort(key=lambda x: -x[2])

    top = [
        {"score": f"{a}-{b}", "probability_pct": round(p * 100, 1)}
        for a, b, p in scorelines[:12]
    ]

    # Top scoreline within each outcome separately
    top_win_a = next(((a, b, p) for a, b, p in scorelines if a > b), None)
    top_draw  = next(((a, b, p) for a, b, p in scorelines if a == b), None)
    top_win_b = next(((a, b, p) for a, b, p in scorelines if a < b), None)

    # Recommended = top scoreline within the most likely outcome
    most_likely = max([("win_a", win_a), ("draw", draw), ("win_b", win_b)], key=lambda x: x[1])
    if most_likely[0] == "win_a" and top_win_a:
        rec = f"{top_win_a[0]}-{top_win_a[1]} ({round(top_win_a[2]*100,1)}%)"
    elif most_likely[0] == "win_b" and top_win_b:
        rec = f"{top_win_b[0]}-{top_win_b[1]} ({round(top_win_b[2]*100,1)}%)"
    elif top_draw:
        rec = f"{top_draw[0]}-{top_draw[1]} ({round(top_draw[2]*100,1)}%)"
    else:
        rec = top[0]["score"]

    return {
        "team_a": team_a,
        "team_b": team_b,
        "elo_a": ra,
        "elo_b": rb,
        "venue": home_label,
        "win_a_pct": round(win_a * 100, 1),
        "draw_pct":  round(draw  * 100, 1),
        "win_b_pct": round(win_b * 100, 1),
        "top_scorelines": top,
        "recommended_scoreline": rec,
        "recommended_note": "Top scoreline within the most likely outcome (win/draw/loss) — use this as the model's data-driven starting point, not the raw top scoreline which can be inflated by the Dixon-Coles correction.",
        "model_note": "Elo + Dixon-Coles bivariate Poisson (calibrated on 913 internationals, Jun 2026)",
    }


# ── Web search ────────────────────────────────────────────────────────────────
def search_web(query: str) -> str:
    print(f"  [Searching: {query}]")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if not results:
            return "No results found."
        parts = []
        for r in results:
            body = (r.get("body") or "")[:400]
            parts.append(f"### {r.get('title','')}\n{body}\n({r.get('href','')})")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Search error: {e}"


# ── Tool definitions ───────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "run_statistical_model",
        "description": (
            "Run the Elo + Dixon-Coles bivariate Poisson statistical model on a match. "
            "Returns win/draw/loss probabilities and the top 12 most likely scorelines with "
            "exact probabilities. Calibrated on 913 real internationals (Oct 2023 – Jun 2026). "
            "Use this as step 5 of the pre-match analysis, after researching WC form, context, "
            "and lineups. Do NOT reference expected goals from this output — ignore that field."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "team_a": {
                    "type": "string",
                    "description": "First team (e.g. 'france', 'brazil', 'south korea'). "
                                   "Spaces and hyphens both work."
                },
                "team_b": {
                    "type": "string",
                    "description": "Second team."
                },
                "home_team": {
                    "type": "string",
                    "description": "Optional. Which team has home advantage (relevant for USA "
                                   "as host nation). Leave blank for neutral venue."
                }
            },
            "required": ["team_a", "team_b"]
        }
    },
    {
        "name": "search_web",
        "description": (
            "Search the web for up-to-date World Cup 2026 information. "
            "PRIORITY SOURCE: Opta Analyst (theanalyst.com) — always try 'site:theanalyst.com [query]' "
            "first for team stats, match previews, and tournament data. Opta is the professional "
            "standard: shots, possession, pressing, set pieces, head-to-head records, player ratings. "
            "Fall back to generic queries for lineup leaks, injuries, live standings, and breaking news."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query."
                }
            },
            "required": ["query"]
        }
    }
]


def run_tool(name: str, input_data: dict) -> str:
    if name == "run_statistical_model":
        result = run_statistical_model(
            input_data["team_a"],
            input_data["team_b"],
            input_data.get("home_team", ""),
        )
        print(f"  [Model: {input_data['team_a']} vs {input_data['team_b']}]")
        return json.dumps(result, indent=2)
    if name == "search_web":
        return search_web(input_data["query"])
    return f"Unknown tool: {name}"


# ── Agent loop ────────────────────────────────────────────────────────────────
def run_agent() -> None:
    client = anthropic.Anthropic()
    messages: list[dict] = []

    available_teams = sorted(RATINGS.keys())
    print()
    print("⚽  World Cup 2026 Prediction Agent")
    print("=" * 45)
    print(f"Model loaded: {len(available_teams)} teams with calibrated Elo ratings.")
    print("Currently: 3rd place, 2pts off the lead. Let's fix that.")
    print("Type 'quit' to exit, 'clear' to reset the conversation.")
    print("Type 'teams' to list all available team names.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGood luck — go get that 1st place! ⚽")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Good luck — go get that 1st place! ⚽")
            break
        if user_input.lower() == "clear":
            messages = []
            print("[Conversation cleared]\n")
            continue
        if user_input.lower() == "teams":
            print("Available teams: " + ", ".join(available_teams) + "\n")
            continue

        messages.append({"role": "user", "content": user_input})

        # Agentic loop — keeps going until the model stops calling tools
        while True:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = run_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
            else:
                reply = "".join(
                    block.text for block in response.content if hasattr(block, "text")
                )
                print(f"\nAgent: {reply}\n")
                # Keep only the last Q&A — enables follow-ups without carrying all search history
                messages = [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": reply},
                ]
                break


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        print("Set it with:  $env:ANTHROPIC_API_KEY='sk-ant-...'  (PowerShell)")
        raise SystemExit(1)
    run_agent()
