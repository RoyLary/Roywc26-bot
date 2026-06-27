#!/usr/bin/env python3
"""
WC2026 Telegram Prediction Bot
Runs 24/7 on Railway (free) — chat from your phone, PC can be off.
Same agent as agent.py: Elo + Dixon-Coles model, 6-step analysis, Opta-first search.
"""

import asyncio, json, logging, math, os, requests
from pathlib import Path
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
import anthropic

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN    = os.environ["TELEGRAM_BOT_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
BOT_PASSWORD      = os.environ["BOT_PASSWORD"]
MODEL      = "claude-sonnet-4-6"
MAX_TOKENS = 4096

# Authorised user IDs (in memory — persists until bot restarts)
AUTHORISED: set[int] = set()

# ── Elo ratings (inline so Railway doesn't need the repo subfolder) ────────────
RATINGS = {
    "argentina":1976,"france":2009,"spain":2010,"brazil":1955,"england":1993,
    "portugal":1945,"netherlands":1894,"germany":1926,"belgium":1878,"italy":1901,
    "colombia":1878,"uruguay":1831,"croatia":1852,"morocco":1874,"switzerland":1812,
    "usa":1826,"mexico":1834,"japan":1825,"senegal":1848,"denmark":1795,
    "ecuador":1829,"australia":1772,"south-korea":1760,"iran":1747,"poland":1731,
    "canada":1740,"serbia":1714,"wales":1688,"ghana":1659,"tunisia":1680,
    "ivory-coast":1732,"nigeria":1671,"saudi-arabia":1657,"qatar":1592,
    "egypt":1695,"algeria":1704,"scotland":1663,"cameroon":1633,"paraguay":1681,
    "venezuela":1625,"chile":1616,"peru":1612,"czech-republic":1651,
    "bosnia-and-herzegovina":1602,"south-africa":1591,"new-zealand":1591,
    "panama":1615,"jamaica":1514,"honduras":1497,"jordan":1548,"haiti":1537,
    "el-salvador":1438,"trinidad-and-tobago":1429,"guatemala":1416,"norway":1880,
    "sweden":1752,"turkey":1731,"austria":1718,"iraq":1599,"uzbekistan":1633,
    "cape-verde":1599,"dr-congo":1650,"curacao":1548,
}

# ── Dixon-Coles bivariate Poisson ──────────────────────────────────────────────
DC_RHO = -0.13

def _dc_tau(a, b, lam, mu):
    if a==0 and b==0: return 1 - lam*mu*DC_RHO
    if a==0 and b==1: return 1 + lam*DC_RHO
    if a==1 and b==0: return 1 + mu*DC_RHO
    if a==1 and b==1: return 1 - DC_RHO
    return 1.0

def _poisson(k, lam):
    if lam <= 0: return 1.0 if k==0 else 0.0
    return math.exp(-lam) * (lam**k) / math.factorial(k)

def _eg(r, opp, bonus=0.0):
    return max(0.3, min(3.5, 1.35 + ((r+bonus)-opp)/400))

def run_statistical_model(team_a: str, team_b: str, home_team: str = "") -> dict:
    ak = team_a.lower().replace(" ", "-")
    bk = team_b.lower().replace(" ", "-")
    ra = RATINGS.get(ak)
    rb = RATINGS.get(bk)
    if ra is None:
        close = [k for k in RATINGS if team_a.lower() in k]
        return {"error": f"Unknown team '{team_a}'. Similar: {close or list(RATINGS)[:8]}"}
    if rb is None:
        close = [k for k in RATINGS if team_b.lower() in k]
        return {"error": f"Unknown team '{team_b}'. Similar: {close or list(RATINGS)[:8]}"}
    hk = home_team.lower().replace(" ", "-")
    hb = 75.0 if hk==ak else (-75.0 if hk==bk else 0.0)
    lam, mu = _eg(ra, rb, hb), _eg(rb, ra, -hb/2)
    scores, wa, d, wb = [], 0.0, 0.0, 0.0
    for a in range(9):
        pa = _poisson(a, lam)
        for b in range(9):
            p = pa * _poisson(b, mu) * _dc_tau(a, b, lam, mu)
            scores.append((a, b, p))
            if a>b: wa+=p
            elif a<b: wb+=p
            else: d+=p
    t = wa+d+wb
    scores = [(a, b, p/t) for a, b, p in scores]
    scores.sort(key=lambda x: -x[2])

    top_win_a = next(((a,b,p) for a,b,p in scores if a>b), None)
    top_draw  = next(((a,b,p) for a,b,p in scores if a==b), None)
    top_win_b = next(((a,b,p) for a,b,p in scores if a<b), None)

    most_likely = max([("win_a", wa/t), ("draw", d/t), ("win_b", wb/t)], key=lambda x: x[1])
    if most_likely[0] == "win_a" and top_win_a:
        rec = f"{top_win_a[0]}-{top_win_a[1]} ({round(top_win_a[2]*100,1)}%)"
    elif most_likely[0] == "win_b" and top_win_b:
        rec = f"{top_win_b[0]}-{top_win_b[1]} ({round(top_win_b[2]*100,1)}%)"
    elif top_draw:
        rec = f"{top_draw[0]}-{top_draw[1]} ({round(top_draw[2]*100,1)}%)"
    else:
        rec = f"{scores[0][0]}-{scores[0][1]}"

    return {
        "team_a": team_a, "team_b": team_b,
        "elo_a": ra, "elo_b": rb,
        "win_a_pct": round(wa/t*100,1), "draw_pct": round(d/t*100,1), "win_b_pct": round(wb/t*100,1),
        "top_scorelines": [{"score":f"{a}-{b}","probability_pct":round(p*100,1)} for a,b,p in scores[:12]],
        "recommended_scoreline": rec,
        "recommended_note": "Top scoreline within the most likely outcome — use this as the data-driven starting point, not the raw top scoreline.",
        "model_note": "Elo + Dixon-Coles bivariate Poisson (913 calibrated internationals, Jun 2026)",
    }

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

def search_web(query: str) -> str:
    log.info(f"  [Search] {query}")
    if not TAVILY_API_KEY:
        return "Search unavailable (TAVILY_API_KEY not set)"
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 3, "search_depth": "basic"},
            timeout=20,
        )
        data = r.json()
        results = data.get("results", [])
        if not results:
            return "No results found."
        return "\n\n".join(f"{r.get('title','')}\n{r.get('content','')[:400]}" for r in results)
    except Exception as e:
        return f"Search error: {e}"

# ── System prompt (identical to agent.py) ──────────────────────────────────────
SCORING_RULES = """
| Stage                        | Exact Score | Correct Result |
|------------------------------|-------------|----------------|
| Group Stage                  | 3 pts       | 1 pt           |
| Round of 32 / Round of 16    | 5 pts       | 2 pts          |
| Quarter Finals               | 8 pts       | 4 pts          |
| Semi Finals + Third Place    | 10 pts      | 5 pts          |
| Final                        | 15 pts      | 8 pts          |
"""

SYSTEM_PROMPT = """You are an elite football analyst and World Cup 2026 prediction assistant.
Your goal: help the user beat their 11 friends in a score prediction competition.
The user is currently 3rd place, 2 points behind 1st — every point matters.
You are running as a Telegram bot — the user is on their phone.

Today's tournament: 2026 FIFA World Cup (Jun 11 – Jul 19, 2026). Group stage ends Jun 27.
Format: 12 groups of 4 teams. Top 2 per group + 8 best 3rd-place finishers advance to Round of 32.

SCORING RULES:
Group Stage: exact score 3pts | correct result 1pt
Round of 32 / Last 16: exact 5pts | correct result 2pts
Quarter Finals: exact 8pts | correct result 4pts
Semi Finals + Third Place: exact 10pts | correct result 5pts
Final: exact 15pts | correct result 8pts

PRIMARY DATA SOURCE — Opta Analyst (theanalyst.com):
Always search "site:theanalyst.com [team] World Cup 2026" first.
Opta has shots, possession, pressing, set pieces, player ratings, match previews.

MANDATORY ANALYSIS BEFORE EVERY PREDICTION:
STEP 1 — Team A form + tactics: (A) "site:theanalyst.com [Team A] World Cup 2026", (B) "[Team A] World Cup 2026 results scorers tactics formation". Extract: last 2-3 match scores, goalscorers, key players in form, defensive record, tactical shape (formation, pressing, attacking patterns), suspensions.
STEP 2 — Same two searches for Team B. Extract identical detail.
STEP 3 — "[Team A] [Team B] World Cup 2026 standings what they need". Extract current points, what each team needs, 3rd place picture, rotation risk.
STEP 4 — (A) "site:theanalyst.com [Team A] vs [Team B] 2026 preview", (B) "[Team A] vs [Team B] confirmed lineup injuries suspended World Cup 2026". Extract confirmed XI if announced (~1hr before kickoff), key absences, rotation risk.
STEP 5 — Run run_statistical_model. The output includes win/draw/loss %, top_scorelines (raw ranked list), and recommended_scoreline (top scoreline within the most likely outcome). Use recommended_scoreline as a model reference point, but the final pick must reflect ALL steps — form, tactics, confirmed lineups, and context. If everything points to 1-1, predict 1-1.
STEP 6 — Synthesise ALL steps into a verdict: predicted score, win/draw/loss % + recommended_scoreline from model, confidence (low/med/high), 1 alternative, key factor driving the pick, bold vs safe play.

KNOCKOUT STAGE SCORING RULE (CRITICAL):
Knockout predictions cover the full 120 minutes including extra time — NOT just 90 minutes.
A draw prediction (e.g. 1-1) means the game goes to penalties.
Let the data drive the pick — do not artificially avoid draws, but do not default to them either.
Group stage predictions are 90-minute scores only.

RULES:
- NEVER use or mention xG — it is rubbish for individual match predictions.
- 3rd place CAN advance — always check the advancement picture. Changes how teams play.
- Rotation alert — teams already through rest starters. Always flag this.
- Contrarian-but-defensible exact scores gain ground on rivals. Think about what 11 friends will pick.
- TELEGRAM FORMATTING — use plain text only. No markdown tables (| col |), no --- dividers, no ## headers.
  Use simple line breaks and emoji instead. Keep it readable on a phone screen.
"""

TOOLS = [
    {
        "name": "run_statistical_model",
        "description": (
            "Elo + Dixon-Coles bivariate Poisson model. Returns win/draw/loss % and top 12 scorelines. "
            "Calibrated on 913 internationals. Use at Step 5. Do NOT reference expected goals."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "team_a": {"type": "string", "description": "First team (lowercase, hyphens OK)"},
                "team_b": {"type": "string", "description": "Second team"},
                "home_team": {"type": "string", "description": "Home team if applicable (WC host USA). Leave blank for neutral."},
            },
            "required": ["team_a", "team_b"],
        },
    },
    {
        "name": "search_web",
        "description": (
            "Search the web for WC 2026 info. "
            "PRIORITY: try 'site:theanalyst.com [query]' first (Opta — best stats source). "
            "Fall back to general queries for lineups, standings, breaking news."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]

def run_tool(name: str, inp: dict) -> str:
    if name == "run_statistical_model":
        result = run_statistical_model(inp["team_a"], inp["team_b"], inp.get("home_team",""))
        log.info(f"  [Model] {inp['team_a']} vs {inp['team_b']}")
        return json.dumps(result, indent=2)
    if name == "search_web":
        return search_web(inp["query"])
    return f"Unknown tool: {name}"

# ── Agent loop (sync — called from thread executor) ────────────────────────────
def run_agent_sync(user_message: str, history: list) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    history.append({"role": "user", "content": user_message})

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=history,
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
            history.append({"role": "assistant", "content": response.content})
            history.append({"role": "user", "content": tool_results})
        else:
            reply = "".join(b.text for b in response.content if hasattr(b, "text"))
            # Keep only last Q&A for follow-up context, strip all tool call history
            history.clear()
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": reply})
            return reply

# ── Auth helpers ───────────────────────────────────────────────────────────────
def is_authorised(user_id: int) -> bool:
    return user_id in AUTHORISED

async def ask_for_password(update: Update):
    await update.message.reply_text(
        "🔒 This bot is private.\nEnter the password to continue:"
    )

# ── Telegram handlers ──────────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_authorised(user_id):
        await ask_for_password(update)
        return
    context.user_data["history"] = []
    await update.message.reply_text(
        "⚽ WC2026 Prediction Agent ready!\n\n"
        "Ask me to predict any match, e.g.:\n"
        "  predict argentina cape verde\n"
        "  predict england colombia\n\n"
        "Full 6-step analysis: form, standings, lineups, Opta stats + statistical model.\n\n"
        "/clear to reset the conversation."
    )

async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_authorised(user_id):
        await ask_for_password(update)
        return
    context.user_data["history"] = []
    await update.message.reply_text("Conversation cleared. Ready for the next match!")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id  = update.effective_user.id
    user_msg = update.message.text.strip()
    if not user_msg:
        return

    # ── Password check ─────────────────────────────────────────────────────────
    if not is_authorised(user_id):
        if BOT_PASSWORD and user_msg == BOT_PASSWORD:
            AUTHORISED.add(user_id)
            context.user_data["history"] = []
            await update.message.reply_text(
                "✅ Access granted!\n\n"
                "⚽ WC2026 Prediction Agent ready. Ask me to predict any match, e.g.:\n"
                "  predict argentina cape verde"
            )
        else:
            await update.message.reply_text("❌ Wrong password. Try again:")
        return

    if "history" not in context.user_data:
        context.user_data["history"] = []
    history = context.user_data["history"]

    thinking = await update.message.reply_text(
        "⚽ Analysing... (searching Opta + form + lineups + running model — ~30 sec)"
    )

    try:
        reply = await asyncio.to_thread(run_agent_sync, user_msg, history)
    except Exception as e:
        log.error(f"Agent error: {e}", exc_info=True)
        reply = f"Something went wrong: {e}"

    await thinking.delete()

    if len(reply) <= 4096:
        await update.message.reply_text(reply)
    else:
        for i in range(0, len(reply), 4096):
            await update.message.reply_text(reply[i:i+4096])

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log.info("Starting WC2026 Telegram bot...")
    log.info(f"BOT_PASSWORD set: {bool(BOT_PASSWORD)} | value: '{BOT_PASSWORD}'")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    log.info("Bot is running. Send /start on Telegram.")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
