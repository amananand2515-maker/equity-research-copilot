import os
import json
import time
import requests
from groq import Groq

# Setup API Clients
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Errors from the most recent AI call, so the run can report *why* AI failed
LAST_ERRORS = []
# Models that told us they are retired/unknown during this run: skip them for the remaining stocks
DEAD_MODELS = set()

SYSTEM_PROMPT = """
You are a senior equity research analyst for Indian markets.
You have access to technical indicators, recent news, and 
memory of your past analysis on this stock.

IMPORTANT RULES:
- You NEVER change the technical score — that is computed 
  separately and is not your job
- You NEVER decide which stocks to include — that is the 
  user's decision via config.json
- Your ONLY job is to add reasoning and context to the 
  scores already computed
- Be specific to Indian markets — mention SEBI, RBI, NSE, 
  FII/DII flows where relevant
- Be direct and actionable — no vague statements

Always respond in this EXACT JSON format, nothing else:
{
  "verdict": "BUY or SELL or HOLD or WATCH",
  "confidence": "HIGH or MEDIUM or LOW",
  "agrees_with_signal": true or false,
  "one_liner": "max 12 words, specific and actionable",
  "reasoning": "2-3 sentences. Mention the 2 most important 
                factors. Reference news if relevant.",
  "key_risk": "Single biggest risk, one sentence",
  "key_catalyst": "Single biggest upside trigger, one sentence",
  "news_impact": "POSITIVE or NEGATIVE or NEUTRAL",
  "time_horizon": "SHORT (1-3 days) or MEDIUM (1-3 weeks) 
                   or LONG (1-3 months)",
  "memory_note": "One sentence referencing past history 
                  if available, else null"
}
"""

def analyze_stock(stock_data: dict, news: dict, memory_history: list, streak: dict, trend_change: dict) -> dict:
    # 1. Prepare User Data
    user_msg = f"""
    Analyze {stock_data['ticker']} ({stock_data.get('name', 'N/A')}).
    Sector: {stock_data.get('sector', 'N/A')}
    
    PRICE DATA:
    - Current: ₹{stock_data['price']}
    - 1D Change: {stock_data['change_1d']}%
    
    COMPUTED SIGNAL:
    - Composite Score: {stock_data['composite_score']}/100
    - Signal: {stock_data['signal']}
    
    TECHNICALS:
    - RSI: {stock_data['key_indicators']['rsi']} ({'Oversold' if stock_data['key_indicators']['rsi'] < 30 else 'Overbought' if stock_data['key_indicators']['rsi'] > 70 else 'Neutral'})
    - MACD crossover: {stock_data['key_indicators'].get('macd_crossover', 'N/A')}
    - SuperTrend: {stock_data['key_indicators'].get('supertrend', 'N/A')}
    - Volume Ratio: {stock_data['key_indicators'].get('volume_ratio', 'N/A')}
    - OBV trend: {stock_data['key_indicators'].get('obv_trend', 'N/A')}
    - Trend Strength (ADX): {stock_data['key_indicators'].get('adx', 'N/A')} ({stock_data['key_indicators'].get('trend_strength', 'N/A')})
    - Price vs 50 DMA: {stock_data['key_indicators'].get('price_vs_sma50', 'N/A')}%
    - Price vs 200 DMA: {stock_data['key_indicators'].get('price_vs_sma200', 'N/A')}%

    RECENT NEWS:
    {json.dumps(news.get('headlines', [])[:5], indent=2)}
    Sentiment: {json.dumps(news.get('sentiment_keywords', {}), indent=2)}

    MEMORY & PATTERN:
    - Streak: {streak.get('message', 'N/A')}
    - Trend Change: {trend_change.get('message', 'N/A')}
    - Past 5 verdicts: {[h.get('ai_verdict') for h in memory_history[-5:]]}
    """
    
    # 2. Try Groq
    LAST_ERRORS.clear()
    ai_result = _try_groq(user_msg)
    
    # 3. Try Gemini Fallback
    if not ai_result:
        ai_result = _try_gemini(user_msg)
        
    # 4. Final Fallback (Safe default)
    if not ai_result:
        ai_result = {
            "verdict": "HOLD",
            "confidence": "LOW",
            "agrees_with_signal": True,
            "one_liner": "Fallback: Manual review needed (AI unavailable)",
            "reasoning": "Both AI models (Groq, Gemini) were unavailable or hit rate limits.",
            "key_risk": "Technical failure / No connection to AI",
            "key_catalyst": "Resolution of connectivity",
            "news_impact": "NEUTRAL",
            "time_horizon": "SHORT (1-3 days)",
            "memory_note": None,
            "_model": None,
            "_ai_failed": True,
            "_errors": LAST_ERRORS[-4:] or [f"All configured models retired this run: {sorted(DEAD_MODELS)}"],
        }
        
    time.sleep(0.8) # Respect rate limits
    return ai_result

def _ai_cfg():
    try:
        from config_loader import CFG
        return CFG.get("ai", {}) or {}
    except Exception:
        return {}


def _parse_json(text):
    """Models sometimes wrap JSON in ``` fences or add a sentence; pull out the object."""
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b > a:
        try:
            return json.loads(text[a:b + 1])
        except Exception:
            return None
    return None


def _is_dead_model_error(e) -> bool:
    msg = str(e).lower()
    return any(k in msg for k in ("decommission", "not found", "does not exist", "no longer supported", "invalid model"))


REQUIRED_KEYS = {"verdict", "confidence", "one_liner", "reasoning"}


def _valid(result):
    return isinstance(result, dict) and REQUIRED_KEYS.issubset(result)


def _try_groq(prompt):
    if not GROQ_API_KEY:
        LAST_ERRORS.append("Groq: GROQ_API_KEY not set")
        return None
    cfg = _ai_cfg()
    models = cfg.get("groq_models") or ["openai/gpt-oss-120b"]
    client = Groq(api_key=GROQ_API_KEY)
    for model in models:
        if model in DEAD_MODELS:
            continue
        for use_json_mode in (True, False):
            try:
                kwargs = dict(
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": prompt}],
                    model=model,
                    temperature=cfg.get("temperature", 0.2),
                    max_tokens=cfg.get("max_tokens", 800),
                )
                if use_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = client.chat.completions.create(**kwargs)
                result = _parse_json(resp.choices[0].message.content)
                if _valid(result):
                    result["_model"] = f"groq/{model}"
                    return result
                LAST_ERRORS.append(f"Groq {model}: reply was not valid JSON")
            except Exception as e:
                LAST_ERRORS.append(f"Groq {model}: {str(e)[:160]}")
                print(f"✗ Groq {model} failed: {str(e)[:160]}")
                # A retired/unknown model won't work without JSON mode either
                if _is_dead_model_error(e):
                    DEAD_MODELS.add(model)
                    break
    return None


def _try_gemini(prompt):
    if not GEMINI_API_KEY:
        LAST_ERRORS.append("Gemini: GEMINI_API_KEY not set")
        return None
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        LAST_ERRORS.append(f"Gemini SDK import failed: {e}")
        return None
    cfg = _ai_cfg()
    client = genai.Client(api_key=GEMINI_API_KEY)
    for model in cfg.get("gemini_models") or ["gemini-3.6-flash"]:
        if model in DEAD_MODELS:
            continue
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    temperature=cfg.get("temperature", 0.2),
                ),
            )
            result = _parse_json(resp.text)
            if _valid(result):
                result["_model"] = f"gemini/{model}"
                return result
            LAST_ERRORS.append(f"Gemini {model}: reply was not valid JSON")
        except Exception as e:
            LAST_ERRORS.append(f"Gemini {model}: {str(e)[:160]}")
            print(f"✗ Gemini {model} failed: {str(e)[:160]}")
            if _is_dead_model_error(e):
                DEAD_MODELS.add(model)
    return None
