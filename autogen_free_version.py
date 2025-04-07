import requests
from autogen import UserProxyAgent, ConversableAgent
from langchain_community.tools import DuckDuckGoSearchRun
import yfinance as yf

# HuggingFace LLM config
HUGGINGFACE_API_KEY = "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"
HUGGINGFACE_MODEL = "HuggingFaceH4/zephyr-7b-beta"

def call_huggingface(prompt):
    url = f"https://api-inference.huggingface.co/models/{HUGGINGFACE_MODEL}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": prompt}
    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    try:
        return result[0]['generated_text']
    except Exception:
        return str(result)

# Web Search Agent
search_tool = DuckDuckGoSearchRun()

class WebSearchAgent(ConversableAgent):
    def generate_reply(self, messages, sender, config=None):
        query = messages[-1]["content"]
        if not query:
            return "No query provided for web search."
        result = search_tool.run(query)
        return f"🔎 Search Results for **{query}**:\n{result}"

# Finance Agent using yFinance + HuggingFace
class FinanceAgent(ConversableAgent):
    def generate_reply(self, messages, sender, config=None):
        msg = messages[-1]["content"]
        tickers = ["TSLA", "AAPL", "NVDA"]
        reply = ""
        for ticker in tickers:
            if ticker.lower() in msg.lower():
                try:
                    info = yf.Ticker(ticker).info
                    price = info.get('currentPrice', 'N/A')
                    sector = info.get('sector', 'N/A')
                    reco = info.get('recommendationKey', 'N/A')
                    
                    hf_prompt = (
                        f"The stock {ticker} is in the {sector} sector, "
                        f"currently priced at {price} USD with an analyst recommendation of '{reco}'. "
                        f"Would you consider this stock a good long-term investment? Explain briefly."
                    )
                    hf_analysis = call_huggingface(hf_prompt)


                    reply += (
                        f"\n📈 {ticker}:\n"
                        f" - Price: {price}\n"
                        f" - Sector: {sector}\n"
                        f" - Recommendation: {reco}\n"
                        f"🧠 LLM Opinion: {hf_analysis.strip()}\n"
                    )
                except Exception as e:
                    reply += f"\n⚠️ Error fetching data for {ticker}: {e}\n"
        return reply or "No known stock ticker mentioned."

# Agents
web_agent = WebSearchAgent(
    name="web_agent",
    system_message="Fetches info using DuckDuckGo.",
)

finance_agent = FinanceAgent(
    name="finance_agent",
    system_message="Provides financial insights using yfinance and HuggingFace LLM.",
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=3,
    code_execution_config={
        "use_docker": False
    }
)

# Conversation
if __name__ == "__main__":
    user_proxy.initiate_chat(
        recipient=finance_agent,
        message="Analyze TSLA, AAPL, and NVDA and tell me which is a good long-term buy.."
    )

    user_proxy.initiate_chat(
        recipient=web_agent,
        message="Give me latest news about TSLA and AAPL for investment decisions."
    )
