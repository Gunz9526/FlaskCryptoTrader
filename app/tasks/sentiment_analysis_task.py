import re
import feedparser
import logging
from google import genai
from google.genai import types
from .celery_app import celery_instance
from app.config import settings
from app.redis_client import redis_client
import json

@celery_instance.task(name="tasks.update_news_sentiment")
def update_news_sentiment():
    if not settings.GEMINI_API_KEY:
        logging.warning("GEMINI_API_KEY not set, using neutral sentiment")
        redis_client.set("news_sentiment_score", 0.0, ex=3600)
        return
    
    try:        
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
    except Exception as e:
        logging.error(f"Failed to configure Gemini API: {e}")
        redis_client.set("news_sentiment_score", 0.0, ex=3600)
        return

    generation_config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=2048,
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]
    )

    logging.info("Updating news sentiment using a single batch API call")
    
    rss_feeds = [
        'https://cointelegraph.com/rss/tag/bitcoin',
        'https://www.coindesk.com/arc/outboundfeeds/rss/',
        'https://decrypt.co/feed',
        'https://bitcoinmagazine.com/.rss/full/'
    ]
    
    articles_to_process = []
    processed_titles = set()

    for feed_url in rss_feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:5]:
                logging.info(f"Fetched article: {entry.title}")
                if entry.title not in processed_titles:
                    processed_titles.add(entry.title)
                    articles_to_process.append({
                        "title": entry.title,
                        "summary": getattr(entry, 'summary', '')[:300]
                    })
        except Exception as e:
            logging.warning(f"Failed to parse feed {feed_url}: {e}")

    if not articles_to_process:
        redis_client.set("news_sentiment_score", 0.0, ex=3600)
        logging.warning("No articles found to analyze, setting neutral sentiment")
        return

    articles_json_string = json.dumps(articles_to_process, indent=2)
    prompt = f"""
    You are a financial analyst specializing in cryptocurrency markets. Your task is to analyze the sentiment of news articles for trading signals.

    **Instructions:**
    1.  Analyze each article in the provided JSON list.
    2.  Consider its potential market impact, regulatory news, and overall investor sentiment.
    3.  Provide a sentiment score from -1.0 (very bearish) to 1.0 (very bullish) for each article.
    4.  Your response MUST be a valid JSON array of objects, containing only the 'title' and 'sentiment_score'. Do not include any other text, explanations, or markdown.

    **Example:**

    **Input Articles:**
    ```json
    [
      {{
        "title": "Global Regulators Announce Strict New Crypto Rules",
        "summary": "A joint task force of international financial regulators has announced a new framework..."
      }},
      {{
        "title": "Tech Giant Launches New Bitcoin ETF, Sparking Market Rally",
        "summary": "Shares of major tech company surged today after the successful launch of their spot Bitcoin ETF..."
      }}
    ]
    ```

    **Your Output:**
    ```json
    [
      {{
        "title": "Global Regulators Announce Strict New Crypto Rules",
        "sentiment_score": -0.8
      }},
      {{
        "title": "Tech Giant Launches New Bitcoin ETF, Sparking Market Rally",
        "sentiment_score": 0.9
      }}
    ]
    ```

    ---

    **Actual Articles to Analyze:**

    **Input Articles:**
    ```json
    {articles_json_string}
    ```

    **Your Output:**
    """

    total_score = 0
    article_count = 0
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=generation_config,
        )
        if not response.candidates:
            block_reason = response.promptFeedback.blockReason
            logging.warning(f"Content blocked for batch request. Reason: {block_reason}")
            redis_client.set("news_sentiment_score", 0.0, ex=3600)
            return

        text_response = response.text.strip()
        logging.debug(f"Raw response from Gemini: {text_response}")
        
        match = re.search(r'```json\s*([\s\S]*?)\s*```', text_response)
        if match:
            text_response = match.group(1)

        results = json.loads(text_response)
        
        for result in results:
            sentiment_score = float(result.get('sentiment_score', 0.0))
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
            total_score += sentiment_score
            article_count += 1
            logging.info(f"Analyzed '{result.get('title', 'N/A')[:50]}...': Score = {sentiment_score:.2f}")

    except Exception as e:
        logging.error(f"Failed to analyze sentiment for batch request: {e}", exc_info=True)
        redis_client.set("news_sentiment_score", 0.0, ex=3600)
        return

    if article_count > 0:
        average_score = total_score / article_count
        redis_client.set("news_sentiment_score", average_score, ex=3600)
        logging.info(f"Updated sentiment score to {average_score:.4f} from {article_count} articles")
    else:
        redis_client.set("news_sentiment_score", 0.0, ex=3600)
        logging.warning("No articles were successfully analyzed in the batch, setting neutral sentiment")
