"""
Generates a daily wisdom quote from the books and posts to Instagram + Twitter.
Run manually or via cron: python daily_post.py
"""

import os
import io
import textwrap
import requests
import tweepy
from pathlib import Path
from datetime import date
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import anthropic
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env", override=True)

anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN")
INSTAGRAM_ACCOUNT_ID = os.getenv("INSTAGRAM_ACCOUNT_ID")
IMAGE_HOST_URL = os.getenv("IMAGE_HOST_URL")

TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ── 1. Generate popular quote via Claude ──────────────────────────────────────

def generate_quote() -> dict:
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": (
                "From the following sacred Hindu texts — Bhagavad Gita, Ashtavakra Gita, "
                "Chandogya Upanishad, Mandukya Upanishad, Katha Upanishad, Mundaka Upanishad, "
                "or the Principal Upanishads — pick one of the most POPULAR, widely-known, "
                "and frequently cited verses or quotes. Choose quotes that are famous, "
                "deeply meaningful, and commonly referenced by spiritual teachers and scholars. "
                "Vary the book each day.\n\n"
                "Format your response exactly as:\n"
                "QUOTE: <the quote in English>\n"
                "SOURCE: <book name and chapter/verse reference>\n"
                "REFLECTION: <2 sentences on why this quote resonates universally>"
            ),
        }],
    )
    text = response.content[0].text
    lines = {}
    for line in text.splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            lines[k.strip()] = v.strip()
    return {
        "quote": lines.get("QUOTE", ""),
        "source": lines.get("SOURCE", "Sacred Texts"),
        "reflection": lines.get("REFLECTION", ""),
    }


# ── 2. Generate audio via OpenAI TTS ─────────────────────────────────────────

def generate_audio(quote_data: dict) -> Path:
    speech_text = f"{quote_data['quote']}. {quote_data['reflection']}"
    response = openai_client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=speech_text,
    )
    audio_path = OUTPUT_DIR / f"quote_{date.today()}.mp3"
    response.stream_to_file(str(audio_path))
    print(f"Audio saved: {audio_path}")
    return audio_path


# ── 3. Generate image ─────────────────────────────────────────────────────────

def generate_background_image() -> bytes:
    prompt = (
        "A serene spiritual background for an Instagram post inspired by Vedanta philosophy. "
        "Soft golden light, sacred geometry, lotus flowers, ancient Sanskrit motifs. "
        "Warm amber and saffron tones. No text. Cinematic quality."
    )
    response = openai_client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return requests.get(response.data[0].url).content


def compose_image(quote_data: dict, bg_bytes: bytes) -> Path:
    bg = Image.open(io.BytesIO(bg_bytes)).convert("RGBA")
    bg = bg.resize((1080, 1080))

    overlay = Image.new("RGBA", bg.size, (0, 0, 0, 150))
    bg = Image.alpha_composite(bg, overlay).convert("RGB")
    draw = ImageDraw.Draw(bg)
    width, height = bg.size

    try:
        font_quote = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 44)
        font_source = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
    except Exception:
        font_quote = ImageFont.load_default()
        font_source = font_quote

    wrapped = textwrap.fill(f'"{quote_data["quote"]}"', width=30)
    draw.multiline_text(
        (width // 2, height // 2 - 60),
        wrapped,
        font=font_quote,
        fill=(255, 250, 220),
        align="center",
        anchor="mm",
        spacing=14,
    )
    draw.text(
        (width // 2, height - 130),
        f"— {quote_data['source']}",
        font=font_source,
        fill=(210, 200, 170),
        anchor="mm",
    )

    image_path = OUTPUT_DIR / f"quote_{date.today()}.jpg"
    bg.save(str(image_path), "JPEG", quality=92)
    print(f"Image saved: {image_path}")
    return image_path


# ── 4. Post to Instagram ──────────────────────────────────────────────────────

def post_to_instagram(image_path: Path, quote_data: dict) -> str:
    if not INSTAGRAM_ACCESS_TOKEN or not INSTAGRAM_ACCOUNT_ID:
        print("Instagram credentials not set, skipping.")
        return ""

    image_url = f"{IMAGE_HOST_URL}/{image_path.name}"
    caption = (
        f'"{quote_data["quote"]}"\n\n'
        f"— {quote_data['source']}\n\n"
        f"{quote_data['reflection']}\n\n"
        f"#Vedanta #BhagavadGita #Upanishads #HinduPhilosophy #DailyWisdom "
        f"#Spirituality #AshtavakraGita #Advaita #SacredTexts #Consciousness"
    )

    container_res = requests.post(
        f"https://graph.facebook.com/v19.0/{INSTAGRAM_ACCOUNT_ID}/media",
        data={"image_url": image_url, "caption": caption, "access_token": INSTAGRAM_ACCESS_TOKEN},
    )
    container_res.raise_for_status()

    publish_res = requests.post(
        f"https://graph.facebook.com/v19.0/{INSTAGRAM_ACCOUNT_ID}/media_publish",
        data={"creation_id": container_res.json()["id"], "access_token": INSTAGRAM_ACCESS_TOKEN},
    )
    publish_res.raise_for_status()
    post_id = publish_res.json()["id"]
    print(f"Posted to Instagram: {post_id}")
    return post_id


# ── 5. Post to Twitter/X ──────────────────────────────────────────────────────

def post_to_twitter(image_path: Path, quote_data: dict) -> str:
    if not all([TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET]):
        print("Twitter credentials not set, skipping.")
        return ""

    # v1.1 client for media upload
    auth_v1 = tweepy.OAuth1UserHandler(
        TWITTER_API_KEY, TWITTER_API_SECRET,
        TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET,
    )
    api_v1 = tweepy.API(auth_v1)

    # v2 client for posting tweet
    client_v2 = tweepy.Client(
        consumer_key=TWITTER_API_KEY,
        consumer_secret=TWITTER_API_SECRET,
        access_token=TWITTER_ACCESS_TOKEN,
        access_token_secret=TWITTER_ACCESS_SECRET,
    )

    media = api_v1.media_upload(str(image_path))

    tweet_text = (
        f'"{quote_data["quote"]}"\n\n'
        f'— {quote_data["source"]}\n\n'
        f'#Vedanta #BhagavadGita #Upanishads #DailyWisdom'
    )
    # Twitter has 280 char limit — truncate if needed
    if len(tweet_text) > 280:
        max_quote = 180
        short_quote = quote_data["quote"][:max_quote] + "…"
        tweet_text = f'"{short_quote}"\n\n— {quote_data["source"]}\n\n#Vedanta #DailyWisdom'

    response = client_v2.create_tweet(text=tweet_text, media_ids=[media.media_id])
    print(f"Posted to Twitter: {response.data['id']}")
    return response.data["id"]


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print(f"Generating daily wisdom post for {date.today()}...")

    quote_data = generate_quote()
    print(f"\nQuote: {quote_data['quote'][:80]}...")
    print(f"Source: {quote_data['source']}")

    audio_path = generate_audio(quote_data)
    bg_bytes = generate_background_image()
    image_path = compose_image(quote_data, bg_bytes)

    post_to_instagram(image_path, quote_data)
    post_to_twitter(image_path, quote_data)

    print("\nDone.")


if __name__ == "__main__":
    run()
