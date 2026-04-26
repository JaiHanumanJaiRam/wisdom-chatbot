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
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import anthropic
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env", override=True)

anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN")
INSTAGRAM_ACCOUNT_ID = os.getenv("INSTAGRAM_ACCOUNT_ID")
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
USED_QUOTES_FILE = Path(__file__).parent / "used_quotes.json"


def load_used_quotes() -> list[str]:
    import json
    if USED_QUOTES_FILE.exists():
        return json.loads(USED_QUOTES_FILE.read_text())
    return []


def save_used_quote(quote: str):
    import json
    used = load_used_quotes()
    used.append(quote)
    used = used[-60:]  # keep last 60 to avoid very long prompts
    USED_QUOTES_FILE.write_text(json.dumps(used, indent=2))


# ── 1. Generate popular quote via Claude ──────────────────────────────────────

def generate_quote() -> dict:
    used = load_used_quotes()
    avoid_section = ""
    if used:
        avoid_list = "\n".join(f"- {q}" for q in used[-30:])
        avoid_section = f"\n\nDo NOT repeat any of these already-used quotes:\n{avoid_list}\n"

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
                "Vary the book each day."
                + avoid_section +
                "\nFormat your response exactly as:\n"
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
        voice="shimmer",
        input=speech_text,
    )
    audio_path = OUTPUT_DIR / f"quote_{datetime.now().strftime('%Y-%m-%d_%H')}.mp3"
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

    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",          # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    font_quote = font_source = ImageFont.load_default()
    for fp in font_paths:
        if Path(fp).exists():
            font_quote = ImageFont.truetype(fp, 44)
            font_source = ImageFont.truetype(fp, 30)
            break

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

    image_path = OUTPUT_DIR / f"quote_{datetime.now().strftime('%Y-%m-%d_%H')}.jpg"
    bg.save(str(image_path), "JPEG", quality=92)
    print(f"Image saved: {image_path}")
    return image_path


# ── 4. Combine image + audio into video ──────────────────────────────────────

def create_reel_video(image_path: Path, audio_path: Path) -> Path:
    import subprocess
    video_path = OUTPUT_DIR / f"reel_{datetime.now().strftime('%Y-%m-%d_%H')}.mp4"
    cmd = [
        "/opt/homebrew/bin/ffmpeg", "-y",
        "-loop", "1", "-i", str(image_path),
        "-i", str(audio_path),
        "-c:v", "libx264",
        "-tune", "stillimage",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=1080:1080",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    print(f"Reel video created: {video_path}")
    return video_path


# ── 5. Upload to Cloudinary ───────────────────────────────────────────────────

def upload_to_cloudinary(file_path: Path, resource_type: str = "image") -> str:
    import cloudinary
    import cloudinary.uploader
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET,
    )
    result = cloudinary.uploader.upload(
        str(file_path),
        folder="wisdom_daily",
        resource_type=resource_type,
        overwrite=True,
    )
    url = result["secure_url"]
    print(f"Uploaded to Cloudinary ({resource_type}): {url}")
    return url


# ── 5. Post to Instagram ──────────────────────────────────────────────────────

def get_page_token() -> str:
    """Exchange user token for the page access token required by Instagram publishing API."""
    r = requests.get(
        f"https://graph.facebook.com/v19.0/me/accounts",
        params={"access_token": INSTAGRAM_ACCESS_TOKEN},
    )
    r.raise_for_status()
    pages = r.json().get("data", [])
    if not pages:
        raise ValueError("No Facebook Pages found. Make sure your Instagram is connected to a Facebook Page.")
    return pages[0]["access_token"]


def post_to_instagram(video_path: Path, quote_data: dict) -> str:
    if not INSTAGRAM_ACCESS_TOKEN or not INSTAGRAM_ACCOUNT_ID:
        print("Instagram credentials not set, skipping.")
        return ""

    page_token = get_page_token()
    video_url = upload_to_cloudinary(video_path, resource_type="video")
    caption = (
        f'"{quote_data["quote"]}"\n\n'
        f"— {quote_data['source']}\n\n"
        f"{quote_data['reflection']}\n\n"
        f"#Vedanta #BhagavadGita #Upanishads #HinduPhilosophy #DailyWisdom "
        f"#Spirituality #AshtavakraGita #Advaita #SacredTexts #Consciousness"
    )

    # Post as Reel (video with audio)
    container_res = requests.post(
        f"https://graph.facebook.com/v19.0/{INSTAGRAM_ACCOUNT_ID}/media",
        data={
            "video_url": video_url,
            "media_type": "REELS",
            "caption": caption,
            "access_token": page_token,
        },
    )
    if not container_res.ok:
        print("Instagram API error:", container_res.json())
    container_res.raise_for_status()
    container_id = container_res.json()["id"]

    # Reels need processing time — poll until ready
    import time
    for _ in range(12):
        status_res = requests.get(
            f"https://graph.facebook.com/v19.0/{container_id}",
            params={"fields": "status_code", "access_token": page_token},
        )
        status = status_res.json().get("status_code")
        print(f"Reel status: {status}")
        if status == "FINISHED":
            break
        time.sleep(10)

    publish_res = requests.post(
        f"https://graph.facebook.com/v19.0/{INSTAGRAM_ACCOUNT_ID}/media_publish",
        data={"creation_id": container_id, "access_token": page_token},
    )
    if not publish_res.ok:
        print("Publish error:", publish_res.json())
    publish_res.raise_for_status()
    post_id = publish_res.json()["id"]
    print(f"Posted Reel to Instagram: {post_id}")
    return post_id


# ── 6. Post to Twitter/X ──────────────────────────────────────────────────────

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
    print(f"Generating wisdom post for {datetime.now().strftime('%Y-%m-%d %H:%M')}...")

    quote_data = generate_quote()
    print(f"\nQuote: {quote_data['quote'][:80]}...")
    print(f"Source: {quote_data['source']}")

    audio_path = generate_audio(quote_data)
    bg_bytes = generate_background_image()
    image_path = compose_image(quote_data, bg_bytes)
    video_path = create_reel_video(image_path, audio_path)

    post_to_instagram(video_path, quote_data)
    save_used_quote(quote_data["quote"])
    try:
        post_to_twitter(image_path, quote_data)
    except Exception as e:
        print(f"Twitter skipped: {e}")

    print("\nDone.")


if __name__ == "__main__":
    run()
