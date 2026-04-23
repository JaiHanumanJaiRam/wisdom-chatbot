"""Test quote + image + audio generation without posting."""
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

from daily_post import generate_quote, generate_audio, generate_background_image, compose_image

print("Generating quote...")
quote_data = generate_quote()
print(f"\nQUOTE: {quote_data['quote']}")
print(f"SOURCE: {quote_data['source']}")
print(f"REFLECTION: {quote_data['reflection']}")

print("\nGenerating audio...")
audio_path = generate_audio(quote_data)

print("\nGenerating background image...")
bg_bytes = generate_background_image()

print("\nComposing final image...")
image_path = compose_image(quote_data, bg_bytes)

print(f"\nAll done! Check the output folder:")
print(f"  Image: {image_path}")
print(f"  Audio: {audio_path}")
