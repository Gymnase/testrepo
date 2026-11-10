import base64
from datetime import UTC, datetime, timedelta
from typing import Any

import pydantic
from temporalio import workflow

import mistralai.workflows as workflows

with workflow.unsafe.imports_passed_through():
    import httpx

from mistralai.workflows import Depends
from mistralai.workflows.plugins.mistralai.connectors import ToolCallClient, connector, uses_connectors
from mistralai.workflows.plugins.mistralai.utils import get_mistral_client

gmail_connector = connector("gmail")

HN_API_BASE = "https://hacker-news.firebaseio.com/v0"
SUMMARY_MODEL = "mistral-medium-latest"
TTS_MODEL = "voxtral-mini-tts-3-0"
TTS_VOICE_ID = "voice_592"
PODCAST_MIN_WORDS = 180
PODCAST_MAX_WORDS = 300


class HnDigestInput(pydantic.BaseModel):
    recipient_email: str = "michel@puharre.com"
    max_stories: int = 10
    language: str = "fr"
    dry_run: bool = False


class HnStory(pydantic.BaseModel):
    id: int
    title: str
    url: str | None = None
    score: int = 0
    author: str = ""
    age_hours: float = 0.0
    comment_count: int = 0
    text: str | None = None


class DigestStory(pydantic.BaseModel):
    title: str
    url: str | None = None
    discussion_url: str
    summary: str
    points: int = 0
    comment_count: int = 0


class DigestContent(pydantic.BaseModel):
    subject: str
    stories: list[DigestStory]
    podcast_script: str


class DigestResult(pydantic.BaseModel):
    subject: str
    stories: list[DigestStory]
    podcast_script: str
    audio_base64: str | None = None
    email_sent: bool = False
    email_id: str | None = None


@workflows.activity(name="fetch-hn-front-page", start_to_close_timeout=timedelta(seconds=120))
async def fetch_front_page(limit: int = 40) -> list[HnStory]:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{HN_API_BASE}/topstories.json")
        response.raise_for_status()
        story_ids = response.json()[:limit]

        async def fetch_item(story_id: int) -> HnStory | None:
            item_response = await client.get(f"{HN_API_BASE}/item/{story_id}.json")
            item_response.raise_for_status()
            item = item_response.json()
            if not item or item.get("type") != "story" or item.get("deleted"):
                return None
            now = datetime.now(UTC)
            posted_at = datetime.fromtimestamp(item.get("time", now.timestamp()), tz=UTC)
            return HnStory(
                id=item["id"],
                title=item.get("title", ""),
                url=item.get("url"),
                score=item.get("score", 0),
                author=item.get("by", ""),
                age_hours=round(max((now - posted_at).total_seconds(), 0) / 3600, 1),
                comment_count=item.get("descendants", 0),
                text=item.get("text"),
            )

        stories = await asyncio.gather(*(fetch_item(story_id) for story_id in story_ids))
    return [story for story in stories if story is not None]


def _story_catalog(stories: list[HnStory]) -> str:
    lines = []
    for index, story in enumerate(stories):
        snippet = (story.text or "")[:300].replace("\n", " ")
        lines.append(
            f"[{index}] {story.title} | source: {story.url or 'Ask HN (text post)'} | "
            f"points: {story.score} | comments: {story.comment_count} | age: {story.age_hours}h"
            + (f" | excerpt: {snippet}" if snippet else "")
        )
    return "\n".join(lines)


def _curation_prompt(stories: list[HnStory], language: str, max_stories: int) -> str:
    return f"""You are the editor of a daily tech newsletter. Below is the current HackerNews front page.

STORIES:
{_story_catalog(stories)}

TASK:
1. Keep ONLY technology-industry news: products, companies, hardware, software, AI, security, programming, startups, tech policy with direct tech impact. EXCLUDE personal blogs, career/lifestyle posts, "Ask HN" opinion threads, and anything not about the tech sector.
2. Select the {max_stories} most newsworthy tech stories (fewer if fewer qualify).
3. Write, in {language}, for each selected story: a newsletter title, and a summary of 2 to 3 factual sentences covering what happened, who is concerned, and why it matters.
4. Write, in {language}, a podcast script of {PODCAST_MIN_WORDS} to {PODCAST_MAX_WORDS} words presenting the selected titles like a short news briefing: a one-sentence hook intro, then one or two spoken sentences per story in the same order, then a one-sentence outro. Plain spoken prose only, no markdown, no lists, no stage directions.
5. Write, in {language}, a short email subject line starting with "HN Tech Digest - ".

Return STRICT JSON with this exact shape, no markdown fence:
{{
  "subject": "...",
  "stories": [
    {{"index": <original index int>, "title": "...", "summary": "..."}}
  ],
  "podcast_script": "..."
}}"""


def _extract_json(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return dict(__import__("json").loads(cleaned))


@workflows.activity(name="curate-and-summarize", start_to_close_timeout=timedelta(seconds=300), retry_policy_max_attempts=3)
async def curate_and_summarize(stories: list[HnStory], language: str, max_stories: int) -> DigestContent:
    client = get_mistral_client()
    response = await client.chat.complete_async(
        model=SUMMARY_MODEL,
        temperature=0.3,
        messages=[
            {"role": "system", "content": "You are a rigorous tech news editor. Answer only with valid JSON."},
            {"role": "user", "content": _curation_prompt(stories, language, max_stories)},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or ""
    payload = _extract_json(content)

    selected: list[DigestStory] = []
    for entry in payload.get("stories", []):
        index = int(entry["index"])
        source = stories[index]
        selected.append(
            DigestStory(
                title=entry["title"],
                url=source.url,
                discussion_url=f"https://news.ycombinator.com/item?id={source.id}",
                summary=entry["summary"],
                points=source.score,
                comment_count=source.comment_count,
            )
        )
    return DigestContent(
        subject=payload.get("subject", "HN Tech Digest"),
        stories=selected,
        podcast_script=payload.get("podcast_script", ""),
    )


@workflows.activity(name="generate-podcast-audio", start_to_close_timeout=timedelta(seconds=300), retry_policy_max_attempts=3)
async def generate_podcast_audio(script: str) -> str:
    client = get_mistral_client()
    response = await client.audio.speech.complete_async(
        model=TTS_MODEL,
        input=script,
        voice_id=TTS_VOICE_ID,
        response_format="mp3",
    )
    return str(response.audio_data)


def build_digest_html(stories: list[DigestStory], has_audio: bool) -> str:
    items = []
    for story in stories:
        source_line = (
            f'<a href="{story.url}">{story.url}</a>' if story.url else "Discussion HackerNews"
        )
        items.append(
            f"""
            <li style="margin-bottom:18px;">
              <h3 style="margin:0 0 4px 0;"><a href="{story.discussion_url}">{story.title}</a></h3>
              <p style="margin:0 0 4px 0;color:#555;">{story.summary}</p>
              <p style="margin:0;font-size:12px;color:#888;">
                Source : {source_line} &middot; {story.points} points &middot;
                <a href="{story.discussion_url}">{story.comment_count} commentaires</a>
              </p>
            </li>
            """
        )
    audio_note = (
        '<p style="background:#f0f4ff;padding:10px;border-radius:6px;">'
        "\U0001F3A7 Le podcast audio du jour (1-2 min) est joint \u00e0 cet email.</p>"
        if has_audio
        else ""
    )
    return f"""
    <html>
      <body style="font-family:sans-serif;max-width:640px;">
        <h2 style="color:#ff6600;">HN Tech Digest</h2>
        {audio_note}
        <ol style="padding-left:20px;">{''.join(items)}</ol>
        <p style="font-size:12px;color:#888;">S\u00e9lection et r\u00e9sum\u00e9s g\u00e9n\u00e9r\u00e9s \u00e0 partir de la page d'accueil de HackerNews.</p>
      </body>
    </html>
    """


def build_digest_text(stories: list[DigestStory]) -> str:
    lines = ["HN Tech Digest", ""]
    for story in stories:
        lines.append(f"* {story.title}")
        lines.append(f"  {story.summary}")
        lines.append(f"  Source : {story.url or story.discussion_url}")
        lines.append(f"  Discussion : {story.discussion_url} ({story.points} points, {story.comment_count} commentaires)")
        lines.append("")
    return "\n".join(lines)


@workflows.activity(name="send-digest-email", start_to_close_timeout=timedelta(seconds=120), retry_policy_max_attempts=3)
async def send_digest_email(
    recipient: str,
    subject: str,
    html_body: str,
    text_body: str,
    audio_base64: str | None,
    gmail: ToolCallClient = Depends(gmail_connector),
) -> dict[str, Any]:
    attachments: list[dict[str, str]] = []
    if audio_base64:
        decoded_size = len(base64.b64decode(audio_base64))
        if decoded_size <= 3_000_000:
            attachments.append(
                {
                    "content": audio_base64,
                    "filename": "hn-tech-podcast.mp3",
                    "mimeType": "audio/mpeg",
                }
            )
    result = await gmail.call_tool(
        tool_name="send_email",
        arguments={
            "to": [recipient],
            "subject": subject,
            "body": text_body,
            "htmlBody": html_body,
            "attachments": attachments,
        },
    )
    if isinstance(result, dict):
        return result
    return {"id": getattr(result, "id", None), "threadId": getattr(result, "thread_id", None)}


@workflows.workflow.define(
    name="hn-tech-digest",
    on_behalf_of=True,
    workflow_display_name="HN Tech Digest",
    workflow_description=(
        "Reads the HackerNews front page, keeps the tech-industry stories, summarizes them in French, "
        "generates a 1-2 minute podcast audio, and emails the digest with the audio attached to your Gmail inbox. "
        "Runs on demand, when you trigger it."
    ),
    execution_timeout=timedelta(minutes=20),
)
@uses_connectors(gmail_connector)
class HnTechDigestWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: HnDigestInput) -> DigestResult:
        raw_stories = await fetch_front_page(limit=40)
        digest = await curate_and_summarize(stories=raw_stories, language=input.language, max_stories=input.max_stories)
        audio_base64 = await generate_podcast_audio(script=digest.podcast_script)

        email_id: str | None = None
        email_sent = False
        if not input.dry_run:
            result = await send_digest_email(
                recipient=input.recipient_email,
                subject=digest.subject,
                html_body=build_digest_html(digest.stories, has_audio=bool(audio_base64)),
                text_body=build_digest_text(digest.stories),
                audio_base64=audio_base64,
            )
            email_sent = True
            email_id = result.get("id") if isinstance(result, dict) else None

        return DigestResult(
            subject=digest.subject,
            stories=digest.stories,
            podcast_script=digest.podcast_script,
            audio_base64=audio_base64,
            email_sent=email_sent,
            email_id=email_id,
        )
