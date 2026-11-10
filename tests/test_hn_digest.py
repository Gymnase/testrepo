import base64
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import mistralai.workflows as workflows
import pytest
from mistralai.workflows.testing import create_test_worker

from worker.workflows.hn_digest import (
    DigestContent,
    DigestStory,
    HnDigestInput,
    HnStory,
    HnTechDigestWorkflow,
    build_digest_html,
    build_digest_text,
    curate_and_summarize,
)


def make_story(story_id: int, title: str, url: str, score: int, comments: int) -> HnStory:
    return HnStory(id=story_id, title=title, url=url, score=score, author="a", age_hours=2.0, comment_count=comments)


STORIES = [
    make_story(1, "New chip announced", "https://example.com/chip", 200, 50),
    make_story(2, "Why I quit my job", "https://blog.example.com/quit", 150, 80),
]

PODCAST_SCRIPT = "Bonjour et bienvenue dans votre brief tech du jour. " + "Une phrase de plus. " * 100


def make_stories() -> list:
    return [story.model_copy() for story in STORIES]


def make_digest() -> DigestContent:
    return DigestContent(
        subject="HN Tech Digest - Test",
        stories=[
            DigestStory(
                title="Nouvelle puce",
                url="https://example.com/chip",
                discussion_url="https://news.ycombinator.com/item?id=1",
                summary="Résumé de la puce.",
                points=200,
                comment_count=50,
            )
        ],
        podcast_script=PODCAST_SCRIPT,
    )


def test_build_digest_html_contains_stories_and_audio_note() -> None:
    html = build_digest_html(make_digest().stories, has_audio=True)
    assert "Nouvelle puce" in html
    assert "https://news.ycombinator.com/item?id=1" in html
    assert "podcast audio" in html


def test_build_digest_text_lists_stories() -> None:
    text = build_digest_text(make_digest().stories)
    assert "Nouvelle puce" in text
    assert "https://example.com/chip" in text


async def test_curate_and_summarize_parses_llm_payload() -> None:
    payload = {
        "subject": "HN Tech Digest - Nouvelle puce annoncée",
        "stories": [{"index": 0, "title": "Nouvelle puce annoncée", "summary": "Un fabricant a annoncé une puce."}],
        "podcast_script": PODCAST_SCRIPT,
    }
    response = MagicMock()
    response.choices[0].message.content = json.dumps(payload)

    with patch("worker.workflows.hn_digest.get_mistral_client") as mock_client_factory:
        mock_client_factory.return_value.chat.complete_async = AsyncMock(return_value=response)
        digest = await curate_and_summarize(stories=make_stories(), language="fr", max_stories=10)

    assert digest.subject.startswith("HN Tech Digest")
    assert len(digest.stories) == 1
    assert digest.stories[0].url == "https://example.com/chip"
    assert digest.stories[0].discussion_url == "https://news.ycombinator.com/item?id=1"
    assert digest.stories[0].points == 200
    assert len(digest.podcast_script.split()) >= 180


def mock_activity(name: str):
    return workflows.activity(name=name)


@mock_activity("fetch-hn-front-page")
async def mock_fetch(limit: int = 40) -> list[HnStory]:
    return make_stories()


@mock_activity("curate-and-summarize")
async def mock_curate(stories: list[HnStory], language: str, max_stories: int) -> DigestContent:
    return make_digest()


@mock_activity("generate-podcast-audio")
async def mock_audio(script: str) -> str:
    return base64.b64encode(b"fake-mp3-bytes").decode()


@mock_activity("send-digest-email")
async def mock_send(recipient: str, subject: str, html_body: str, text_body: str, audio_base64: str | None) -> dict:
    return {"id": "msg-123", "threadId": "thread-1"}


MOCK_ACTIVITIES = [mock_fetch, mock_curate, mock_audio, mock_send]


async def run_digest(temporal_env, input_data: HnDigestInput) -> dict:
    async with create_test_worker(temporal_env, [HnTechDigestWorkflow], activities=MOCK_ACTIVITIES):
        return await temporal_env.client.execute_workflow(
            HnTechDigestWorkflow.run,
            input_data,
            id=f"hn-digest-test-{uuid.uuid4()}",
            task_queue="test-task-queue",
        )


async def test_workflow_dry_run_does_not_send_email(temporal_env) -> None:
    result = await run_digest(temporal_env, HnDigestInput(dry_run=True))
    assert result["email_sent"] is False
    assert result["email_id"] is None
    assert result["audio_base64"] == base64.b64encode(b"fake-mp3-bytes").decode()
    assert result["stories"][0]["title"] == "Nouvelle puce"


async def test_workflow_sends_email_when_not_dry_run(temporal_env) -> None:
    result = await run_digest(temporal_env, HnDigestInput(dry_run=False))
    assert result["email_sent"] is True
    assert result["email_id"] == "msg-123"


def test_no_import_errors() -> None:
    assert HnTechDigestWorkflow is not None
