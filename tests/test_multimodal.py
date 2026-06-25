"""Image input in canonical messages (design §18.6).

Images ride alongside ``Message.content`` as typed ``ImageContent`` parts. Each
adapter translates them to its wire shape; providers that don't accept images
drop them so a text-only run is never broken.
"""

from __future__ import annotations

import base64

from agentkernel.context.manager import IMAGE_TOKEN_ESTIMATE, estimate_tokens
from agentkernel.providers import anthropic, openai
from agentkernel.providers.local import LocalProvider
from agentkernel.providers.openai import OpenAIProvider
from agentkernel.types import ImageContent, Message

PNG_BYTES = (  # 1x1 transparent PNG
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


# --- canonical type ----------------------------------------------------------


def test_image_content_round_trips_through_dict():
    img = ImageContent(data="abc", media_type="image/jpeg", kind="base64")
    assert ImageContent.from_dict(img.to_dict()) == img


def test_message_serializes_images():
    msg = Message(role="user", content="hi", images=[ImageContent(data="abc")])
    restored = Message.from_dict(msg.to_dict())
    assert restored.images == [ImageContent(data="abc")]
    assert restored.content == "hi"


def test_message_without_images_stays_empty():
    assert Message.from_dict(Message(role="user", content="x").to_dict()).images == []


def test_image_from_path_infers_media_type(tmp_path):
    p = tmp_path / "pixel.png"
    p.write_bytes(PNG_BYTES)
    img = ImageContent.from_path(p)
    assert img.kind == "base64"
    assert img.media_type == "image/png"
    assert base64.b64decode(img.data) == PNG_BYTES


def test_image_as_data_uri():
    b64 = ImageContent(data="QUJD", media_type="image/png")
    assert b64.as_data_uri() == "data:image/png;base64,QUJD"
    url = ImageContent.from_url("https://example.com/cat.png")
    assert url.as_data_uri() == "https://example.com/cat.png"


# --- adapter translation -----------------------------------------------------


def test_anthropic_renders_base64_image_block():
    msg = Message(
        role="user",
        content="what is this?",
        images=[ImageContent(data="QUJD", media_type="image/png")],
    )
    wire = anthropic.render_messages([msg])
    content = wire[0]["content"]
    assert content[0] == {"type": "text", "text": "what is this?"}
    assert content[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
    }


def test_anthropic_renders_url_image_block():
    msg = Message(role="user", images=[ImageContent.from_url("https://x/y.png")])
    block = anthropic.render_messages([msg])[0]["content"][0]
    assert block == {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}


def test_anthropic_plain_text_user_unchanged():
    wire = anthropic.render_messages([Message(role="user", content="hello")])
    assert wire == [{"role": "user", "content": "hello"}]


def test_openai_renders_image_url_part():
    msg = Message(
        role="user",
        content="caption",
        images=[ImageContent(data="QUJD", media_type="image/png")],
    )
    parts = openai.render_messages([msg])[0]["content"]
    assert parts[0] == {"type": "text", "text": "caption"}
    assert parts[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,QUJD"},
    }


def test_openai_include_images_false_drops_them():
    msg = Message(role="user", content="caption", images=[ImageContent(data="QUJD")])
    wire = openai.render_messages([msg], include_images=False)
    assert wire == [{"role": "user", "content": "caption"}]


# --- capability flags --------------------------------------------------------


def test_supports_images_flags():
    assert anthropic.AnthropicProvider("m").supports_images is True
    assert OpenAIProvider("m", api_key="k").supports_images is True
    # Local defaults to no image support (arbitrary models vary)...
    assert LocalProvider("m").supports_images is False
    # ...but can opt in for a vision-capable endpoint.
    assert LocalProvider("m", supports_images=True).supports_images is True


def test_with_model_preserves_image_capability():
    base = OpenAIProvider("m", api_key="k")
    assert base.with_model("other").supports_images is True


# --- token accounting --------------------------------------------------------


def test_cli_load_images_handles_paths_and_urls(tmp_path):
    from agentkernel.cli import _load_images

    assert _load_images(None) is None
    p = tmp_path / "pixel.png"
    p.write_bytes(PNG_BYTES)
    images = _load_images([str(p), "https://example.com/cat.jpg"])
    assert images[0].kind == "base64" and images[0].media_type == "image/png"
    assert images[1].kind == "url" and images[1].data == "https://example.com/cat.jpg"


def test_estimate_tokens_charges_for_images():
    text_only = Message(role="user", content="hello there")
    with_image = Message(
        role="user", content="hello there", images=[ImageContent(data="x")]
    )
    assert estimate_tokens(with_image) - estimate_tokens(text_only) == IMAGE_TOKEN_ESTIMATE
