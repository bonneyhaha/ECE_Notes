import base64
from openai import OpenAI as HostedOpenAI
from llama_index.core.llms import (
    ChatMessage, ChatResponse, LLMMetadata,
    TextBlock, ImageBlock, MessageRole
)

class HostedOpenAIToLlamaIndex:
    def __init__(self, client: HostedOpenAI, model: str):
        self._client = client
        self._model = model

    def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        oai_messages = []
        for m in messages:
            content_items = []
            for block in m.content:
                if isinstance(block, TextBlock):
                    content_items.append({"type": "text", "text": block.text})
                elif isinstance(block, ImageBlock):
                    if isinstance(block.image, bytes):
                        b64_str = base64.b64encode(block.image).decode("utf-8")
                        content_items.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_str}"}
                        })
                    elif isinstance
                  (block.image, str) and block.image.startswith("data:image"):
                        content_items.append({"type": "image_url", "image_url": {"url": block.image}})
            oai_messages.append({"role": m.role.value, "content": content_items})

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=oai_messages,
            **kwargs
        )

        text_out = resp.choices[0].message.content

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=[TextBlock(text=text_out)]
            ),
            raw=resp
        )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self._model, is_chat_model=True, multimodal=True)
