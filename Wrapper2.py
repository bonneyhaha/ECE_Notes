import base64
from openai import OpenAI as HostedOpenAI
from llama_index.core.llms import ChatMessage, ChatResponse, LLMMetadata, TextBlock, ImageBlock, MessageRole
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI


class HostedOpenAIToLlamaIndex(LlamaIndexOpenAI):
    def __init__(self, client: HostedOpenAI, model: str):
        super().__init__(model=model)
        self.client = client
        self.model = model

    def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        # Convert LlamaIndex ChatMessage -> OpenAI format
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
                    elif isinstance(block.image, str) and block.image.startswith("data:image"):
                        content_items.append({"type": "image_url", "image_url": {"url": block.image}})
                    else:
                        raise ValueError("Unsupported image format")
            oai_messages.append({"role": m.role.value, "content": content_items})

        # Call hosted OpenAI
        resp = self.client.chat.completions.create(
            model=self.model,
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
        return LLMMetadata(model_name=self.model, is_chat_model=True, multimodal=True)
