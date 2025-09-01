from llama_index.multi_modal_llms import MultiModalLLM
from llama_index.core.multi_modal_llms import MultiModalChatMessage, MultiModalChatResponse

import base64
from openai import OpenAI
from llama_index.multi_modal_llms.base import MultiModalLLM
from llama_index.core.multi_modal_llms import (
    MultiModalChatMessage,
    MultiModalChatResponse,
    MultiModalLLMMetadata,
)


class HostedOpenAIToLlamaIndexMM(MultiModalLLM):
    def __init__(self, openai_client: OpenAI, model: str):
        self.client = openai_client
        self.model = model

    def chat(
        self,
        messages: list[MultiModalChatMessage],
        **kwargs
    ) -> MultiModalChatResponse:
        # Convert LlamaIndex messages -> OpenAI format
        oai_messages = []
        for m in messages:
            content_items = []
            for item in m.content:
                if item.type == "text":
                    content_items.append({"type": "text", "text": item.text})
                elif item.type == "image":
                    # convert to base64 if raw bytes given
                    if isinstance(item.image, bytes):
                        b64_str = base64.b64encode(item.image).decode("utf-8")
                        content_items.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_str}"}
                        })
                    elif isinstance(item.image, str) and item.image.startswith("data:image"):
                        # already base64 string
                        content_items.append({
                            "type": "image_url",
                            "image_url": {"url": item.image}
                        })
                    else:
                        raise ValueError("Unsupported image format")
            oai_messages.append({"role": m.role, "content": content_items})

        # Call your hosted OpenAI
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=oai_messages,
            **kwargs
        )

        content = resp.choices[0].message.content

        return MultiModalChatResponse(
            message=MultiModalChatMessage(role="assistant", content=[{"type": "text", "text": content}]),
            raw=resp
        )

    @property
    def metadata(self) -> MultiModalLLMMetadata:
        return MultiModalLLMMetadata(model_name=self.model, is_chat_model=True)


from openai import OpenAI
from llama_index.core.multi_modal_llms import MultiModalChatMessage

# Step 1: Your hosted OpenAI client
client = OpenAI(base_url="http://localhost:8000/v1", api_key="ollama")

# Step 2: Wrap into LlamaIndex multimodal LLM
mm_llm = HostedOpenAIToLlamaIndexMM(openai_client=client, model="my-mm-model")

# Step 3: Build input message with image + text
with open("cat.png", "rb") as f:
    image_bytes = f.read()

msg = MultiModalChatMessage(
    role="user",
    content=[
        {"type": "image", "image": image_bytes},
        {"type": "text", "text": "What is in this image?"}
    ]
)

# Step 4: Call
resp = mm_llm.chat([msg])
print(resp.message.content[0]["text"])
