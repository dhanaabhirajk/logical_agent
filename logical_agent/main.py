import os
import uuid
import asyncio
from typing import List
from dataclasses import dataclass
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

import openai
import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity
from pydantic_numpy import np_array_pydantic_annotated_typing

from autogen_core import (
    MessageContext,
    RoutedAgent,
    message_handler,
    TypeSubscription,
    TopicId,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core import SingleThreadedAgentRuntime
from autogen_ext.models.openai import OpenAIChatCompletionClient


@dataclass
class ListenMessage:
    content: str


@dataclass
class ObservationMessage:
    content: str


@dataclass
class RealObservationMessage:
    content: str


@dataclass
class SayMessage:
    content: str

openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
EMBEDDING_VECTOR_DIM = 1536

# @default_subscription
class LogicalAssistant(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("An assistant agent.")
        self._model_client = model_client
        self._chat_history: List[LLMMessage] = [
            SystemMessage(
                content="""Write a casual response.""",
            )
        ]

    @message_handler
    async def handle_observation_message(
        self, message: ObservationMessage, ctx: MessageContext
    ) -> None:
        print(f"Agent {self.id} Observing")
        self._chat_history.append(UserMessage(content=message.content, source="user"))
        result = await self._model_client.create(self._chat_history)
        print(f"\n{'-'*80}\nAssistant {self.type}:\n{result.content}")
        self._chat_history.append(AssistantMessage(content=result.content, source="assistant"))  # type: ignore
        await self.publish_message(RealObservationMessage(content=result.content), TopicId(type="assistant", source="assistant"))  # type: ignore


class Assistant(LogicalAssistant):
    @message_handler
    async def handle_real_observation_message(
        self, message: RealObservationMessage, ctx: MessageContext
    ) -> None:
        self._chat_history.append(UserMessage(content=message.content, source="user"))
        result = await self._model_client.create(self._chat_history)
        print(f"\n{'-'*80}\nAssistant {self.type}:\n{result.content}")
        self._chat_history.append(
            AssistantMessage(content=result.content, source="assistant")
        )

    @message_handler
    async def handle_message(self, message: ListenMessage, ctx: MessageContext) -> None:
        # find last assistant message
        self._chat_history.append(UserMessage(content=message.content, source="user"))
        result = await self._model_client.create(self._chat_history)
        print(f"\n{'-'*80}\nAssistant {self.type}:\n{result.content}")
        self._chat_history.append(
            AssistantMessage(content=result.content, source="assistant")
        )


class ChatHistory(BaseModel):
    role: str  # Role of the person sending the message (e.g., "user", "agent")
    message: str  # The message text
    vector: np_array_pydantic_annotated_typing(data_type=np.float32)  # The vector embedding of the message


class AgentPerspective(BaseModel):
    agent: object
    chat_history: List[ChatHistory] = []
    perspective_vector: np_array_pydantic_annotated_typing(data_type=np.float32) = Field(default_factory= lambda: np.zeros(EMBEDDING_VECTOR_DIM)) # the dimension of the embedding vector

    
    def add_chat_message(self, role: str, message: str, generate_embedding_fn):
        # Generate the embedding for the new message
        embedding = generate_embedding_fn(message)
        
        # Create a new ChatHistory object
        chat_history_entry = ChatHistory(role=role, message=message, vector=embedding)
        
        # Append it to the chat history
        self.chat_history.append(chat_history_entry)
        
        # Update the perspective vector by averaging the old vector and the new message vector
        if self.perspective_vector is not None:
            self.perspective_vector = (self.perspective_vector + embedding) / 2
        else:
            self.perspective_vector = embedding



class GroupManager:
    def __init__(self, runtime, topic_type, agent_factory, agent_class, num_of_logical_agents, num_active_logical_agents):
        self.runtime = runtime
        self.topic_type = topic_type
        self.agent_factory = agent_factory
        self.agent_class = agent_class
        self.agents = {}  # Dictionary with agent_id as key and AgentPerspective as value
        self.num_of_logical_agents = num_of_logical_agents
        self.num_active_logical_agents = num_active_logical_agents  # New parameter

    async def initialize_agents(self):
        for i in range(1, self.num_of_logical_agents + 1):
            await self.add_agent(f"logical_assistant_{i}")

    async def add_agent(self, agent_id):
        print(f"Agent {agent_id} added.")
        agent = await self.agent_class.register(
            self.runtime,
            agent_id,
            lambda: self.agent_factory(),
        )
        await self.runtime.add_subscription(
            TypeSubscription(topic_type=self.topic_type, agent_type=agent.type)
        )
        self.agents[agent_id] = AgentPerspective(agent=agent, chat_history=[], last_activity=None, activity_score=0)

    async def subscribe_agent(self, agent_id):
        print(f"Agent {agent_id} subscribed.")
        if agent_id in self.agents:
            agent_perspective = self.agents[agent_id]
            await self.runtime.add_subscription(
                TypeSubscription(topic_type=self.topic_type, agent_type=agent_perspective.agent.type)
            )

    async def unsubscribe_agent(self, agent_id):
        print(f"Agent {agent_id} unsubscribed.")

        if agent_id in self.agents:
            agent_perspective = self.agents[agent_id]
            await self.runtime.remove_subscription(
                TypeSubscription(topic_type=self.topic_type, agent_type=agent_perspective.agent.type)
            )

    async def observe(self, message, session_id, logical_agent_epochs, generate_embedding_fn):
        for _ in range(logical_agent_epochs):
            await self.runtime.publish_message(
                ObservationMessage(message),
                TopicId(type=self.topic_type, source=session_id),
                message_id="observation_message",
            )
            while [
                message
                for message in self.runtime.unprocessed_messages
                if message.message_id == "observation_message"
            ]:
                await asyncio.sleep(1)
            # Append the message to the chat history of all agents and update the perspective vector
            for agent_perspective in self.agents.values():
                agent_perspective.add_chat_message(role="user", message=message, generate_embedding_fn=generate_embedding_fn)
                agent_perspective.last_activity = datetime.now()
                agent_perspective.activity_score += 1
                if "recommend" in message.lower():
                    agent_perspective.topic_history.append("recommend")

    async def say_again(self, message, session_id):
        await self.runtime.publish_message(
            ListenMessage(message), TopicId(type="assistant", source=session_id)
        )

    async def index_agents(self, num_active_logical_agents: int):
        # Criteria: find the nearest agents based on their vector similarity (cosine similarity)
        agent_vectors = [
            (agent_id, perspective.perspective_vector)
            for agent_id, perspective in self.agents.items()
        ]
        
        # Compute cosine similarity between agent vectors
        similarities = []
        for agent_id, agent_vector in agent_vectors:
            similarity = cosine_similarity([agent_vector], [self.agents["logical_assistant_1"].perspective_vector])[0][0]
            similarities.append((agent_id, similarity))

        # Sort agents by their similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Activate top 'num_active_logical_agents' agents
        for agent_id, _ in similarities[:num_active_logical_agents]:
            await self.subscribe_agent(agent_id)


def generate_embedding(message: str):
    # Using OpenAI's embedding API to generate an embedding for the message
    response = openai_client.embeddings.create(
        input="Hello, my name is Abhi", model="text-embedding-ada-002"
    )
    embedding = np.array(response.data[0].embedding)
    return embedding

async def main():
    # Create a local embedded runtime.
    runtime = SingleThreadedAgentRuntime()

    # Register the assistant.
    assistant = await Assistant.register(
        runtime,
        "assistant",
        lambda: Assistant(
            OpenAIChatCompletionClient(
                model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")
            )
        ),
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type="assistant", agent_type=assistant.type)
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type="logical_assistant", agent_type=assistant.type)
    )

    # Create and manage logical assistants using GroupManager.
    total_logical_agents = 3
    num_active_logical_agents = 2  # For example, we want to activate the top 2 agents based on vector similarity
    logical_assistant_manager = GroupManager(
        runtime,
        topic_type="logical_assistant",
        agent_factory=lambda: LogicalAssistant(
            OpenAIChatCompletionClient(
                model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")
            )
        ),
        agent_class=LogicalAssistant,
        num_of_logical_agents=total_logical_agents,
        num_active_logical_agents=num_active_logical_agents
    )

    # Initialize all logical assistants.
    await logical_assistant_manager.initialize_agents()

    # Make the agents observe and say again.
    session_id = str(uuid.uuid4())
    message = "Hi my name is Abhi"
    logical_agent_epochs = 1

    await logical_assistant_manager.observe(message, session_id, logical_agent_epochs, generate_embedding_fn=generate_embedding)

    await logical_assistant_manager.say_again(message, session_id)

    # Index agents to recommend subscriptions based on vector similarity.
    await logical_assistant_manager.index_agents(num_active_logical_agents)

    # Start the runtime and stop when idle.
    runtime.start()
    await runtime.stop_when_idle()


if __name__ == "__main__":
    print("Application Started")
    asyncio.run(main())