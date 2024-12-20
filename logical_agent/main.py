import os
import uuid
import asyncio
import logging
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

logging.basicConfig(filename="app.log", level="DEBUG")

logger = logging.getLogger(__name__)


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

def generate_embedding(message: str):
    # Using OpenAI's embedding API to generate an embedding for the message
    response = openai_client.embeddings.create(
        input="Hello, my name is Abhi", model="text-embedding-ada-002"
    )
    embedding = np.array(response.data[0].embedding)
    return embedding

class ChatHistory(BaseModel):
    message: LLMMessage  # The message text
    vector: np_array_pydantic_annotated_typing(data_type=np.float32)  # The vector embedding of the message


class AgentPerspective(BaseModel):
    chat_history: List[ChatHistory] = Field(default_factory=list)
    perspective_vector: np_array_pydantic_annotated_typing(data_type=np.float32) = Field(default_factory= lambda: np.zeros(EMBEDDING_VECTOR_DIM)) # the dimension of the embedding vector

    
    def add_chat_message(self, message: LLMMessage):
        # Generate the embedding for the new message
        embedding = generate_embedding(message.content)
        
        # Create a new ChatHistory object
        chat_history_entry = ChatHistory(message=message, vector=embedding)
        
        # Append it to the chat history
        self.chat_history.append(chat_history_entry)
        
        # Update the perspective vector by averaging the old vector and the new message vector
        if self.perspective_vector is not None:
            self.perspective_vector = (self.perspective_vector + embedding) / 2
        else:
            self.perspective_vector = embedding

    def get_chat_history_for_llm(self):
        return [entry.message for entry in self.chat_history]


# @default_subscription
class LogicalAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, agent_perspective: AgentPerspective) -> None:
        super().__init__("Agent")
        self._model_client = model_client
        self._agent_perspective = agent_perspective
        self._chat_history: List[LLMMessage] = [
            SystemMessage(
                content="""Write a casual response.""",
            )
        ]

    @message_handler
    async def handle_observation_message(
        self, message: ObservationMessage, ctx: MessageContext
    ) -> None:
        logger.debug(f"Agent {self.id} Observing")

        self._agent_perspective.add_chat_message(UserMessage(content=message.content, source="user"))
        result = await self._model_client.create(self._agent_perspective.get_chat_history_for_llm())
        logger.debug(f"\n{'-'*80}\nAssistant {self.type}:\n{result.content}")
        self._agent_perspective.add_chat_message(AssistantMessage(content=result.content, source="assistant"))
        await self.publish_message(RealObservationMessage(content=result.content), TopicId(type="agent", source="assistant"))  # type: ignore


class RealAgent(LogicalAgent):
    @message_handler
    async def handle_real_observation_message(
        self, message: RealObservationMessage, ctx: MessageContext
    ) -> None:
        self._agent_perspective.add_chat_message(UserMessage(content=message.content, source="user"))
        result = await self._model_client.create(self._agent_perspective.get_chat_history_for_llm())
        logger.debug(f"\n{'-'*80}\nAssistant {self.type}:\n{result.content}")
        self._agent_perspective.add_chat_message(AssistantMessage(content=result.content, source="assistant"))

    @message_handler
    async def handle_message(self, message: ListenMessage, ctx: MessageContext) -> None:
        # find last agent message
        self._agent_perspective.add_chat_message(UserMessage(content=message.content, source="user"))
        result = await self._model_client.create(self._agent_perspective.get_chat_history_for_llm())
        logger.debug(f"\n{'-'*80}\nAssistant {self.type}:\n{result.content}")
        self._agent_perspective.add_chat_message(AssistantMessage(content=result.content, source="assistant"))


class GroupManager:
    def __init__(self, runtime, topic_type, agent_factory, agent_class, num_of_logical_agents, num_active_logical_agents):
        self.runtime = runtime
        self.topic_type = topic_type
        self.agent_factory = agent_factory
        self.agent_class = agent_class
        self.agents = {}  # Dictionary with agent_id as key and AgentPerspective as value
        self.agent_perspectives = {}
        self.num_of_logical_agents = num_of_logical_agents
        self.num_active_logical_agents = num_active_logical_agents  # New parameter

    async def initialize_agents(self):
        for i in range(1, self.num_of_logical_agents + 1):
            await self.add_agent(f"logical_agent_{i}")

    async def add_agent(self, agent_id):
        logger.debug(f"Agent {agent_id} added.")
        agent_perspective = AgentPerspective(chat_history=[])
        agent = await self.agent_class.register(
            self.runtime,
            agent_id,
            lambda: self.agent_factory(agent_perspective),
        )
        self.agent_perspectives[agent_id] = agent_perspective
        self.agents[agent_id] = agent
        await self.subscribe_agent(agent_id=agent_id)

    async def subscribe_agent(self, agent_id):
        logger.debug(f"Agent {agent_id} subscribed.")  
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            await self.runtime.add_subscription(
                TypeSubscription(topic_type=self.topic_type, agent_type=agent.type)
            )

    async def unsubscribe_agent(self, agent_id):
        logger.debug(f"Agent {agent_id} unsubscribed.")

        if agent_id in self.agents:
            agent = self.agents[agent_id]
            await self.runtime.remove_subscription(
                TypeSubscription(topic_type=self.topic_type, agent_type=agent.type)
            )

    async def observe(self, message, session_id, logical_agent_epochs, generate_embedding_fn):
        logger.debug(f"Topic of group manager: {self.topic_type}")
        for _ in range(logical_agent_epochs):
            await self.runtime.publish_message(
                ObservationMessage(content=message),
                TopicId(type=self.topic_type, source=session_id),
                message_id="observation_message",
            )
            while [
                message
                for message in self.runtime.unprocessed_messages
                if message.message_id == "observation_message"
            ]:
                await asyncio.sleep(1)

    async def say_again(self, message, session_id):
        await self.runtime.publish_message(
            ListenMessage(message), TopicId(type="agent", source=session_id)
        )

    async def index_agents(self, num_active_logical_agents: int, message: str):
        # Criteria: find the nearest agents based on their vector similarity (cosine similarity)
        agent_vectors = [
            (agent_id, perspective.perspective_vector)
            for agent_id, perspective in self.agent_perspectives.items()
        ]

        embedding = generate_embedding(message=message)
        
        # Compute cosine similarity between agent vectors
        similarities = []
        for agent_id, agent_vector in agent_vectors:
            similarity = cosine_similarity([agent_vector], [embedding])[0][0]
            similarities.append((agent_id, similarity))

        # Sort agents by their similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Activate top 'num_active_logical_agents' agents
        for agent_id, _ in similarities[:num_active_logical_agents]:
            try:
                await self.subscribe_agent(agent_id)
            except ValueError as e:
                logger.error(str(e))


async def main():
    # Create a local embedded runtime.
    runtime = SingleThreadedAgentRuntime()

    agent_perspective = AgentPerspective(chat_history=[])

    # Register the agent.
    agent = await RealAgent.register(
        runtime,
        "agent",
        lambda : RealAgent(
            OpenAIChatCompletionClient(
                model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")
            ),
            agent_perspective=agent_perspective
        ),
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type="agent", agent_type=agent.type)
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type="logical_agent", agent_type=agent.type)
    )

    # Create and manage logical agents using GroupManager.
    total_logical_agents = 1
    num_active_logical_agents = 2  # For example, we want to activate the top 2 agents based on vector similarity
    logical_agent_manager = GroupManager(
        runtime,
        topic_type="logical_agent",
        agent_factory=lambda agent_perspective: LogicalAgent(
            OpenAIChatCompletionClient(
                model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")
            ),
            agent_perspective=agent_perspective,
        ),
        agent_class=LogicalAgent,
        num_of_logical_agents=total_logical_agents,
        num_active_logical_agents=num_active_logical_agents
    )

    # Initialize all logical agents.
    await logical_agent_manager.initialize_agents()
    runtime.start()

    # Make the agents observe and say again.
    session_id = str(uuid.uuid4())
    message = "Hi my name is Abhi"
    logical_agent_epochs = 1

    await logical_agent_manager.observe(message, session_id, logical_agent_epochs, generate_embedding_fn=generate_embedding)

    # Index agents to recommend subscriptions based on vector similarity.
    await logical_agent_manager.index_agents(num_active_logical_agents, message=message)

    await logical_agent_manager.say_again(message, session_id)

    while runtime.unprocessed_messages:
        await asyncio.sleep(1)

    print(agent_perspective.chat_history[-1])
    # Start the runtime and stop when idle.
    await runtime.stop_when_idle()



if __name__ == "__main__":
    logger.debug("Application Started")
    asyncio.run(main())