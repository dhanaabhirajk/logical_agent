import asyncio
import os
import uuid
import logging

import typer

from logical_agent import (
    GroupManager,
    RealAgent,
    LogicalAgent,
    TypeSubscription,
    AgentPerspective,
)
from autogen_core import (
    SingleThreadedAgentRuntime,
)
from autogen_ext.models.openai import OpenAIChatCompletionClient

logging.basicConfig(filename="app.log", level="DEBUG")

app = typer.Typer()

@app.command()
def chat(
    total_logical_agents: int = typer.Option(1, help="Total number of logical agents."),
    num_active_logical_agents: int = typer.Option(2, help="Number of active logical agents based on vector similarity."),
):
    """
    Interact with the logical agent system in a loop until 'exit' is entered.
    """
    async def main():
        # Create a local embedded runtime.
        runtime = SingleThreadedAgentRuntime()

        agent_perspective = AgentPerspective(chat_history=[])

        # Register the agent.
        agent = await RealAgent.register(
            runtime,
            "agent",
            lambda: RealAgent(
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
        print(r"""  _                _           _      _                    _   
 | |    ___   __ _(_) ___ __ _| |    / \   __ _  ___ _ __ | |_ 
 | |   / _ \ / _` | |/ __/ _` | |   / _ \ / _` |/ _ \ '_ \| __|
 | |__| (_) | (_| | | (_| (_| | |  / ___ \ (_| |  __/ | | | |_ 
 |_____\___/ \__, |_|\___\__,_|_| /_/   \_\__, |\___|_| |_|\__|
             |___/                        |___/                """)
        print(f"Started {total_logical_agents} Logical Agents")
        print("To exit say, exit")
        print("----------------------------------")
        runtime.start()

        while True:
            message = input("You: ").strip()
            if message.lower() == "exit":
                break

            # Make the agents observe and say again.
            session_id = str(uuid.uuid4())
            logical_agent_epochs = 1

            await logical_agent_manager.observe(message, session_id, logical_agent_epochs)

            # Index agents to recommend subscriptions based on vector similarity.
            await logical_agent_manager.index_agents(num_active_logical_agents, message=message)

            await logical_agent_manager.say_again(message, session_id)

            while runtime.idle:
                await asyncio.sleep(1)

            # Retrieve and return the last chat history message.
            last_message = agent_perspective.chat_history[-1].message.content
            print(f"Agent: {last_message}")

        # Stop the runtime when exiting.
        await runtime.stop_when_idle()

    asyncio.run(main())

if __name__ == "__main__":
    app()
