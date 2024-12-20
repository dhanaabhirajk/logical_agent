# Logical agent

One Agent, but Multi agent. Reduce the error of Agent with Multi Logical Agents

Humans learn from others. Real Agent learns from Logical Agents

Logical agents solve the real project agent problems.

- One Agent, but multi agent Collabaration
- Improved Communication layer of agents
- High scalable for complex problems with just configuration
- No more prompts to be maintained
- Architecture fits into current Agentic Framworks

Logical Agent is open source architecture for LLM Agent

- [Key Features](#key-features)
- [Layers](#layers)
- [Applications](#applications)
- [Quickstart](#quickstart)
- [Roadmap](#roadmap)
- [Limitations](#limitations)

## Key Features

- Continuous training in inference
- Scalable based on logical agents
- Real time observation
- Multi Perspective
- Agent have high accuracy on τ-bench.
- Agent is made in a way where human in loop is invloved (interruption)

```text
Note: Predicts the letters in strawberry correctly. Try out "How many r's in strawberry?"
```

### Agent have high accuracy on τ-bench

A good threshold of logical agents will improve τ-bench.

### Agent is made in a way where human in loop is invloved (interruption)

Agent has two stages. Make agents to observe and say. Say can be changed with interruption.

```text
Core Logic: Humans learn from others. Real Agent learns from Logical Agents
```

## Layers

Logical Agent has core layers. The Logical Agent observes the user message and gives its real observation to real agent. The real observation message makes the real agent to learn with those observations from logical agent.(Humans learn from others. Real Agent learns from Logical Agents). The real agent inference is given to user when Listen message is sent.

- Logical Agent
- Perspective
- GroupManager
- Real Agent

### Logical Agent

Observes the user message and give observation to real observation. This simulates other people.

### Perspective

Perspective shows on how real agent needs to see a logical agent. Uses perspective to store the agent learnings.

### Group Manager

Indexes the logical agents to be active for observing. Uses agent perspective for indexing.

### Real Agent

Your focus is here. Use the real agent as a template and write your own agent. Use Autogen is only supported.

## Applications

- Create a real human agent (more logical agents, with good level of active logical agents and logical agent epochs)
- Solve Complex Mathematical Problem (By passing the the problem by sentence by sentence to the agent)
- Solve ARC problem with continous learning

## Quickstart

Clone the repo and change base directory to repo directory

Copy the .env.example to .env and place your `OPENAI_API_KEY`. The current implementation only supports openai provider.

First install all the package dependencies using poetry.

```bash
poetry install
```

### Running the agent

```bash
> poetry run python examples/main.py
  _                _           _      _                    _   
 | |    ___   __ _(_) ___ __ _| |    / \   __ _  ___ _ __ | |_ 
 | |   / _ \ / _` | |/ __/ _` | |   / _ \ / _` |/ _ \ '_ \| __|
 | |__| (_) | (_| | | (_| (_| | |  / ___ \ (_| |  __/ | | | |_ 
 |_____\___/ \__, |_|\___\__,_|_| /_/   \_\__, |\___|_| |_|\__|
             |___/                        |___/                
Started 1 Logical Agents
To exit, say exit
----------------------------------
> You: How many r's in strawberry?
> Agent: Yes, that's correct! The word "strawberry" contains three "r's."
> You: exit
```

## Roadmap

- Use Logical Agents to implement the above mentioned applications and improve the Logical Agents
- Improve history usage in logical agent as well
- Implement variety of perspectives like role model perspective and improve indexing
- Release this a pacakge to use across projects

### Limitations

- The response would include like repeating the message. The chat history usage needs to be optimized.
- The current agent is set with casual agent
