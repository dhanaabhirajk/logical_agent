# logical_agent

One Agent, but Multi agent. Reduce the error of Agent with Multi Logical Agents

Logical agents solve the real project agent problems 

- One Agent, but multi agent Collabaration
- Improved Communication layer of agents
- High scalable for complex problems with just configuration
- No more prompts to be maintained
- Architecture fits into current Agentic Framworks


## Key Features

- Agent have high accuracy on τ-bench.
- Agent is made in a way where human in loop is invloved (interruption)
- Continuous training in inference

### Agent have high accuracy on τ-bench

A good threshold of logical agents will improve τ-bench.


### Agent is made in a way where human in loop is invloved (interruption)

Agent has two stages observation of logical agents and Listen user.

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

First install all the package dependencies using poetry.

```bash
poetry install
```

Run the agent

```bash
poetry run python app/main.py
```

## Future Works

- Use Logical Agents to implement the above mentioned applications and improve the Logical Agents
- Improve history usage in logical agent as well
- Implement variety of perspectives like role model perspective and improve indexing
- Release this a pacakge to use across projects
