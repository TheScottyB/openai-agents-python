from my_openai_agents.core.agent import Agent

def main():
    # Create a simple agent
    agent = Agent(
        name="greeter",
        instructions="You are a friendly assistant that greets users."
    )
    
    # Print agent configuration
    print(f"Agent name: {agent.name}")
    print(f"Agent instructions: {agent.instructions}")
    print(f"Agent config: {agent.to_dict()}")

if __name__ == "__main__":
    main()
