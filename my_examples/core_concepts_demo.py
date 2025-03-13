async def demonstrate_calculator():
    """Demonstrate the basic calculator tool usage."""
    
    print("=" * 80)
    print("CALCULATOR TOOL DEMONSTRATION")
    print("=" * 80)
    
    calculations = [
        "5 + 7",
        "12 * 6",
        "144 / 12",
        "(8 + 4) * 3"
    ]
    
    context = AgentContext(user_query="Perform some calculations")
    
    print("\nPerforming basic calculations:")
    for expression in calculations:
        print(f"\nCalculating: {expression}")
        try:
            result = await researcher_agent.tools[2]._run(expression)
            print(f"Result: {result.output}")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\nUsing calculator through the research agent:")
    query = "What is the square of 13 plus the cube of 5?"
    print(f"Query: {query}")
    
    try:
        research_response = await researcher_agent.run(query, context=context)
        print(f"Research agent response: {research_response}")
    except Exception as e:
        print(f"Error: {str(e)}")

async def demonstrate_coordinator():
    """Demonstrate the coordinator agent workflow with handoffs."""
    
    print("=" * 80)
    print("COORDINATOR AGENT WORKFLOW DEMONSTRATION")
    print("=" * 80)
    
    query = "I need a comprehensive analysis of quantum computing's impact on cryptography"
    print(f"\nCoordination task: {query}\n")
    
    try:
        # Demonstrate the full workflow with coordinator agent
        result = await run_research_workflow(query)
        
        # Display the result
        print("\nCOORDINATOR WORKFLOW RESULT:")
        print("=" * 80)
        print(result)
        print("=" * 80)
    except Exception as e:
        logger.error(f"Error during coordinator workflow: {str(e)}")
        print(f"\nError: {str(e)}")

async def demonstrate_research():
    """Demonstrate the research agent capabilities."""
    
    print("=" * 80)
    print("RESEARCH AGENT CAPABILITIES DEMONSTRATION")
    print("=" * 80)
    
    research_queries = [
        "What are the latest advancements in AI ethics?",
        "How does nuclear fusion differ from nuclear fission?"
    ]
    
    context = AgentContext(user_query="Research topics")
    
    for query in research_queries:
        print(f"\nResearch query: {query}")
        try:
            # Directly use the research agent
            result = await researcher_agent.run(query, context=context)
            
            # Display the structured result
            print("\nRESEARCH RESULTS:")
            print(f"Sources found: {len(result.sources)}")
            print(f"Key facts: {len(result.key_facts)}")
            print(f"Summary: {result.summary[:100]}...")
        except Exception as e:
            logger.error(f"Error during research: {str(e)}")
            print(f"Error: {str(e)}")

async def demonstrate_workflow():
    """Run a comprehensive demonstration of all core concepts."""
    
    # Setup logging
    setup_logger()
    logger.info("Starting core concepts demonstration")
    
    # Demonstrate calculator tool
    await demonstrate_calculator()
    
    # Demonstrate research capabilities
    await demonstrate_research()
    
    # Demonstrate coordinator workflow
    await demonstrate_coordinator()
    
    logger.info("Core concepts demonstration completed")
