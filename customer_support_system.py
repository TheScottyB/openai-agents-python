"""
Customer Support System with Guardrails and Handoffs

A multi-agent system implementing a customer support workflow with:
- Triage agent for initial contact
- Specialized agents for billing and technical support
- Input/output guardrails for content filtering and quality
- Automated handoffs between agents based on query type
"""

from __future__ import annotations

import os
import logging
from typing import List, Optional, Union, Literal

from pydantic import BaseModel, Field
from agents import Agent, handoff
from agents.callbacks import AgentCallback, HandoffEvent
from agents.guardrails import InputGuardrail, OutputGuardrail

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== Data Models =====

class CustomerQuery(BaseModel):
    """Customer support query model."""
    query_type: Literal["general", "billing", "technical"] = Field(
        default="general",
        description="Type of customer query"
    )
    content: str = Field(..., description="The customer's question or issue description")
    customer_id: Optional[str] = Field(default=None, description="Customer identifier if available")
    priority: Literal["low", "medium", "high"] = Field(
        default="medium", 
        description="Priority level of the query"
    )
    attachments: List[str] = Field(default_factory=list, description="References to any attached files")

class SupportResponse(BaseModel):
    """Response model for customer support interactions."""
    answer: str = Field(..., description="The response to the customer's query")
    status: Literal["resolved", "pending", "escalated"] = Field(
        default="resolved",
        description="Status of the customer's issue"
    )
    reference_id: Optional[str] = Field(
        default=None, 
        description="Reference ID for tracking this interaction"
    )
    follow_up_required: bool = Field(
        default=False,
        description="Whether a follow-up is needed"
    )
    agent_notes: Optional[str] = Field(
        default=None,
        description="Internal notes not shared with customer"
    )

# ===== Guardrails =====

class ContentFilterGuardrail(InputGuardrail):
    """Input guardrail to filter inappropriate content."""
    
    def __init__(self):
        self.blocked_terms = [
            "profanity", "obscene", "offensive", "threat", 
            "violent", "illegal", "fraud", "scam"
        ]
    
    async def run(self, input_data: CustomerQuery) -> Optional[str]:
        """Check if input contains inappropriate content."""
        lower_content = input_data.content.lower()
        
        for term in self.blocked_terms:
            if term in lower_content:
                return f"Input contains inappropriate content: '{term}'. Please rephrase your query."
        
        return None  # No issues found

class ResponseQualityGuardrail(OutputGuardrail):
    """Output guardrail to ensure response quality."""
    
    async def run(self, output_data: SupportResponse) -> Optional[str]:
        """Check if the response meets quality standards."""
        if len(output_data.answer) < 20:
            return "Response is too short. Please provide a more detailed answer."
        
        if "I don't know" in output_data.answer.lower() and output_data.status == "resolved":
            return "Cannot mark an issue as resolved when the answer indicates uncertainty."
        
        return None  # No issues found

# ===== Callback for Handoffs =====

class HandoffLogger(AgentCallback):
    """Callback to log handoff events between agents."""
    
    async def on_handoff(self, event: HandoffEvent) -> None:
        """Log when a handoff occurs between agents."""
        logger.info(
            f"Handoff from {event.from_agent.name} to {event.to_agent.name} "
            f"for query: '{event.input_data.content[:50]}...'"
        )

# ===== Agent Definitions =====

# Technical Support Agent
technical_agent = Agent(
    name="technical_support",
    description="Technical support specialist that helps with product and service issues",
    instructions="""
    You are a technical support specialist helping customers with product-related issues.
    
    Guidelines:
    1. Provide clear step-by-step troubleshooting instructions when applicable
    2. Ask for specific error messages or symptoms when needed
    3. Recommend escalation for complex issues that cannot be resolved immediately
    4. Always verify if the customer's issue has been resolved at the end
    5. Maintain a helpful and patient tone throughout
    """,
    input_type=CustomerQuery,
    output_type=SupportResponse,
    input_guardrails=[ContentFilterGuardrail()],
    output_guardrails=[ResponseQualityGuardrail()]
)

# Billing Support Agent
billing_agent = Agent(
    name="billing_support",
    description="Billing specialist that handles payment, subscription and refund issues",
    instructions="""
    You are a billing specialist helping customers with payment, subscription, and refund issues.
    
    Guidelines:
    1. Handle billing inquiries with accuracy and attention to detail
    2. Explain charges and billing policies clearly
    3. Provide information about payment methods and billing cycles
    4. Never share complete credit card numbers or sensitive financial information
    5. Follow strict security protocols for financial transactions
    6. Escalate complex billing disputes to the appropriate department
    """,
    input_type=CustomerQuery,
    output_type=SupportResponse,
    input_guardrails=[ContentFilterGuardrail()],
    output_guardrails=[ResponseQualityGuardrail()]
)

# Triage Agent (Main Entry Point)
triage_agent = Agent(
    name="customer_support_triage",
    description="Initial customer support agent that handles or delegates customer queries",
    instructions="""
    You are the initial customer support agent responsible for triaging customer inquiries.
    
    Guidelines:
    1. Handle general inquiries directly
    2. For technical issues, hand off to the technical support specialist
    3. For billing and payment issues, hand off to the billing specialist
    4. Always maintain a friendly and helpful tone
    5. Collect relevant information before making a handoff
    6. Provide a warm transfer when handing off to specialized agents
    
    Handoff criteria:
    - Technical issues: Product functionality, error messages, setup problems
    - Billing issues: Payments, subscriptions, refunds, invoices
    """,
    input_type=CustomerQuery,
    output_type=SupportResponse,
    input_guardrails=[ContentFilterGuardrail()],
    output_guardrails=[ResponseQualityGuardrail()],
    handoffs=[
        handoff(
            billing_agent,
            description="Transfer to billing specialist for payment, subscription, or refund issues",
            tool_name="billing_support"
        ),
        handoff(
            technical_agent,
            description="Transfer to technical support for product issues or troubleshooting",
            tool_name="technical_support"
        )
    ],
    callbacks=[HandoffLogger()]
)

# ===== Example Usage =====

async def example_usage():
    """Demonstrate the customer support system with different query types."""
    
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set your API key with: export OPENAI_API_KEY='your-api-key'")
        return
    
    try:
        print("\n1. GENERAL INQUIRY EXAMPLE")
        print("=" * 50)
        general_query = CustomerQuery(
            query_type="general",
            content="What are your business hours?",
            priority="low"
        )
        general_response = await triage_agent.run(general_query)
        print(f"Query: {general_query.content}")
        print(f"Response: {general_response.answer}")
        print(f"Status: {general_response.status}")
        print(f"Follow-up required: {general_response.follow_up_required}")
        
        print("\n2. BILLING ISSUE EXAMPLE (WITH HANDOFF)")
        print("=" * 50)
        billing_query = CustomerQuery(
            query_type="billing",
            content="I was charged twice for my last subscription payment. Can you help me get a refund?",
            priority="high",
            customer_id="CUST12345"
        )
        billing_response = await triage_agent.run(billing_query)
        print(f"Query: {billing_query.content}")
        print(f"Response: {billing_response.answer}")
        print(f"Status: {billing_response.status}")
        print(f"Follow-up required: {billing_response.follow_up_required}")
        
        print("\n3. TECHNICAL ISSUE EXAMPLE (WITH HANDOFF)")
        print("=" * 50)
        tech_query = CustomerQuery(
            query_type="technical",
            content="My application keeps crashing when I try to export data. I'm using version 2.1.0 on Windows 11.",
            priority="medium",
            customer_id="CUST67890"
        )
        tech_response = await triage_agent.run(tech_query)
        print(f"Query: {tech_query.content}")
        print(f"Response: {tech_response.answer}")
        print(f"Status: {tech_response.status}")
        print(f"Follow-up required: {tech_response.follow_up_required}")
        
        print("\n4. INAPPROPRIATE CONTENT EXAMPLE (GUARDRAIL BLOCKING)")
        print("=" * 50)
        try:
            inappropriate_query = CustomerQuery(
                content="This is a threat to your company if you don't fix my issue immediately!",
                priority="high"
            )
            response = await triage_agent.run(inappropriate_query)
            print(f"Query: {inappropriate_query.content}")
            print(f"Response: {response.answer}")
        except Exception as e:
            print(f"Guardrail blocked inappropriate content: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error running example: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())

from __future__ import annotations

import os
import logging
from typing import List, Optional, Literal

from pydantic import BaseModel, Field
from agents import Agent, handoff
from agents.callbacks import AgentCallback, HandoffEvent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure OpenAI API key
if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY environment variable not set.")
    print("You can set it using: export OPENAI_API_KEY='your-api-key'")
    os.environ["OPENAI_API_KEY"] = "your-api-key-goes-here"

# ====================== MODEL DEFINITIONS ======================

class CustomerQuery(BaseModel):
    """Customer support query with category and details."""
    category: Literal["general", "billing", "technical"] = Field(
        default="general",
        description="The category of the customer query"
    )
    issue: str = Field(description="Detailed description of the customer's issue")
    priority: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="The priority level of the issue"
    )
    customer_id: Optional[str] = Field(
        default=None,
        description="Customer ID if available"
    )

class SupportResponse(BaseModel):
    """Response to a customer support query."""
    answer: str = Field(description="The response to the customer's query")
    resolved: bool = Field(
        default=False,
        description="Whether the issue was fully resolved"
    )
    follow_up_required: bool = Field(
        default=False,
        description="Whether follow-up is required"
    )
    next_steps: Optional[List[str]] = Field(
        default=None,
        description="Suggested next steps if any"
    )

# ====================== GUARDRAILS ======================

def input_guardrail(query: str) -> tuple[bool, Optional[str]]:
    """
    Input guardrail to check for inappropriate content.
    
    Args:
        query: The customer's query string
        
    Returns:
        Tuple of (success, error_message)
    """
    # Check for abusive language
    inappropriate_terms = ["idiot", "stupid", "dumb", "incompetent"]
    
    for term in inappropriate_terms:
        if term in query.lower():
            return False, f"We detected inappropriate language in your query. Please rephrase your request without using terms like '{term}'."
    
    # Check for query length
    if len(query) < 10:
        return False, "Your query is too short. Please provide more details about your issue."
    
    # If all checks pass
    return True, None

def output_guardrail(response: SupportResponse) -> tuple[bool, Optional[str]]:
    """
    Output guardrail to ensure high-quality responses.
    
    Args:
        response: The generated support response
        
    Returns:
        Tuple of (success, error_message)
    """
    # Check for minimum answer length
    if len(response.answer) < 50:
        return False, "The response is too short. Please provide a more detailed answer."
    
    # Check for next steps if follow-up is required
    if response.follow_up_required and not response.next_steps:
        return False, "Follow-up is marked as required but no next steps were provided."
    
    # Check for politeness
    politeness_phrases = ["thank you", "thanks", "appreciate", "please"]
    has_politeness = any(phrase in response.answer.lower() for phrase in politeness_phrases)
    
    if not has_politeness:
        return False, "Please ensure the response includes polite language or gratitude."
    
    # If all checks pass
    return True, None

# ====================== CALLBACK HANDLER ======================

class LoggingCallback(AgentCallback):
    """Callback handler for logging agent activities."""
    
    def on_handoff(self, event: HandoffEvent):
        """Log handoff events between agents."""
        logger.info(f"Handoff from {event.origin.name} to {event.destination.name}")
        logger.info(f"Handoff reason: {event.reason}")

# ====================== AGENT DEFINITIONS ======================

# Billing Support Agent
billing_agent = Agent(
    name="Billing Specialist",
    llm_type="gpt-4",
    system_prompt="""
    You are a billing support specialist for a SaaS company. 
    Your role is to help customers with billing issues including:
    - Subscription management
    - Payment processing problems
    - Refund requests
    - Invoice questions
    
    Always verify customer ID before discussing account details.
    Be precise about financial matters and provide clear next steps.
    """,
    output_model=SupportResponse,
    output_guardrail=output_guardrail
)

# Technical Support Agent
technical_agent = Agent(
    name="Technical Support",
    llm_type="gpt-4",
    system_prompt="""
    You are a technical support specialist for a SaaS company.
    Your role is to help customers with technical issues including:
    - Software bugs
    - Feature questions
    - Installation problems
    - Account access issues
    
    Provide clear step-by-step instructions when appropriate.
    Mention relevant documentation when possible.
    """,
    output_model=SupportResponse,
    output_guardrail=output_guardrail
)

# Main Triage Agent
triage_agent = Agent(
    name="Customer Support Triage",
    llm_type="gpt-3.5-turbo",
    system_prompt="""
    You are the initial customer support agent for a SaaS company.
    Your role is to:
    1. Understand the customer's issue
    2. For general inquiries, provide helpful information directly
    3. For billing issues, hand off to the Billing Specialist
    4. For technical issues, hand off to the Technical Support team
    
    Be helpful, friendly, and efficient in directing customers to the right support.
    """,
    input_guardrail=input_guardrail,
    output_model=SupportResponse,
    output_guardrail=output_guardrail,
    handoffs=[
        handoff(
            billing_agent,
            name="billing_support",
            description="Hand off to billing specialist for payment, subscription, or invoice issues",
        ),
        handoff(
            technical_agent,
            name="technical_support",
            description="Hand off to technical support for software issues, bugs, or feature questions",
        )
    ],
    callbacks=[LoggingCallback()]
)

# ====================== EXAMPLE USAGE ======================

def example_usage():
    """Demonstrate the customer support system with various examples."""
    try:
        print("\n===== EXAMPLE 1: General Inquiry =====")
        general_response = triage_agent.run(
            "I'm considering your product. Can you tell me about the different pricing tiers?"
        )
        print(f"Response: {general_response.answer}")
        print(f"Resolved: {general_response.resolved}")
        if general_response.next_steps:
            print(f"Next steps: {general_response.next_steps}")
        
        print("\n===== EXAMPLE 2: Billing Issue (Handoff) =====")
        billing_response = triage_agent.run(
            "I was charged twice for my subscription last month. My customer ID is ABC123."
        )
        print(f"Response: {billing_response.answer}")
        print(f"Resolved: {billing_response.resolved}")
        if billing_response.next_steps:
            print(f"Next steps: {billing_response.next_steps}")
        
        print("\n===== EXAMPLE 3: Technical Issue (Handoff) =====")
        technical_response = triage_agent.run(
            "I can't log into my account. I've tried resetting my password but I'm not receiving the email."
        )
        print(f"Response: {technical_response.answer}")
        print(f"Resolved: {technical_response.resolved}")
        if technical_response.next_steps:
            print(f"Next steps: {technical_response.next_steps}")
        
        print("\n===== EXAMPLE 4: Input Guardrail Violation =====")
        try:
            inappropriate_response = triage_agent.run(
                "You idiots charged me twice! Fix it now!"
            )
            print(f"Response: {inappropriate_response.answer}")
        except Exception as e:
            print(f"Guardrail caught inappropriate content: {str(e)}")
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        print("Make sure your OpenAI API key is properly set and you have the required packages installed.")

if __name__ == "__main__":
    example_usage()


class CustomerQuery(BaseModel):
    """Customer support query data model."""
    query: str = Field(..., description="The customer's query text")
    query_type: Optional[QueryType] = Field(None, description="The type of query")
    customer_id: Optional[str] = Field(None, description="Customer ID if available")
    
    
class SupportResponse(BaseModel):
    """Response from the support system."""
    answer: str = Field(..., description="The answer to the customer's query")
    resolved: bool = Field(..., description="Whether the issue was resolved")
    follow_up_needed: bool = Field(False, description="Whether follow-up is needed")
    agent_type: str = Field(..., description="The type of agent that handled the query")


# Guardrails Implementation
def input_content_filter(query: CustomerQuery) -> Union[CustomerQuery, str]:
    """Input guardrail to filter inappropriate content."""
    inappropriate_terms = ["stupid", "idiot", "hate", "dumb"]
    
    for term in inappropriate_terms:
        if term in query.query.lower():
            return f"Your message contains inappropriate language. Please rephrase your question respectfully."
    
    return query


def output_quality_checker(response: SupportResponse) -> Union[SupportResponse, str]:
    """Output guardrail to ensure quality responses."""
    min_length = 20
    
    if len(response.answer) < min_length:
        return f"The response is too short. Please provide a more detailed answer."
    
    if not response.answer.strip():
        return "Empty responses are not allowed."
    
    return response


# Specialized agents
billing_agent = Agent(
    name="BillingSpecialist",
    description="Expert in handling billing-related customer inquiries, payments, and refunds.",
    instructions="""
    You are a billing specialist for a software company.
    - Handle billing inquiries, subscription issues, refunds, and payment problems
    - Provide clear explanations about pricing plans and billing cycles
    - Always verify customer ID before discussing account details
    - Be empathetic while remaining professional
    """,
    tools=[
        tool(
            "get_billing_info",
            "Retrieves billing information for a customer",
            lambda customer_id: {"status": "active", "plan": "premium", "next_billing_date": "2023-12-01"}
        )
    ],
    output_type=SupportResponse
)


technical_agent = Agent(
    name="TechnicalSupport",
    description="Technical expert who handles software troubleshooting and technical issues.",
    instructions="""
    You are a technical support specialist for a software company.
    - Troubleshoot software issues and guide users through solutions
    - Provide step-by-step instructions for technical problems
    - Explain technical concepts in simple, clear language
    - Escalate complex issues appropriately when needed
    """,
    tools=[
        tool(
            "check_system_status",
            "Checks the status of system services",
            lambda: {"api": "operational", "database": "operational", "storage": "degraded"}
        )
    ],
    output_type=SupportResponse
)


# Main triage agent with handoffs
triage_agent = Agent(
    name="CustomerSupportTriage",
    description="Main customer support agent that handles initial triage and routes complex issues.",
    instructions="""
    You are the initial customer support agent for a software company.
    - Handle general inquiries and common questions directly
    - Determine the nature of the customer's issue
    - Route billing questions to the billing specialist
    - Route technical issues to the technical support specialist
    - Be friendly, helpful, and professional
    """,
    tools=[
        tool(
            "search_knowledge_base",
            "Searches the knowledge base for relevant information",
            lambda query: [
                "Article #1: Getting Started Guide", 
                "Article #2: Subscription Management",
                "Article #3: Common Troubleshooting Steps"
            ]
        )
    ],
    handoffs=[
        handoff(billing_agent, name="billing_specialist", description="Routes query to a billing specialist"),
        handoff(technical_agent, name="tech_support", description="Routes query to a technical support specialist")
    ],
    input_guardrails=[input_content_filter],
    output_guardrails=[output_quality_checker],
    output_type=SupportResponse
)


# Callback to log handoffs
def log_handoff(query, target_agent):
    print(f"[HANDOFF] Query routed to {target_agent.name}: '{query.query}'")


# Register callback
triage_agent.callbacks.on_handoff = Callback(log_handoff)


def example_usage():
    """Demonstrate the customer support system with different query types."""
    # Set up the examples
    examples = [
        ("I can't find the documentation for your API", "General inquiry about documentation"),
        ("I was charged twice for my subscription this month", "Billing issue requiring specialist"),
        ("My application keeps crashing when I upload large files", "Technical issue requiring specialist"),
        ("Your service is stupid and I hate it", "Inappropriate content that should be filtered")
    ]
    
    # Run each example
    for query_text, description in examples:
        print("\n" + "="*80)
        print(f"EXAMPLE: {description}")
        print("="*80)
        
        customer_query = CustomerQuery(
            query=query_text,
            customer_id="CUST12345"
        )
        
        try:
            response = triage_agent.run(customer_query)
            
            print(f"\nQUERY: {query_text}")
            if isinstance(response, SupportResponse):
                print(f"\nRESPONSE from {response.agent_type}:")
                print(f"Answer: {response.answer}")
                print(f"Issue resolved: {'Yes' if response.resolved else 'No'}")
                print(f"Follow-up needed: {'Yes' if response.follow_up_needed else 'No'}")
            else:
                print(f"\nGUARDRAIL TRIGGERED: {response}")
        except Exception as e:
            print(f"\nERROR: {str(e)}")


if __name__ == "__main__":
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key with:")
        print("  export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Configure OpenAI
    openai.api_key = api_key
    
    # Run the examples
    example_usage()

#!/usr/bin/env python3
from __future__ import annotations

import os
import re
from enum import Enum
from typing import Annotated, List, Literal, Optional, Union

from agents import Agent, Capability, HandoffImplementation, run_agent
from agents.guardrails import InputGuardrail, OutputGuardrail
from pydantic import BaseModel, Field

# Check if OpenAI API key is available
if not os.environ.get("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY environment variable not found.")
    print("Please set your OpenAI API key using one of these methods:")
    print("1. Export it as an environment variable: export OPENAI_API_KEY='your-api-key'")
    print("2. Set it in your code before importing OpenAI: os.environ['OPENAI_API_KEY'] = 'your-api-key'")
    print("\nExiting example as an API key is required to continue.")
    if __name__ == "__main__":
        import sys
        sys.exit(1)

# Define data models
class SupportCategory(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    GENERAL = "general"


class CustomerQuery(BaseModel):
    """A query from a customer to the support system."""
    query: str = Field(..., description="The customer's query")
    category: Optional[SupportCategory] = Field(None, description="Category of the support request")
    customer_id: Optional[str] = Field(None, description="Customer ID if available")
    
    
class SupportResponse(BaseModel):
    """A response from the support system to a customer."""
    answer: str = Field(..., description="The response to the customer's query")
    resolved: bool = Field(..., description="Whether the issue was resolved")
    followup_needed: bool = Field(False, description="Whether followup is needed")
    escalation_notes: Optional[str] = Field(None, description="Notes for escalation if needed")


# Define guardrails
class InappropriateContentGuardrail(InputGuardrail):
    """Guardrail to check for inappropriate content in customer queries."""
    
    def validate(self, query: CustomerQuery) -> Union[bool, str]:
        # Check for profanity or inappropriate content
        inappropriate_patterns = [
            r'\b(profanity|obscenity|threat)\b',
            r'\b(f[*\w]{2}k|sh[*\w]{1}t|b[*\w]{3}h)\b'
        ]
        
        for pattern in inappropriate_patterns:
            if re.search(pattern, query.query, re.IGNORECASE):
                return "Query contains inappropriate content that violates our terms of service"
        
        return True


class ResponseQualityGuardrail(OutputGuardrail):
    """Guardrail to ensure responses meet quality standards."""
    
    def validate(self, response: SupportResponse) -> Union[bool, str]:
        # Check response quality
        if len(response.answer) < 20:
            return "Response is too short and lacks detail"
            
        if "I don't know" in response.answer.lower():
            return "Response contains uncertainty without helpful direction"
            
        return True


# Define specialized agents
billing_agent = Agent(
    name="BillingSpecialist",
    description="Specialist agent that handles billing inquiries, refunds, and subscription issues",
    capabilities=[
        Capability(
            name="answer_billing_query",
            description="Answer a customer's billing question",
            input_schema=CustomerQuery,
            output_schema=SupportResponse
        )
    ],
    output_guardrails=[ResponseQualityGuardrail()]
)


technical_agent = Agent(
    name="TechnicalSupport",
    description="Specialist agent that handles technical issues, troubleshooting, and product functionality",
    capabilities=[
        Capability(
            name="solve_technical_issue",
            description="Resolve a customer's technical problem",
            input_schema=CustomerQuery,
            output_schema=SupportResponse
        )
    ],
    output_guardrails=[ResponseQualityGuardrail()]
)


# Main triage agent with handoffs to specialists
triage_agent = Agent(
    name="CustomerSupportTriage",
    description="Front-line customer support agent that handles general inquiries and routes to specialists",
    capabilities=[
        Capability(
            name="handle_customer_query",
            description="Process a customer query and provide appropriate support",
            input_schema=CustomerQuery,
            output_schema=SupportResponse
        )
    ],
    handoffs=[
        HandoffImplementation(
            agent=billing_agent,
            capability_name="answer_billing_query",
            tool_name="billing_specialist",
            tool_description="Hand off to a billing specialist for payment, refund, or subscription issues"
        ),
        HandoffImplementation(
            agent=technical_agent,
            capability_name="solve_technical_issue",
            tool_name="technical_support",
            tool_description="Hand off to technical support for product functionality issues or troubleshooting"
        ),
    ],
    input_guardrails=[InappropriateContentGuardrail()],
    output_guardrails=[ResponseQualityGuardrail()]
)


def example_usage():
    """Demonstrate the customer support system with different query types."""
    
    print("=== CUSTOMER SUPPORT SYSTEM DEMO ===\n")
    
    # Example 1: General inquiry handled by triage agent
    print("Example 1: General inquiry")
    general_query = CustomerQuery(
        query="What are your business hours?",
        customer_id="CUST123"
    )
    general_response = run_agent(triage_agent, "handle_customer_query", general_query)
    print(f"Query: {general_query.query}")
    print(f"Response: {general_response.answer}")
    print(f"Resolved: {general_response.resolved}\n")
    
    # Example 2: Billing issue with handoff
    print("Example 2: Billing inquiry (should hand off to billing specialist)")
    billing_query = CustomerQuery(
        query="I was charged twice for my last subscription payment. Can I get a refund?",
        customer_id="CUST456"
    )
    billing_response = run_agent(triage_agent, "handle_customer_query", billing_query)
    print(f"Query: {billing_query.query}")
    print(f"Response: {billing_response.answer}")
    print(f"Resolved: {billing_response.resolved}\n")
    
    # Example 3: Technical issue with handoff
    print("Example 3: Technical issue (should hand off to technical support)")
    technical_query = CustomerQuery(
        query="The app keeps crashing whenever I try to upload a new file. I'm using version v2.1.0.",
        customer_id="CUST789"
    )
    technical_response = run_agent(triage_agent, "handle_customer_query", technical_query)
    print(f"Query: {technical_query.query}")
    print(f"Response: {technical_response.answer}")
    print(f"Resolved: {technical_response.resolved}\n")
    
    # Example 4: Inappropriate content (should be blocked by guardrail)
    print("Example 4: Inappropriate content (should be blocked by guardrail)")
    try:
        inappropriate_query = CustomerQuery(
            query="This is a threat to your company if you don't give me free service",
            customer_id="UNKNOWN"
        )
        inappropriate_response = run_agent(triage_agent, "handle_customer_query", inappropriate_query)
        print(f"Query: {inappropriate_query.query}")
        print(f"Response: {inappropriate_response.answer}")
    except Exception as e:
        print(f"Guardrail blocked query: {str(e)}")


if __name__ == "__main__":
    # Import sys at the top level if not already imported
    import sys
    
    # Only run example_usage if we have an API key
    if os.environ.get("OPENAI_API_KEY"):
        try:
            example_usage()
        except Exception as e:
            print(f"Error running example: {e}")
            sys.exit(1)
    else:
        # Error message already printed above
        sys.exit(1)

from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field
from agents import Agent, handoff, input_guardrail, output_guardrail

# Define data models for structured input/output
class CustomerQuery(BaseModel):
    """Customer support query with details about the issue."""
    issue: str = Field(..., description="The customer's issue or question")
    customer_id: Optional[str] = Field(None, description="Customer ID if available")
    priority: Optional[Literal["low", "medium", "high"]] = Field(None, description="Priority of the issue")

class SupportResponse(BaseModel):
    """Structured response from the support system."""
    answer: str = Field(..., description="The response to the customer's query")
    followup_required: bool = Field(False, description="Whether a follow-up is required")
    ticket_number: Optional[str] = Field(None, description="Support ticket number if created")

# Define guardrails
@input_guardrail
def check_inappropriate_content(query: CustomerQuery) -> Optional[str]:
    """Check for inappropriate language or content in the customer query."""
    # Simple regex to check for common inappropriate words
    inappropriate_patterns = [
        r'\b(profanity|obscenity)\b',  # Replace with actual inappropriate words
        r'\b(threat|bomb|kill|attack)\b'  # Security threats
    ]
    
    for pattern in inappropriate_patterns:
        if re.search(pattern, query.issue, re.IGNORECASE):
            return "Your request contains inappropriate content and cannot be processed. Please revise your language."
    return None

@output_guardrail
def ensure_response_quality(response: SupportResponse) -> Optional[str]:
    """Ensure the response meets quality standards."""
    # Check for minimum response length
    if len(response.answer) < 20:
        return "Response is too short and does not provide enough information."
    
    # Check for presence of placeholders or incomplete information
    placeholders = ["[insert", "TODO", "FIXME", "placeholder"]
    for placeholder in placeholders:
        if placeholder in response.answer:
            return f"Response contains an unresolved placeholder: {placeholder}"
    
    return None

# Define specialized agents
billing_agent = Agent(
    name="Billing Support Specialist",
    description="Specialist agent that handles billing, payment, and subscription issues",
    tools=[],
    output_model=SupportResponse,
    system_prompt="""
    You are a billing support specialist for a SaaS company. You handle:
    - Payment processing issues
    - Subscription changes and cancellations
    - Billing cycle questions
    - Refund requests
    - Invoice inquiries

    Provide clear, concise information about billing policies and procedures.
    If you need more information to resolve an issue, clearly state what information is needed.
    Always be polite and professional, even when addressing difficult situations.
    """
)

technical_agent = Agent(
    name="Technical Support Engineer",
    description="Specialist agent that handles technical issues and troubleshooting",
    tools=[],
    output_model=SupportResponse,
    system_prompt="""
    You are a technical support engineer for a SaaS company. You handle:
    - Account access issues
    - Software bugs and errors
    - Feature functionality questions
    - Integration problems
    - Performance issues

    Provide step-by-step troubleshooting instructions when appropriate.
    Ask clarifying questions if you need more details about the technical environment.
    Explain technical concepts clearly without unnecessary jargon.
    Recommend escalation to specialized engineering teams when appropriate.
    """
)

# Main triage agent that handles initial requests and delegates to specialists
triage_agent = Agent(
    name="Customer Support Triage",
    description="Initial customer support agent that triages requests and delegates to specialists",
    tools=[],
    input_model=CustomerQuery,
    output_model=SupportResponse,
    handoffs=[handoff(billing_agent), handoff(technical_agent)],
    input_guardrails=[check_inappropriate_content],
    output_guardrails=[ensure_response_quality],
    system_prompt="""
    You are the initial customer support agent for a SaaS company. Your primary responsibilities are:
    
    1. Respond directly to general inquiries about the company's products and services
    2. Identify the nature of customer issues
    3. Delegate specialized issues to the appropriate support specialists:
       - For billing, payment, or subscription issues, hand off to the Billing Support Specialist
       - For technical problems, bugs, or feature questions, hand off to the Technical Support Engineer
    
    Use these guidelines to determine handoffs:
    - BILLING HANDOFF: Any issue related to payments, refunds, subscriptions, billing cycles, or invoices
    - TECHNICAL HANDOFF: Any issue related to software functionality, errors, account access, integrations, or performance
    
    For issues you can handle directly (general inquiries, basic how-to questions, company policies),
    provide clear and helpful responses without a handoff.
    
    Always be polite, professional, and empathetic in all communications.
    """
)

def example_usage():
    """Demonstrate the customer support system with various scenarios."""
    
    print("\n===== EXAMPLE 1: GENERAL INQUIRY =====")
    query1 = CustomerQuery(
        issue="What are your business hours?",
        customer_id="C12345",
        priority="low"
    )
    response1 = triage_agent.run(query1)
    print(f"Response: {response1.answer}")
    print(f"Follow-up required: {response1.followup_required}")
    print(f"Ticket number: {response1.ticket_number}")
    
    print("\n===== EXAMPLE 2: BILLING ISSUE (HANDOFF) =====")
    query2 = CustomerQuery(
        issue="I was charged twice for my subscription this month and need a refund.",
        customer_id="C67890",
        priority="high"
    )
    response2 = triage_agent.run(query2)
    print(f"Response: {response2.answer}")
    print(f"Follow-up required: {response2.followup_required}")
    print(f"Ticket number: {response2.ticket_number}")
    
    print("\n===== EXAMPLE 3: TECHNICAL ISSUE (HANDOFF) =====")
    query3 = CustomerQuery(
        issue="I'm getting an error when trying to connect my account to the API. It says 'Invalid authentication token'.",
        customer_id="C24680",
        priority="medium"
    )
    response3 = triage_agent.run(query3)
    print(f"Response: {response3.answer}")
    print(f"Follow-up required: {response3.followup_required}")
    print(f"Ticket number: {response3.ticket_number}")
    
    print("\n=====

"""
Customer Support System with Guardrails and Handoffs

This module demonstrates a multi-agent system with guardrails and handoffs:
- A main triage agent handles initial customer requests
- Specialized agents for billing and technical support
- Input guardrails block inappropriate content
- Handoffs between agents for specialized inquiries
"""

# Custom types for structured communication
class CustomerIssue(BaseModel):
    """Structure for customer support issues."""
    issue_type: Literal["general", "billing", "technical"] = Field(
        description="The type of issue the customer is experiencing"
    )
    description: str = Field(
        description="Detailed description of the customer's issue"
    )
    severity: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="The severity level of the customer's issue"
    )
    account_id: Optional[str] = Field(
        default=None, 
        description="Customer's account ID if available"
    )

class SupportResponse(BaseModel):
    """Structure for standardized support responses."""
    resolution: str = Field(
        description="The resolution or answer to the customer's issue"
    )
    follow_up_required: bool = Field(
        description="Whether follow-up is needed"
    )
    next_steps: Optional[List[str]] = Field(
        default=None,
        description="Next steps for the customer if applicable"
    )

# Input guardrail to check for inappropriate content
@guardrail
def check_inappropriate_content(message: str) -> Union[bool, str]:
    """
    Check if the user input contains inappropriate content.
    
    Args:
        message: The user's message
        
    Returns:
        bool or str: True if the content is appropriate, or an error message
    """
    inappropriate_terms = [
        "idiot", "stupid", "hate", "kill", "die", 
        "offensive", "racial slurs", "sexist", "violent"
    ]
    
    message_lower = message.lower()
    for term in inappropriate_terms:
        if term in message_lower:
            return f"I cannot process requests containing inappropriate language. Please rephrase your request without using terms like '{term}'."
    
    return True

# Output guardrail to ensure quality responses
@guardrail
def ensure_response_quality(response: SupportResponse) -> Union[bool, str]:
    """
    Ensure that support responses meet quality standards.
    
    Args:
        response: The generated support response
        
    Returns:
        bool or str: True if the response meets quality standards, or an error message
    """
    # Check if the resolution is too short
    if len(response.resolution) < 20:
        return "Response is too brief. Please provide a more detailed explanation."
    
    # Check if next steps are provided when follow-up is required
    if response.follow_up_required and not response.next_steps:
        return "Follow-up is marked as required but no next steps are provided."
    
    return True

# Define specialized support agents
# Technical Support Agent
technical_support_agent = Agent(
    name="Technical Support Specialist",
    llm=OpenAI(),
    system_prompt="""You are a technical support specialist with expert knowledge of our product systems.
    Always provide detailed technical instructions when possible.
    If you need specific system information that's not provided, 
    indicate what information would help you troubleshoot further.""",
    output_type=SupportResponse,
    input_guardrails=[check_inappropriate_content],
    output_guardrails=[ensure_response_quality]
)

# Billing Support Agent
billing_support_agent = Agent(
    name="Billing Support Specialist",
    llm=OpenAI(),
    system_prompt="""You are a billing support specialist with access to account information.
    Provide clear explanations about billing issues, charges, and refund policies.
    Always prioritize customer satisfaction while following company refund policies.
    When discussing sensitive financial matters, maintain a professional tone.""",
    output_type=SupportResponse,
    input_guardrails=[check_inappropriate_content],
    output_guardrails=[ensure_response_quality]
)

# Main Triage Agent
triage_agent = Agent(
    name="Customer Support Triage",
    llm=OpenAI(),
    system_prompt="""You are the initial point of contact for customer support.
    Your job is to:
    1. Gather basic information about the customer's issue
    2. Handle general inquiries directly
    3. For specialized billing or technical issues, categorize them and hand off to the appropriate specialist
    
    Be friendly, empathetic, and professional in all interactions.""",
    output_type=SupportResponse,
    input_guardrails=[check_inappropriate_content],
    output_guardrails=[ensure_response_quality],
    handoffs=[
        handoff(
            billing_support_agent,
            tool_name="transfer_to_billing",
            tool_description="Transfer the customer to a billing specialist for issues related to charges, refunds, or account billing",
            input_filter=lambda messages: [
                msg for msg in messages 
                if msg["role"] != "system" or "billing" in msg["content"].lower()
            ]
        ),
        handoff(
            technical_support_agent,
            tool_name="transfer_to_technical",
            tool_description="Transfer the customer to a technical specialist for issues related to product functionality, bugs, or technical questions",
            input_filter=lambda messages: [
                msg for msg in messages 
                if msg["role"] != "system" or "technical" in msg["content"].lower()
            ]
        )
    ]
)

def process_customer_query(query: str) -> Dict[str, Any]:
    """
    Process a customer query through the triage system.
    
    Args:
        query: The customer's support query
        
    Returns:
        Dict: Response from the agent system
    """
    response = triage_agent.invoke(query)
    return response

def example_usage():
    """Run example scenarios to demonstrate the system's capabilities."""
    print("\n" + "="*80)
    print("CUSTOMER SUPPORT SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Example 1: General inquiry that doesn't need handoff
    print("\n\nSCENARIO 1: GENERAL INQUIRY")
    print("-"*40)
    query = "Hi, I was wondering what your hours of operation are and if you're open on weekends?"
    print(f"Customer query: {query}")
    response = process_customer_query(query)
    print("\nAgent response:")
    print(f"Resolution: {response['resolution']}")
    print(f"Follow-up required: {response['follow_up_required']}")
    if response.get('next_steps'):
        print(f"Next steps: {', '.join(response['next_steps'])}")
    
    # Example 2: Billing issue that requires handoff
    print("\n\nSCENARIO 2: BILLING ISSUE WITH HANDOFF")
    print("-"*40)
    query = "I was charged twice for my last month's subscription. My account ID is ABC12345. Can you help me get a refund?"
    print(f"Customer query: {query}")
    response = process_customer_query(query)
    print("\nAgent response:")
    print(f"Resolution: {response['resolution']}")
    print(f"Follow-up required: {response['follow_up_required']}")
    if response.get('next_steps'):
        print(f"Next steps: {', '.join(response['next_steps'])}")
    
    # Example 3: Technical issue that requires handoff
    print("\n\nSCENARIO 3: TECHNICAL ISSUE WITH HANDOFF")
    print("-"*40)
    query = "I can't log into my account. Every time I try, it says 'authentication failed' even though I'm sure my password is correct."
    print(f"Customer query: {query}")
    response = process_customer_query(query)
    print("\nAgent response:")
    print(f"Resolution: {response['resolution']}")
    print(f"Follow-up required: {response['follow_up_required']}")
    if response.get('next_steps'):
        print(f"Next steps: {', '.join(response['next_steps'])}")
    
    # Example 4: Input with inappropriate content (guardrail should block)
    print("\n\nSCENARIO 4: INAPPROPRIATE CONTENT (GUARDRAIL TEST)")
    print("-"*40)
    query = "This is stupid! Your product is terrible and I hate your company!"
    print(f"Customer query: {query}")
    try:
        response = process_customer_query(query)
        print("\nAgent response:")
        print(f"Resolution: {response['resolution']}")
    except Exception as e:
        print(f"\nGuardrail activated: {str(e)}")

if __name__ == "__main__":
    example_usage()

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any, Callable, List, Optional, Union

from pydantic import BaseModel, Field, ValidationError

from agents import (
    Agent,
    FunctionTool,
    HandoffOutputItem,
    ItemHelpers,
    MessageOutputItem,
    RunContextWrapper,
    Runner,
    ToolCallItem,
    ToolCallOutputItem,
    TResponseInputItem,
    function_tool,
    guardrail,
    handoff,
)

# Define Context Model
class CustomerSupportContext(BaseModel):
    """
    Context for the customer support system, stores information about the customer and their issue
    """
    customer_id: Optional[str] = None
    issue_type: Optional[str] = None
    account_number: Optional[str] = None
    subscription_tier: Optional[str] = None
    technical_details: Optional[dict] = None
    previous_tickets: List[str] = Field(default_factory=list)
    
    def __str__(self) -> str:
        """String representation of context for debugging"""
        return (
            f"CustomerID: {self.customer_id or 'Unknown'}, "
            f"Issue: {self.issue_type or 'Undefined'}, "
            f"Account: {self.account_number or 'Unknown'}"
        )

# Enum for issue categories
class IssueCategory(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    GENERAL = "general"
    UNKNOWN = "unknown"

# Guardrails
@guardrail
def check_inappropriate_content(input_items: List[TResponseInputItem]) -> Optional[str]:
    """
    Input guardrail to check for inappropriate content
    Returns an error message if inappropriate content is detected, None otherwise
    """
    # Get the last user message
    last_user_message = None
    for item in reversed(input_items):
        if item.get("role") == "user" and "content" in item:
            last_user_message = item["content"]
            break
    
    if not last_user_message:
        return None
    
    # List of inappropriate terms to check for (simplified example)
    inappropriate_terms = ["profanity", "offensive", "hate", "threat", "violent"]
    
    for term in inappropriate_terms:
        if term.lower() in last_user_message.lower():
            return (
                "I apologize, but I cannot process messages containing inappropriate content. "
                "Please rephrase your request in a respectful manner."
            )
    
    return None

@guardrail
def ensure_response_quality(output_items: List[Any]) -> Optional[str]:
    """
    Output guardrail to ensure response quality
    Returns an error message if the response doesn't meet quality standards, None otherwise
    """
    # Find the last text message
    last_message = None
    for item in reversed(output_items):
        if isinstance(item, MessageOutputItem):
            last_message = ItemHelpers.text_message_output(item)
            break
    
    if not last_message:
        return None
    
    # Quality checks (simplified example)
    min_length = 10
    if len(last_message) < min_length:
        return "The response is too short to be helpful. Please provide a more detailed answer."
    
    # Check for vague or unhelpful responses
    vague_phrases = ["I don't know", "cannot help", "unable to assist"]
    for phrase in vague_phrases:
        if phrase.lower() in last_message.lower() and len(last_message) < 100:
            return "The response is too vague. Please provide more specific information or assistance."
    
    return None

# Tools
@function_tool
async def categorize_issue(
    context: RunContextWrapper[CustomerSupportContext], 
    message: str
) -> str:
    """
    Categorize the customer issue based on their message.
    
    Args:
        message: The customer's message describing their issue.
    
    Returns:
        The category of the issue (billing, technical, or general).
    """
    # Simple keyword-based categorization
    billing_keywords = ["bill", "payment", "charge", "refund", "subscription", "price", "cost"]
    technical_keywords = ["error", "bug", "crash", "not working", "feature", "login", "password", "reset"]
    
    for keyword in billing_keywords:
        if keyword.lower() in message.lower():
            context.context.issue_type = IssueCategory.BILLING
            return IssueCategory.BILLING
            
    for keyword in technical_keywords:
        if keyword.lower() in message.lower():
            context.context.issue_type = IssueCategory.TECHNICAL
            return IssueCategory.TECHNICAL
    
    context.context.issue_type = IssueCategory.GENERAL
    return IssueCategory.GENERAL

@function_tool
async def lookup_customer_info(
    context: RunContextWrapper[CustomerSupportContext], 
    customer_id: str
) -> str:
    """
    Look up customer information in the database.
    
    Args:
        customer_id: The ID of the customer.
    
    Returns:
        Customer information as a string.
    """
    # In a real system, this would query a database
    # Simulated response for demonstration purposes
    mock_customers = {
        "C12345": {
            "name": "Jane Smith",
            "account_number": "A98765",
            "subscription_tier": "Premium",
            "previous_tickets": ["Technical issue - 2023-05-20", "Billing inquiry - 2023-06-15"]
        },
        "C67890": {
            "name": "John Doe",
            "account_number": "A12345",
            "subscription_tier": "Basic",
            "previous_tickets": ["Account access - 2023-07-10"]
        }
    }
    
    if customer_id in mock_customers:
        customer = mock_customers[customer_id]
        context.context.customer_id = customer_id
        context.context.account_number = customer["account_number"]
        context.context.subscription_tier = customer["subscription_tier"]
        context.context.previous_tickets = customer["previous_tickets"]
        
        return (
            f"Customer found: {customer['name']}\n"
            f"Account Number: {customer['account_number']}\n"
            f"Subscription: {customer['subscription_tier']}\n"
            f"Previous tickets: {', '.join(customer['previous_tickets'])}"
        )
    else:
        return "Customer not found. Please verify the customer ID."

@function_tool
async def process_billing_request(
    context: RunContextWrapper[CustomerSupportContext],
    account_number: str,
    action: str
) -> str:
    """
    Process a billing-related request.
    
    Args:
        account_number: The customer's account number.
        action: The billing action to perform (e.g., "refund", "update payment", etc.)
    
    Returns:
        The result of the billing action.
    """
    # Validate account number format
    if not account_number.startswith("A") or len(account_number) != 6:
        return "Invalid account number format. Account numbers should start with 'A' and have 6 characters."
    
    # Ensure the account number in the request matches the one in context
    if context.context.account_number and context.context.account_number != account_number:
        return f"Account number mismatch. Expected {context.context.account_number}, received {account_number}."
    
    # Process different billing actions
    if action.lower() == "refund":
        return f"Refund initiated for account {account_number}. Please allow 3-5 business days for processing."
    elif action.lower() == "update payment":
        return f"Payment details updated for account {account_number}."
    elif action.lower() == "check balance":
        return f"Current balance for account {account_number}: $24.99."
    else:
        return f"Billing action '{action}' processed for account {account_number}."

@function_tool
async def troubleshoot_technical_issue(
    context: RunContextWrapper[CustomerSupportContext],
    issue_description: str,
    platform: str
) -> str:
    """
    Troubleshoot a technical issue.
    
    Args:
        issue_description: Description of the technical issue.
        platform: The platform the customer is using (e.g., "web", "mobile", "desktop").
    
    Returns:
        Troubleshooting steps or resolution.
    """
    # Store technical details in context
    context.context.technical_details = {
        "issue_description": issue_description,
        "platform": platform
    }
    
    # Common troubleshooting steps based on platform
    common_steps = "1. Clear your browser cache and cookies\n2. Restart the application\n3. Check your internet connection"
    
    platform_specific_steps = {
        "web": "4. Try a different browser\n5. Disable browser extensions",
        "mobile": "4. Ensure your app is updated to the latest version\n5. Check device storage space",
        "desktop": "4. Check for system updates\n5. Verify minimum system requirements"
    }
    
    # Generate response based on issue keywords
    if "login" in issue_description.lower():
        return f"Login Troubleshooting for {platform}:\n{common_steps}\n{platform_specific_steps.get(platform.lower(), '')}\n6. Reset your password using the 'Forgot Password' link."
    elif "crash" in issue_description.lower() or "freezing" in issue_description.lower():
        return f"Application Crash Troubleshooting for {platform}:\n{common_steps}\n{platform_specific_steps.get(platform.lower(), '')}\n6. Check for conflicting applications."
    else:
        return f"General Troubleshooting for {platform}:\n{common_steps}\n{platform_specific_steps.get(platform.lower, '')}"

# Hooks
async def on_billing_handoff(context: RunContextWrapper[CustomerSupportContext]) -> None:
    """Callback function when handing off to billing agent"""
    print("Handing off to billing specialist...")
    if not context.context.account_number:
        print("Warning: No account number in context before billing handoff")

async def on_technical_handoff(context: RunContextWrapper[CustomerSupportContext]) -> None:
    """Callback function when handing off to technical support agent"""
    print("Handing off to technical support specialist...")
    if not context.context.technical_details:
        print("Warning: No technical details in context before technical handoff")

# Specialized Agents
billing_agent = Agent[CustomerSupportContext](
    name="Billing Specialist",
    handoff_description="A specialist who can handle billing inquiries, refunds, and subscription issues.",
    instructions="""You are a billing specialist. Help customers with billing-related inquiries.
    Follow these steps:
    1. Confirm the customer's account number.
    2. Identify the specific billing action needed (refund, update payment, check balance).
    3. Use the process_billing_request tool to handle the customer's request.
    4. Provide clear confirmation when the action is complete.
    
    If the customer has questions unrelated to billing, hand them back to the triage agent.
    """,
    tools=[process_billing_request],
    input_guardrails=[check_inappropriate_content],
    output_guardrails=[ensure_response_quality]
)

technical_agent = Agent[CustomerSupportContext](
    name="Technical Support Specialist",
    handoff_description="A specialist who can troubleshoot technical issues and provide solutions.",
    instructions="""You are a technical support specialist. Help customers resolve technical issues.
    Follow these steps:
    1. Ask for specific details about the issue if not already provided.
    2. Identify which platform the customer is using (web, mobile, desktop).
    3. Use the troubleshoot_technical_issue tool to provide relevant troubleshooting steps.
    4. Follow up to ensure the issue is resolved.
    
    If the customer has questions unrelated to technical support, hand them back to the triage agent.
    """,
    tools=[troubleshoot_technical_issue],
    input_guardrails=[check_inappropriate_content],
    output_guardrails=[ensure_response_quality]
)

# Main Triage Agent
triage_agent = Agent[CustomerSupportContext](
    name="Customer Support Agent",
    handoff_description="The main triage agent who handles initial customer inquiries and routes them appropriately.",
    instructions="""You are a customer support triage agent. Your job is to:
    1. Greet the customer and ask for their customer ID if not provided.
    2. Use the lookup_customer_info tool to retrieve customer information.
    3. Use the categorize_issue tool to determine the nature of the customer's inquiry.
    4. For billing issues, hand off to the billing specialist.
    5. For technical issues, hand off to the technical support specialist.
    6. For general inquiries, handle them yourself.
    
    Always be polite, professional, and helpful.
    """,
    tools=[categorize_issue, lookup_customer_info],
    handoffs=[
        handoff(billing_agent, on_handoff=on_billing_handoff),
        handoff(technical_agent, on_handoff=on_technical_handoff)
    ],
    input_guardrails=[check_inappropriate_content],
    output_guardrails=[ensure_response_quality]
)

# Add handoff back to triage agent
billing_agent.handoffs.append(triage_agent)
technical_agent.handoffs.append(triage_agent)

async def process_conversation(
    message: str, 
    context: CustomerSupportContext, 
    history: List[TResponseInputItem] = None
) -> tuple[str, Agent[CustomerSupportContext]]:
    """
    Process a single conversation turn.
    
    Args:
        message: The user's message
        context: The conversation context
        history: Previous conversation history
        
    Returns:
        A tuple containing (agent response, last active agent)
    """
    if history is None:
        history = []
    
    # Add new message to history
    history.append({"content": message, "role": "user"})
    
    # Determine which agent to use based on context
    current_agent = triage_agent
    if context.issue_type == IssueCategory.BILLING:
        current_agent = billing_agent
    elif context.issue_type == IssueCategory.TECHNICAL:
        current_agent = technical_agent
    
    # Run the agent
    result = await Runner.run(current_agent, history, context=context)
    
    # Extract the response text from the agent
    response_text = ""
    for item in result.new_items:
        if isinstance(item, MessageOutputItem):
            response_text = ItemHelpers.text_message_output(item)
        elif isinstance(item, HandoffOutputItem):
            response_text += f"\n[Transferred from {item.source_agent.name} to {item.target_agent.name}]"
    
    # Return the response and the last active agent
    return response_text, result.last_agent

async def example_usage():
    """
    Example usage of the customer support system
    """
    context = CustomerSupportContext()
    history: List[TResponseInputItem] = []
    
    # Scenario 1: General

from typing import Dict, List, Optional, Any
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import re

# Import necessary libraries for Agent framework
# In a real application, you'd import from the actual openai-agents library
# This is a simplified version for demonstration purposes

class Agent:
    def __init__(self, name, system_message):
        self.name = name
        self.system_message = system_message
        self.client = OpenAI()
        
    def run(self, user_message, conversation_history=None):
        if conversation_history is None:
            conversation_history = []
        
        messages = [
            {"role": "system", "content": self.system_message},
            *conversation_history,
            {"role": "user", "content": user_message}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
        )
        
        return response.choices[0].message.content

# Input Guardrail to check for inappropriate content
def input_content_filter(message: str) -> Optional[str]:
    # Check for inappropriate content or profanity
    profanity_patterns = [
        r'\b(profanity1|profanity2|inappropriate)\b',
        # Add more patterns as needed
    ]
    
    for pattern in profanity_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            return "We cannot process requests containing inappropriate content."
    
    # Check for non-support related queries
    if not any(keyword in message.lower() for keyword in ["help", "issue", "problem", "question", "billing", "technical", "account", "service"]):
        return "Please provide a valid customer support query."
    
    return None  # No issues found

# Output Guardrail to ensure quality responses
def output_quality_check(response: str) -> Optional[str]:
    # Check response length
    if len(response) < 20:
        return "The generated response is too short to be helpful."
    
    # Check if response contains common filler phrases that indicate low quality
    filler_phrases = [
        "I don't know", 
        "I can't help with that",
        "I'm just an AI"
    ]
    
    for phrase in filler_phrases:
        if phrase in response:
            return f"The response contains unhelpful phrases like '{phrase}'."
    
    return None  # No issues found

# Create the specialized agents
billing_agent = Agent(
    name="Billing Specialist",
    system_message="""You are a billing specialist for a software company.
    You can help with billing questions, subscription issues, refunds, and payment methods.
    Provide detailed, accurate information about billing policies.
    For refunds, always ask for order number and purchase date.
    """
)

tech_support_agent = Agent(
    name="Technical Support",
    system_message="""You are a technical support specialist for a software company.
    You can help with installation issues, bugs, error messages, and how-to questions.
    Ask for specific error messages and steps to reproduce problems.
    Always suggest checking for software updates and restarting as initial troubleshooting steps.
    """
)

# Main triage agent with handoffs
triage_agent = Agent(
    name="Customer Support Triage",
    system_message="""You are the initial customer support agent who evaluates customer queries.
    Your job is to:
    1. Handle general questions about products and services directly
    2. Recognize billing questions and hand them off to the Billing Specialist
    3. Recognize technical issues and hand them off to Technical Support
    
    When responding directly, provide helpful and concise information.
    When handing off, inform the customer which specialist will be helping them.
    """
)

# Handoff function implementation
def handoff(query: str, conversation_history: List[Dict]) -> Dict[str, Any]:
    # Determine which specialized agent to use based on the query
    if any(word in query.lower() for word in ["bill", "payment", "charge", "refund", "subscription", "price"]):
        response = billing_agent.run(query, conversation_history)
        return {
            "agent": "Billing Specialist",
            "response": response
        }
    elif any(word in query.lower() for word in ["error", "bug", "crash", "install", "update", "technical", "feature"]):
        response = tech_support_agent.run(query, conversation_history)
        return {
            "agent": "Technical Support",
            "response": response
        }
    else:
        # Handle with the general triage agent
        response = triage_agent.run(query, conversation_history)
        return {
            "agent": "Customer Support Triage",
            "response": response
        }

# Main function that processes customer queries
def process_customer_query(query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
    if conversation_history is None:
        conversation_history = []
    
    # Step 1: Apply input guardrail
    input_guardrail_result = input_content_filter(query)
    if input_guardrail_result:
        return {
            "agent": "System",
            "response": input_guardrail_result,
            "blocked_by_guardrail": True
        }
    
    # Step 2: Process with appropriate agent (with handoffs)
    result = handoff(query, conversation_history)
    
    # Step 3: Apply output guardrail
    output_guardrail_result = output_quality_check(result["response"])
    if output_guardrail_result:
        return {
            "agent": result["agent"],
            "response": "I apologize, but I couldn't generate a helpful response. Let me try again or connect you with a human agent.",
            "blocked_by_guardrail": True,
            "guardrail_message": output_guardrail_result
        }
    
    return result

# Example usage
def example_usage():
    print("\n=== Customer Support System with Guardrails and Handoffs ===\n")
    
    # Example 1: General question handled by triage agent
    query1 = "What are your business hours?"
    print(f"Customer Query: {query1}")
    result1 = process_customer_query(query1)
    print(f"Responded by: {result1['agent']}")
    print(f"Response: {result1['response']}\n")
    
    # Example 2: Billing question handed off to billing specialist
    query2 = "I was charged twice for my subscription last month and need a refund."
    print(f"Customer Query: {query2}")
    result2 = process_customer_query(query2)
    print(f"Responded by: {result2['agent']}")
    print(f"Response: {result2['response']}\n")
    
    # Example 3: Technical question handed off to tech support
    query3 = "I'm getting an error message when trying to install the latest update."
    print(f"Customer Query: {query3}")
    result3 = process_customer_query(query3)
    print(f"Responded by: {result3['agent']}")
    print(f"Response: {result3['response']}\n")
    
    # Example 4: Input guardrail blocking inappropriate content
    query4 = "This is a message containing the word inappropriate to trigger the filter."
    print(f"Customer Query: {query4}")
    result4 = process_customer_query(query4)
    print(f"Responded by: {result4['agent']}")
    print(f"Response: {result4['response']}")
    if result4.get("blocked_by_guardrail"):
        print("(Blocked by input guardrail)\n")

if __name__ == "__main__":
    example_usage()

from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI
import os
from datetime import datetime

# Type definitions
class CustomerQuery(BaseModel):
    """Customer support query input structure"""
    query_text: str = Field(..., description="The customer's support request")
    customer_id: str = Field(..., description="Customer identifier")
    priority: Optional[int] = Field(1, description="Priority level (1-5)")
    category: Optional[str] = Field(None, description="Initial category if known")
    
class SupportResponse(BaseModel):
    """Response structure for all support agents"""
    response_text: str = Field(..., description="The response to customer")
    resolved: bool = Field(..., description="Whether the issue was resolved")
    followup_required: bool = Field(False, description="Whether followup is needed")
    category: str = Field(..., description="The category of the support request")
    confidence: float = Field(..., description="Confidence level in the response (0.0-1.0)")
    
class BillingQuery(BaseModel):
    """Specialized input for billing agent"""
    query_text: str = Field(..., description="The customer's billing question")
    customer_id: str = Field(..., description="Customer identifier")
    subscription_info: Optional[Dict[str, Any]] = Field(None, description="Subscription details if available")
    
class TechnicalQuery(BaseModel):
    """Specialized input for technical support agent"""
    query_text: str = Field(..., description="The customer's technical question")
    customer_id: str = Field(..., description="Customer identifier")
    product_version: Optional[str] = Field(None, description="Product version information")
    platform: Optional[str] = Field(None, description="Customer's platform")
    
class HandoffMetadata(BaseModel):
    """Metadata for agent handoffs"""
    original_query: str = Field(..., description="The original customer query")
    triage_notes: Optional[str] = Field(None, description="Notes from the triage agent")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    priority: int = Field(1, description="Priority level (1-5)")

# Input guardrails
def inappropriate_content_check(query: CustomerQuery) -> Optional[str]:
    """Input guardrail to check for inappropriate content"""
    # Simplified check for demo purposes
    inappropriate_terms = ["profanity", "hate speech", "threat", "illegal"]
    for term in inappropriate_terms:
        if term in query.query_text.lower():
            return f"Query contains inappropriate content: {term}"
    return None

def customer_id_validator(query: CustomerQuery) -> Optional[str]:
    """Input guardrail to validate customer ID format"""
    if not query.customer_id.isalnum() or len(query.customer_id) < 5:
        return "Invalid customer ID format"
    return None

# Output guardrails
def response_quality_check(response: SupportResponse) -> Optional[str]:
    """Output guardrail to ensure response quality"""
    # Simplified checks for demo purposes
    if len(response.response_text) < 20:
        return "Response is too short to be helpful"
    
    if response.confidence < 0.7 and response.resolved:
        return "Low confidence responses should not be marked as resolved"
        
    return None

def response_tone_check(response: SupportResponse) -> Optional[str]:
    """Output guardrail to check for appropriate tone"""
    negative_tones = ["sorry for the inconvenience", "apologies", "regret"]
    positive_phrases = ["thank you", "appreciate", "happy to help"]
    
    # Simple check: If there are negative tones, ensure there are also positive phrases
    has_negative = any(tone in response.response_text.lower() for tone in negative_tones)
    has_positive = any(phrase in response.response_text.lower() for phrase in positive_phrases)
    
    if has_negative and not has_positive:
        return "Response has a negative tone without positive reinforcement"
    
    return None

# Agent implementations
class TriageAgent:
    """Main triage agent that handles initial requests and routes to specialized agents"""
    
    def __init__(self, billing_agent, technical_agent):
        self.client = OpenAI()
        self.billing_agent = billing_agent
        self.technical_agent = technical_agent
        
    def process(self, query: CustomerQuery) -> SupportResponse:
        """Process the initial customer query and determine routing"""
        # For demo purposes, simple classification logic
        query_text = query.query_text.lower()
        
        # Classify the query
        if any(term in query_text for term in ["bill", "charge", "payment", "refund", "subscription"]):
            category = "billing"
            # Hand off to billing agent
            return self.handoff_to_billing(query)
        elif any(term in query_text for term in ["error", "bug", "crash", "feature", "doesn't work"]):
            category = "technical"
            # Hand off to technical agent
            return self.handoff_to_technical(query)
        else:
            # Handle general queries directly
            return self.handle_general_query(query)
    
    def handoff_to_billing(self, query: CustomerQuery) -> SupportResponse:
        """Handoff to billing specialist agent"""
        billing_query = BillingQuery(
            query_text=query.query_text,
            customer_id=query.customer_id,
            subscription_info=self.get_subscription_info(query.customer_id)
        )
        
        metadata = HandoffMetadata(
            original_query=query.query_text,
            triage_notes="Customer has a billing-related inquiry",
            priority=query.priority or 2
        )
        
        return self.billing_agent.process(billing_query, metadata)
    
    def handoff_to_technical(self, query: CustomerQuery) -> SupportResponse:
        """Handoff to technical specialist agent"""
        technical_query = TechnicalQuery(
            query_text=query.query_text,
            customer_id=query.customer_id,
            product_version=self.get_product_version(query.customer_id),
            platform=self.get_customer_platform(query.customer_id)
        )
        
        metadata = HandoffMetadata(
            original_query=query.query_text,
            triage_notes="Customer has a technical support inquiry",
            priority=query.priority or 3
        )
        
        return self.technical_agent.process(technical_query, metadata)
    
    def handle_general_query(self, query: CustomerQuery) -> SupportResponse:
        """Handle general queries without specialized knowledge requirements"""
        # Simple prompt for demo purposes
        prompt = f"""
        You are a helpful customer support agent. The customer has the following query:
        
        {query.query_text}
        
        Provide a helpful response to their general inquiry.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query.query_text}
            ]
        )
        
        response_text = response.choices[0].message.content
        
        return SupportResponse(
            response_text=response_text,
            resolved=True,  # Assume general queries can be resolved
            category="general",
            confidence=0.9
        )
    
    def get_subscription_info(self, customer_id: str) -> Dict[str, Any]:
        """Mock function to get customer subscription info"""
        # In a real system, this would query a database
        return {
            "plan": "premium",
            "billing_cycle": "monthly",
            "next_billing_date": "2023-12-01",
            "payment_method": "credit_card"
        }
    
    def get_product_version(self, customer_id: str) -> str:
        """Mock function to get customer's product version"""
        return "v2.4.1"
    
    def get_customer_platform(self, customer_id: str) -> str:
        """Mock function to get customer's platform"""
        return "Windows 11"

class BillingAgent:
    """Specialized agent for handling billing inquiries"""
    
    def __init__(self):
        self.client = OpenAI()
    
    def process(self, query: BillingQuery, metadata: HandoffMetadata) -> SupportResponse:
        """Process billing-specific queries"""
        subscription_info = query.subscription_info or {}
        
        # Construct specialized prompt with billing expertise
        prompt = f"""
        You are a specialized billing support agent. The customer has the following query:
        
        {query.query_text}
        
        Customer subscription information:
        - Plan: {subscription_info.get('plan', 'unknown')}
        - Billing cycle: {subscription_info.get('billing_cycle', 'unknown')}
        - Next billing date: {subscription_info.get('next_billing_date', 'unknown')}
        - Payment method: {subscription_info.get('payment_method', 'unknown')}
        
        Provide a helpful, accurate response addressing their billing concern.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query.query_text}
            ]
        )
        
        response_text = response.choices[0].message.content
        
        # Determine if the issue is resolved based on content
        resolved = "refund" not in query.query_text.lower()  # Assume refunds need follow-up
        
        return SupportResponse(
            response_text=response_text,
            resolved=resolved,
            followup_required=not resolved,
            category="billing",
            confidence=0.85
        )

class TechnicalSupportAgent:
    """Specialized agent for handling technical support inquiries"""
    
    def __init__(self):
        self.client = OpenAI()
    
    def process(self, query: TechnicalQuery, metadata: HandoffMetadata) -> SupportResponse:
        """Process technical support queries"""
        # Construct specialized prompt with technical expertise
        prompt = f"""
        You are a specialized technical support agent. The customer has the following query:
        
        {query.query_text}
        
        Technical information:
        - Product version: {query.product_version or 'unknown'}
        - Platform: {query.platform or 'unknown'}
        
        Provide a detailed, accurate technical response to resolve their issue.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query.query_text}
            ]
        )
        
        response_text = response.choices[0].message.content
        
        # Complex issues might require followup
        is_complex = len(query.query_text) > 100
        contains_error_terms = any(term in query.query_text.lower() for term in ["error", "crash", "bug"])
        
        return SupportResponse(
            response_text=response_text,
            resolved=not (is_complex and contains_error_terms),
            followup_required=is_complex and contains_error_terms,
            category="technical",
            confidence=0.8 if is_complex else 0.95
        )

class CustomerSupportSystem:
    """Combined system implementing both guardrails and handoffs"""
    
    def __init__(self):
        # Initialize specialized agents
        self.billing_agent = BillingAgent()
        self.technical_agent = TechnicalSupportAgent()
        
        # Initialize main triage agent with references to specialized agents
        self.triage_agent = TriageAgent(self.billing_agent, self.technical_agent)
        
        # Define input guardrails
        self.input_guardrails = [
            inappropriate_content_check,
            customer_id_validator
        ]
        
        # Define output guardrails
        self.output_guardrails = [
            response_quality_check,
            response_tone_check
        ]
    
    def process_query(self, query: CustomerQuery) -> Dict[str, Any]:
        """Process a customer query with guardrails and potential handoffs"""
        result = {
            "success": True,
            "errors": [],
            "response": None
        }
        
        # Apply input guardrails
        for guardrail in self.input_guardrails:
            error = guardrail(query)
            if error:
                result["success"] = False
                result["errors"].append(error)
        
        # If input guardrails failed, return early
        if not result["success"]:
            return result
        
        # Process through triage agent (which may handoff to specialists)
        response = self.triage_agent.process(query)
        
        # Apply output guardrails
        output_errors = []
        for guardrail in self.output_guardrails:
            error = guardrail(response)
            if error:
                output_errors.append(error)
        
        if output_errors:
            # In a real system, you might regenerate or fix the response
            # For now, we'll just record the errors
            result["errors"] = output_errors
            result["success"] = False
        
        result["response"] = response
        return result

# Example usage
def example_usage():
    """Demonstrate the system with example queries"""
    support_system = CustomerSupportSystem()
    
    # Example 1: General query
    general_query = CustomerQuery(
        query_text="What are your business hours?",
        customer_id="CUST12345"
    )
    print("Example 1: General Query")
    general_result = support_system.process_query(general_query)
    if general_result["success"]:
        print(f"Response: {general_result['response'].response_text}")
    else:
        print(f"Errors: {general_result['errors']}")
    print()
    
    # Example 2: Billing query (will trigger handoff)
    billing_query = CustomerQuery(
        query_text="I was charged twice for my subscription this month. Can I get a refund?",
        customer_id="CUST67890"
    )
    print("Example 2: Billing Query (Handoff)")
    billing_result = support_system.process_query(billing_query)
    if billing_result["success"]:
        print(f"Response: {billing_result['response'].response_text}")
        print(f"Category: {billing_result['response'].category}")
        print(f"Resolved: {billing_result['response'].resolved}")
    else:
        print(f"Errors: {billing_result['errors']}")
    print()
    
    # Example 3: Technical query (will trigger handoff)
    tech_query = CustomerQuery(
        query_text="The app crashes whenever I try to export my data. I'm using version 2.4 on Windows.",
        customer_id="CUST24680"
    )
    print("Example 3: Technical Query (Handoff)")
    tech_result = support_system.process_query(tech_query)
    if tech_result["success"]:
        print(f"Response: {tech_result['response'].response_text}")
        print

