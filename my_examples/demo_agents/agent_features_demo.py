#!/usr/bin/env python3
"""
Agent Features Demo
==================

This example demonstrates a comprehensive implementation of a customer support system
using the OpenAI Agents SDK. It showcases:

1. Basic agent configuration with tools
2. Custom context for user information
3. Structured output types using Pydantic
4. Multiple specialized agents with handoffs
5. Dynamic instructions based on user context
6. Lifecycle hooks for logging and monitoring
7. Input and output guardrails

To run:
    python agent_features_demo.py
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union, cast

from pydantic import BaseModel, Field

# Import the Agents SDK components
from agents import (
    Agent,
    AgentHooks,
    FunctionTool,
    HandoffOptions,
    ModelSettings,
    RunContext,
    Runner,
    function_tool,
)
from agents.guardrails import (
    InputGuardrail,
    OutputGuardrail,
    ValidationResult,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("support_system")

# -----------------------------------------------------------------------------
# SECTION 1: Context and Output Types
# -----------------------------------------------------------------------------

class UserTier(str, Enum):
    """User subscription tier levels."""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"

@dataclass
class UserContext:
    """Custom context with user information for agent dependency injection."""
    user_id: str
    username: str
    tier: UserTier
    join_date: datetime
    is_first_conversation: bool
    previous_tickets: List[Dict[str, Any]]
    
    def get_subscription_features(self) -> Dict[str, bool]:
        """Return features available for the user's subscription tier."""
        features = {
            "priority_support": False,
            "24_7_support": False,
            "technical_specialist": False,
            "dedicated_account_manager": False,
        }
        
        if self.tier == UserTier.BASIC:
            features["priority_support"] = True
        elif self.tier == UserTier.PREMIUM:
            features["priority_support"] = True
            features["24_7_support"] = True
            features["technical_specialist"] = True
        elif self.tier == UserTier.ENTERPRISE:
            features.update(dict.fromkeys(features, True))
            
        return features

class TicketCategory(str, Enum):
    """Categories for support tickets."""
    ACCOUNT = "account"
    BILLING = "billing"
    TECHNICAL = "technical"
    FEATURE_REQUEST = "feature_request"
    OTHER = "other"

class TicketPriority(str, Enum):
    """Priority levels for support tickets."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class SupportTicket(BaseModel):
    """Structured output type for the triage agent."""
    ticket_id: str = Field(description="Unique identifier for the ticket")
    category: TicketCategory = Field(description="The category of the support request")
    summary: str = Field(description="Brief summary of the issue")
    priority: TicketPriority = Field(description="Priority level for this ticket")
    requires_specialist: bool = Field(description="Whether this requires a specialist")
    needs_account_info: bool = Field(description="Whether account info is needed to resolve")
    estimated_resolution_time: str = Field(description="Estimated time to resolve")

class BillingResponse(BaseModel):
    """Structured output for billing-related queries."""
    issue_type: str = Field(description="Type of billing issue")
    recommended_action: str = Field(description="Recommended action to resolve")
    refund_eligible: bool = Field(description="Whether the customer is eligible for a refund")
    next_steps: List[str] = Field(description="List of next steps to resolve the issue")
    reference_links: List[str] = Field(description="Relevant documentation links")

class TechnicalResponse(BaseModel):
    """Structured output for technical support queries."""
    problem_identified: str = Field(description="Description of the identified problem")
    solution_steps: List[str] = Field(description="Step-by-step solution instructions")
    code_snippets: Optional[Dict[str, str]] = Field(None, description="Any relevant code examples")
    requires_escalation: bool = Field(description="Whether this needs escalation to engineering")
    affected_components: List[str] = Field(description="System components affected")

# -----------------------------------------------------------------------------
# SECTION 2: Lifecycle Hooks
# -----------------------------------------------------------------------------

class SupportAgentHooks(AgentHooks):
    """Custom hooks for monitoring the agent lifecycle."""
    
    def __init__(self):
        self.start_time = None
        self.messages_processed = 0
        self.tools_called = 0
        
    async def on_run_start(self, agent, context, run_id, input_message):
        """Called when an agent run starts."""
        self.start_time = datetime.now()
        logger.info(f"Agent run started: {agent.name} - Run ID: {run_id}")
        logger.info(f"User: {context.context.username} (Tier: {context.context.tier})")
        
    async def on_run_end(self, agent, context, run_id, response):
        """Called when an agent run completes."""
        duration = datetime.now() - self.start_time if self.start_time else None
        logger.info(f"Agent run completed: {agent.name} - Run ID: {run_id}")
        logger.info(f"Duration: {duration}")
        logger.info(f"Messages processed: {self.messages_processed}")
        logger.info(f"Tools called: {self.tools_called}")
        
    async def on_message_received(self, agent, context, run_id, message):
        """Called when a message is received from the user."""
        self.messages_processed += 1
        logger.debug(f"Message received: {message.content[:50]}...")
        
    async def on_tool_call(self, agent, context, run_id, tool_call):
        """Called when a tool is invoked."""
        self.tools_called += 1
        logger.info(f"Tool called: {tool_call.name} with args: {tool_call.arguments}")
        
    async def on_handoff(self, source_agent, target_agent, context, run_id, handoff_args):
        """Called when an agent hands off to another agent."""
        logger.info(f"Handoff: {source_agent.name} → {target_agent.name}")
        logger.info(f"Handoff args: {handoff_args}")

# -----------------------------------------------------------------------------
# SECTION 3: Guardrails
# -----------------------------------------------------------------------------

class SupportTopicGuardrail(InputGuardrail):
    """Input guardrail to ensure user queries are support-related."""
    
    async def validate(self, input_text: str, context: RunContext[UserContext]) -> ValidationResult:
        # Define keywords for support-related topics
        support_keywords = [
            "help", "issue", "problem", "error", "can't", "doesn't work",
            "billing", "account", "subscription", "payment", "charge",
            "upgrade", "downgrade", "feature", "access", "password",
            "login", "technical", "bug", "error message"
        ]
        
        # Check if any support keyword is in the input
        if any(keyword in input_text.lower() for keyword in support_keywords):
            return ValidationResult(valid=True)
        
        # Invalid input not related to support
        return ValidationResult(
            valid=False,
            message="I can only help with support-related questions. Please provide details about your account, billing, or technical issue."
        )

class ProfanityGuardrail(InputGuardrail):
    """Input guardrail to filter out profanity."""
    
    async def validate(self, input_text: str, context: RunContext[UserContext]) -> ValidationResult:
        # Simple list of profanity words (would use a more comprehensive solution in production)
        profanity_list = ["damn", "hell", "ass", "crap"]
        
        # Check for profanity
        if any(word in input_text.lower().split() for word in profanity_list):
            return ValidationResult(
                valid=False,
                message="Please refrain from using inappropriate language. How can I assist you with your support needs today?"
            )
        
        return ValidationResult(valid=True)

class ResponseQualityGuardrail(OutputGuardrail):
    """Output guardrail to ensure responses meet quality standards."""
    
    async def validate(self, output_text: str, context: RunContext[UserContext]) -> ValidationResult:
        # Ensure minimum response length
        if len(output_text) < 50:
            return ValidationResult(
                valid=False,
                message="Please provide a more detailed response to the user's query."
            )
        
        # Ensure response has a greeting for first-time users
        if context.context.is_first_conversation and not any(greeting in output_text.lower() 
                                                        for greeting in ["welcome", "hello", "hi"]):
            return ValidationResult(
                valid=False,
                message="Please include a friendly greeting for this first-time user."
            )
            
        return ValidationResult(valid=True)

# -----------------------------------------------------------------------------
# SECTION 4: Tools
# -----------------------------------------------------------------------------

@function_tool
def get_user_subscription_details(context: RunContext[UserContext]) -> Dict[str, Any]:
    """Get detailed information about the user's subscription."""
    user = context.context
    subscription_features = user.get_subscription_features()
    
    return {
        "tier": user.tier.value,
        "join_date": user.join_date.isoformat(),
        "features": subscription_features,
        "renewal_date": (user.join_date.replace(year=user.join_date.year + 1)).isoformat(),
    }

@function_tool
def search_knowledge_base(query: str, category: Optional[str] = None) -> List[Dict[str, str]]:
    """Search the knowledge base for relevant articles."""
    # In a real implementation, this would query a database
    knowledge_base = [
        {
            "id": "KB001",
            "title": "How to reset your password",
            "category": "account",
            "summary": "Step-by-step guide to reset your password",
            "url": "https://example.com/help/reset-password"
        },
        {
            "id": "KB002",
            "title": "Understanding your billing cycle",
            "category": "billing",
            "summary": "Explanation of how billing cycles work and when you're charged",
            "url": "https://example.com/help/billing-cycle"
        },
        {
            "id": "KB003",
            "title": "Troubleshooting API connection issues",
            "category": "technical",
            "summary": "Common API connection problems and their solutions",
            "url": "https://example.com/help/api-troubleshooting"
        },
        {
            "id": "KB004",
            "title": "How to upgrade your subscription",
            "category": "billing",
            "summary": "Process for upgrading to a higher subscription tier",
            "url": "https://example.com/help/upgrade-subscription"
        },
        {
            "id": "KB005",
            "title": "Setting up two-factor authentication",
            "category": "account",
            "summary": "Enable 2FA to secure your account",
            "url": "https://example.com/help/2fa-setup"
        }
    ]
    
    # Filter by category if provided
    if category:
        filtered_articles = [a for a in knowledge_base if a["category"] == category]
    else:
        filtered_articles = knowledge_base
    
    # Simple keyword matching (in a real system, use a proper search algorithm)
    query_terms = query.lower().split()
    results = []
    
    for article in filtered_articles:
        # Check if any query term is in the title or summary
        if any(term in article["title"].lower() or term in article["summary"].lower() 
               for term in query_terms):
            results.append(article)
    
    return results

@function_tool
async def check_account_status(context: RunContext[UserContext]) -> Dict[str, Any]:
    """Check the status of the user's account."""
    user = context.context
    
    # Simulate API call with delay
    await asyncio.sleep(0.5)
    
    return {
        "user_id": user.user_id,
        "username": user.username,
        "account_active": True,
        "email_verified": True,
        "last_login": datetime.now().isoformat(),
        "account_age_days": (datetime.now() - user.join_date).days,
        "previous_tickets_count": len(user.previous_tickets)
    }

@function_tool
async def create_support_ticket(
    title: str, 
    description: str, 
    category: TicketCategory,
    priority: TicketPriority,
    context: RunContext[UserContext]
) -> Dict[str, Any]:
    """Create a new support ticket in the system."""
    user = context.context
    
    # Generate a ticket ID (would be done by the ticketing system)
    ticket_id = f"TICK-{user.user_id[:4]}-{datetime.now().strftime('%Y%m%d%H%M')}"
    
    # Simulate API call with delay
    await asyncio.sleep(0.8)
    
    ticket = {
        "ticket_id": ticket_id,
        "user_id": user.user_id,
        "title": title,
        "description": description,
        "category": category,
        "priority": priority,
        "status": "open",
        "created_at": datetime.now().isoformat(),
    }
    
    logger.info(f"Created ticket: {ticket_id} for user: {user.username}")
    return ticket

@function_tool
def get_billing_history(context: RunContext[UserContext]) -> List[Dict[str, Any]]:
    """Retrieve the billing history for the user."""
    # In a real implementation, this would query a database
    
    # Sample billing history
    billing_history = [
        {
            "transaction_id": "T12345",
            "date": (datetime.now().replace(month=datetime.now().month - 1)).isoformat(),
            "amount": 9.99 if context.context.tier == UserTier.BASIC else
                     29.99 if context.context.tier == UserTier.PREMIUM else
                     99.99 if context.context.tier == UserTier.ENTERPRISE else 0.00,
            "description": f"{context.context.tier.value.capitalize()} Plan - Monthly Subscription",
            "status": "completed"
        },
        {
            "transaction_id": "T12346",
            "date": (datetime.now().replace(month=datetime.now().month - 2)).isoformat(),
            "amount": 9.99 if context.context.tier == UserTier.BASIC else
                     29.99 if context.context.tier == UserTier.PREMIUM else
                     99.99 if context.context.tier == UserTier.ENTERPRISE else 0.00,
            "description": f"{context.context.tier.value.capitalize()} Plan - Monthly Subscription",
            "status": "completed"
        },
        {
            "transaction_id": "T12347",
            "date": datetime.now().isoformat(),
            "amount": 9.99 if context.context.tier == UserTier.BASIC else
                     29.99 if context.context.tier == UserTier.PREMIUM else
                     99.99 if context.context.tier == UserTier.ENTERPRISE else 0.00,
            "description": f"{context.context.tier.value.capitalize()} Plan - Monthly Subscription",
            "status": "pending"
        }
    ]
    
    return billing_history

@function_tool
async def process_refund(
    transaction_id: str,
    reason: str,
    context: RunContext[UserContext]
) -> Dict[str, Any]:
    """Process a refund for a specific transaction."""
    # Simulate API call with delay
    await asyncio.sleep(1.0)
    
    # In a real implementation, this would interact with payment processor
    if transaction_id.startswith("T"):
        return {
            "refund_id": f"REF-{transaction_id[1:]}",
            "transaction_id": transaction_id,
            "amount_refunded": 9.99 if context.context.tier == UserTier.BASIC else
                              29.99 if context.context.tier == UserTier.PREMIUM else
                              99.99 if context.context.tier == UserTier.ENTERPRISE else 0.00,
            "status": "processed",
            "reason": reason,
            "processed_at": datetime.now().isoformat(),
            "estimated_arrival": (datetime.now().replace(day=datetime.now().day + 3)).isoformat()
        }
    else:
        return {
            "error": "Invalid transaction ID",
            "status": "failed"
        }

# -----------------------------------------------------------------------------
# SECTION 5: Agent Implementations with Dynamic Instructions
# -----------------------------------------------------------------------------

# Dynamic instructions based on user context
def triage_instructions(context: RunContext[UserContext], agent: Agent[UserContext]) -> str:
    """Dynamic instructions for the triage agent based on user context."""
    user = context.context
    tier_specific = ""
    
    if user.tier == UserTier.FREE:
        tier_specific = "Note that free tier users have limited support options and should be guided to self-service resources when possible."
    elif user.tier == UserTier.PREMIUM or user.tier == UserTier.ENTERPRISE:
        tier_specific = f"This is a {user.tier.value} tier customer and should receive priority support. Offer advanced options and faster resolution paths."
    
    first_time = ""
    if user.is_first_conversation:
        first_time = "This is the user's first support conversation. Be extra welcoming and thorough in your responses."
    
    return f"""You are a support triage agent for a SaaS platform.
Your job is to understand the user's issue and either resolve it directly or route to the appropriate specialized agent.

{tier_specific}
{first_time}

User Information:
- Username: {user.username}
- Subscription tier: {user.tier.value}
- Join date: {user.join_date.strftime('%Y-%m-%d')}
- Previous tickets: {len(user.previous_tickets)}

Follow these steps:
1. Greet the user appropriately
2. Understand their issue completely
3. Categorize the issue (billing, technical, account, feature request)
4. Attempt to resolve simple issues directly using your tools
5. For complex issues, create a support ticket and hand off to the appropriate specialized agent

Your top priority is user satisfaction and efficient resolution of issues.
"""

def billing_agent_instructions(context: RunContext[UserContext], agent: Agent[UserContext]) -> str:
    """Dynamic instructions for the billing agent based on user context."""
    user = context.context
    
    refund_policy = "Standard 30-day refund policy applies."
    if user.tier == UserTier.PREMIUM:
        refund_policy = "Premium users have a 60-day refund window."
    elif user.tier == UserTier.ENTERPRISE:
        refund_policy = "Enterprise customers have flexible refund options and can request exceptions to standard policies."
    
    return f"""You are a billing specialist for a SaaS platform.
Your job is to help users with billing-related issues, subscriptions, and refunds.

User Information:
- Username: {user.username}
- Subscription tier: {user.tier.value}
- Join date: {user.join_date.strftime('%Y-%m-%d')}

Refund Policy:
{refund_policy}

Subscription Information:
- Current plan: {user.tier.value.capitalize()}
- Price: {9.99 if user.tier == UserTier.BASIC else 29.99 if user.tier == UserTier.PREMIUM else 99.99 if user.tier == UserTier.ENTERPRISE else 0.00} per month
- Billing cycle: Monthly on the {user.join_date.day}th

Follow these steps:
1. Understand the billing issue completely
2. Check the customer's billing history
3. Explain relevant policies clearly
4. Process refunds or make adjustments as appropriate
5. Provide clear next steps and expectations

Always explain billing terms in simple language and be transparent about all charges.
"""

def technical_agent_instructions(context: RunContext[UserContext], agent: Agent[UserContext]) -> str:
    """Dynamic instructions for the technical agent based on user context."""
    user = context.context
    
    tech_support_level = "basic troubleshooting"
    if user.tier == UserTier.PREMIUM:
        tech_support_level = "advanced troubleshooting including code review and integration support"
    elif user.tier == UserTier.ENTERPRISE:
        tech_support_level = "comprehensive technical support including dedicated engineer time and custom solutions"
    
    return f"""You are a technical support specialist for a SaaS platform.
Your job is to help users resolve technical issues, bugs, and implementation problems.

User Information:
- Username: {user.username}
- Subscription tier: {user.tier.value}
- Support level: {tech_support_level}

You can provide:
- API implementation guidance
- Error troubleshooting
- Integration support
- Feature usage explanation
- Best practices advice

Follow these steps:
1. Understand the technical issue thoroughly
2. Ask for relevant error messages or logs if needed
3. Provide clear, step-by-step solutions
4. Include code examples when helpful
5. Explain why the issue occurred to prevent future problems

Always validate that your solution worked for the user before closing the conversation.
"""

# Create the specialized agents
billing_agent = Agent[UserContext](
    name="Billing Specialist",
    instructions=billing_agent_instructions,
    model="o3-mini",
    output_type=BillingResponse,
    tools=[
        get_user_subscription_details,
        get_billing_history,
        process_refund,
        search_knowledge_base,
    ],
    hooks=SupportAgentHooks(),
    guardrails=[ResponseQualityGuardrail()]
)

technical_agent = Agent[UserContext](
    name="Technical Support",
    instructions=technical_agent_instructions,
    model="o3-mini",
    output_type=TechnicalResponse,
    tools=[
        search_knowledge_base,
        check_account_status,
    ],
    hooks=SupportAgentHooks(),
    guardrails=[ResponseQualityGuardrail()]
)

# Main triage agent with handoffs to specialized agents
triage_agent = Agent[UserContext](
    name="Support Triage",
    instructions=triage_instructions,
    model="o3-mini",
    output_type=SupportTicket,
    tools=[
        get_user_subscription_details,
        search_knowledge_base,
        check_account_status,
        create_support_ticket,
    ],
    handoffs=[
        HandoffOptions(
            agent=billing_agent,
            name="Billing Specialist",
            description="Handles billing issues, refunds, subscription questions",
        ),
        HandoffOptions(
            agent=technical_agent,
            name="Technical Support",
            description="Resolves technical issues, bugs, API problems, integration questions",
        ),
    ],
    hooks=SupportAgentHooks(),
    guardrails=[
        SupportTopicGuardrail(),
        ProfanityGuardrail(),
        ResponseQualityGuardrail(),
    ]
)

# Clone the agents to create variants with different configurations
basic_triage_agent = triage_agent.clone(
    name="Basic Support Triage",
    model_settings=ModelSettings(temperature=0.2),  # More deterministic
)

premium_triage_agent = triage_agent.clone(
    name="Premium Support Triage",
    model_settings=ModelSettings(temperature=0.7),  # More creative
)

# -----------------------------------------------------------------------------
# SECTION 6: Main Execution Logic
# -----------------------------------------------------------------------------

async def simulate_conversation(agent: Agent[UserContext], user_context: UserContext, messages: List[str]) -> None:
    """Simulate a conversation with an agent."""
    runner = Runner()
    
    print("\n" + "="*80)
    print(f"Starting conversation with {agent.name}")
    print(f"User: {user_context.username} ({user_context.tier.value} tier)")
    print("="*80)
    
    # Create a session for this conversation
    session = await runner.create_session(agent, user_context)
    
    # Process each message in the conversation
    for message in messages:
        print(f"\nUser: {message}")
        response = await runner.run(session, message)
        
        # Handle the response based on its type
        if isinstance(response, str):
            print(f"\n{agent.name}: {response}")
        elif hasattr(response, 'model_dump'):
            # For pydantic models
            print(f"\n{agent.name} response:")
            formatted_response = json.dumps(response.model_dump(), indent=2)
            print(formatted_response)
        else:
            # For other types
            print(f"\n{agent.name} response:")
            print(response)
            
    print("\n" + "="*80 + "\n")

async def main():
    """Main entry point for the demo."""
    
    # Create user contexts for different scenarios
    basic_user = UserContext(
        user_id="usr_12345",
        username="johndoe",
        tier=UserTier.BASIC,
        join_date=datetime.now().replace(year=datetime.now().year - 1),
        is_first_conversation=False,
        previous_tickets=[
            {"id": "TICK-1234", "title": "Password reset issue"},
            {"id": "TICK-2345", "title": "Billing question"}
        ]
    )
    
    premium_user = UserContext(
        user_id="usr_67890",
        username="janesmith",
        tier=UserTier.PREMIUM,
        join_date=datetime.now().replace(month=datetime.now().month - 6),
        is_first_conversation=False,
        previous_tickets=[
            {"id": "TICK-5678", "title": "API connection problem"}
        ]
    )
    
    new_user = UserContext(
        user_id="usr_99999",
        username="newuser",
        tier=UserTier.FREE,
        join_date=datetime.now().replace(day=datetime.now().day - 2),
        is_first_conversation=True,
        previous_tickets=[]
    )
    
    # Scenario 1: Basic user with billing issue (demonstrates handoff)
    basic_user_billing_conversation = [
        "Hi, I was charged twice this month for my subscription. Can you help?",
        "I see the charges on my credit card statement from yesterday and last week.",
        "Yes, I'd like a refund for the duplicate charge."
    ]
    
    # Scenario 2: Premium user with technical issue
    premium_user_technical_conversation = [
        "I'm having trouble with the API integration. I keep getting 403 errors.",
        "I'm using the authentication token as described in the docs but it's not working.",
        "I'll try that solution and get back to you if it doesn't work. Thanks!"
    ]
    
    # Scenario 3: New user onboarding (demonstrates dynamic instructions for first-time users)
    new_user_conversation = [
        "Hello, I just signed up yesterday. How do I get started?",
        "I'm interested in the API documentation and setting up my first integration.",
        "That was very helpful, thank you!"
    ]
    
    # Run the conversations
    print("\n🤖 AGENT FEATURES DEMO 🤖")
    print("This demo showcases the complete range of agent features.\n")
    
    # Run Scenario 1
    print("\n📋 SCENARIO 1: Billing Issue with Handoff")
    await simulate_conversation(triage_agent, basic_user, basic_user_billing_conversation)
