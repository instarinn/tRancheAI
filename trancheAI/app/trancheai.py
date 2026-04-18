import os
import json
import re
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
import psycopg
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# =========================================================
# CONFIG
# =========================================================

OPENAI_MODEL = "gpt-4o-mini"
SUPABASE_URL = os.getenv("SUPABASE_URL")

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set")

if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_DB_URL is not set")

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

# =========================================================
# SCHEMA CONTEXT
# Replace this with your real TranchIQ schema
# =========================================================

SCHEMA_CONTEXT = """
You are working with a Supabase/Postgres database for TranchIQ.

Available tables:

1. lead_ai_analytics_360
-assigned_salesperson_id (uuid)
-attributed_builder_id (text)
-attributed_salesperson_id (text)
-avg_brochure_engagement_score (numeric)
-avg_cost_sheet_engagement_score (numeric)
-avg_interactions_per_day (numeric)
-avg_quote_engagement_score (numeric)
-brochure_first_viewer_browser (text)
-brochure_first_viewer_device (text)
-brochure_first_viewer_os (text)
-browser (text)
-channel_partner_id (uuid)
-city (text)
-cost_sheet_first_viewer_browser (text)
-cost_sheet_first_viewer_city (text)
-cost_sheet_first_viewer_device (text)
-cost_sheet_first_viewer_os (text)
-customer_source (text)
-days_since_last_activity (numeric)
-days_until_next_follow_up (numeric)
-device_type (text)
-eligibility_score (integer)
-email (text)
-fastest_brochure_view_seconds (integer)
-fastest_cost_sheet_view_seconds (integer)
-fastest_quote_view_seconds (integer)
-full_name (text)
-has_completed_site_visit (boolean)
-has_document_forwards (boolean)
-has_high_document_engagement (boolean)
-has_multiple_visit_failures (boolean)
-has_property_finalized (boolean)
-has_recent_activity_3_days (boolean)
-is_dormant_14_days (boolean)
-is_dormant_7_days (boolean)
-is_follow_up_overdue (boolean)
-is_high_views_but_inactive (boolean)
-landing_page (text)
-language (text)
-last_activity_at (timestamp with time zone)
-latest_brochure_first_viewed_at (timestamp with time zone)
-latest_brochure_forwarded_at (timestamp with time zone)
-latest_brochure_last_viewed_at (timestamp with time zone)
-latest_brochure_shared_at (timestamp with time zone)
-latest_cost_sheet_first_viewed_at (timestamp with time zone)
-latest_cost_sheet_forwarded_at (timestamp with time zone)
-latest_cost_sheet_last_viewed_at (timestamp with time zone)
-latest_cost_sheet_shared_at (timestamp with time zone)
-latest_follow_up_at (timestamp with time zone)
-latest_legacy_site_visit_date (date)
-latest_quote_first_viewed_at (timestamp with time zone)
-latest_quote_forwarded_at (timestamp with time zone)
-latest_quote_last_viewed_at (timestamp with time zone)
-latest_quote_shared_at (timestamp with time zone)
-latest_site_visit_completed_at (timestamp with time zone)
-latest_site_visit_date (date)
-latest_site_visit_interest_level (text)
-latest_site_visit_scheduled_datetime (timestamp with time zone)
-lead_age_days (numeric)
-lead_created_at (timestamp with time zone)
-lead_created_by_user_id (uuid)
-lead_generation_method (text)
-lead_id (uuid)
-lead_notes (text)
-lead_remarks (text)
-lead_temperature (character varying)
-lead_type (text)
-lead_updated_at (timestamp without time zone)
-loan_amount (numeric)
-mobile (numeric)
-next_scheduled_follow_up (timestamp without time zone)
-os (text)
-priority (character varying)
-project_id (uuid)
-property_finalized (boolean)
-property_type (text)
-property_value (numeric)
-purchase_timeline (character varying)
-purpose (text)
-quote_first_viewer_browser (text)
-quote_first_viewer_city (text)
-quote_first_viewer_device (text)
-quote_first_viewer_os (text)
-record_created_at (timestamp with time zone)
-referrer (text)
-sales_stage (text)
-selected_bank (character varying)
-site_visit_follow_up_required (boolean)
-site_visit_next_follow_up_date (date)
-site_visits_cancelled_count (bigint)
-site_visits_completed_count (bigint)
-site_visits_confirmed_count (bigint)
-site_visits_no_show_count (bigint)
-site_visits_scheduled_count (bigint)
-source (text)
-status (text)
-timezone (text)
-total_brochure_forwards (bigint)
-total_brochure_organic_forwards (bigint)
-total_brochure_unique_viewers (bigint)
-total_brochure_views (bigint)
-total_brochures_shared (bigint)
-total_call_follow_ups (bigint)
-total_cost_sheet_forwards (bigint)
-total_cost_sheet_organic_forwards (bigint)
-total_cost_sheet_unique_viewers (bigint)
-total_cost_sheet_views (bigint)
-total_cost_sheets_shared (bigint)
-total_document_forwards (bigint)
-total_document_views (bigint)
-total_documents_shared (bigint)
-total_follow_ups (bigint)
-total_interactions_count (bigint)
-total_legacy_cost_sheet_events (bigint)
-total_legacy_quote_events (bigint)
-total_legacy_site_visits (bigint)
-total_quote_forwards (bigint)
-total_quote_organic_forwards (bigint)
-total_quote_unique_viewers (bigint)
-total_quote_views (bigint)
-total_quotes_draft (bigint)
-total_quotes_shared (bigint)
-total_site_visits_scheduled (bigint)
-total_whatsapp_follow_ups (bigint)
-unique_lead_id (text)
-unit_number (character varying)
-unit_price (numeric)
-unit_type (character varying)
-utm_campaign (text)
-utm_medium (text)
-utm_source (text)
-view_generated_at (timestamp with time zone)

2. customer_360_view
-age (text)
-application_type (text)
-applied_loan_amount (numeric)
-approved_loan_amount (numeric)
-assigned_user_mobile (text)
-assigned_user_name (text)
-attributed_builder_id (text)
-attributed_salesperson_id (text)
-builder_email (character varying)
-builder_id (uuid)
-builder_name (character varying)
-builder_phone (character varying)
-cibil_score (text)
-city (text)
-creator_id (uuid)
-creator_mobile (character varying)
-creator_name (character varying)
-creator_type (character varying)
-creator_user_id (uuid)
-current_status (text)
-customer_created_at (timestamp with time zone)
-customer_email (text)
-customer_id (uuid)
-customer_mobile (text)
-customer_name (text)
-customer_updated_at (timestamp without time zone)
-desired_loan_amount (numeric)
-disbursed_applications (bigint)
-employment_type (text)
-first_contact_date (timestamp with time zone)
-latest_disbursal_date (date)
-latest_payment_date (date)
-lead_status (text)
-lender_name (text)
-loan_type (text)
-monthly_income (double precision)
-open_tickets (bigint)
-paid_payment_count (bigint)
-partner_email (text)
-partner_id (uuid)
-partner_mobile (text)
-partner_name (text)
-pending_payment_count (bigint)
-project_developer (character varying)
-project_id (uuid)
-project_location (text)
-project_name (character varying)
-property_finalized (boolean)
-property_value (numeric)
-purchase_timeline (character varying)
-salesperson_email (character varying)
-salesperson_id (uuid)
-salesperson_mobile (character varying)
-salesperson_name (character varying)
-salesperson_user_type (character varying)
-search_text (text)
-status_created_at (timestamp with time zone)
-status_customer_name (text)
-status_full_name (text)
-status_id (uuid)
-status_lead_id (uuid)
-status_mobile (text)
-status_product (text)
-status_unique_lead_id (text)
-status_updated_at (timestamp with time zone)
-total_disbursed_amount (numeric)
-total_documents (bigint)
-total_loan_applications (bigint)
-total_paid_amount (numeric)
-total_payment_records (bigint)
-total_pending_amount (numeric)
-total_support_tickets (bigint)
-total_tranches (bigint)
-unique_lead_id (text)
-unit_number (character varying)
-unit_price (numeric)
-unit_type (character varying)
-verified_documents (bigint)

3. loan_application_status_212d7de7
-application_type (text)
-applied_loan_amount (numeric)
-approved_loan_amount (numeric)
-assigned_user_mobile (text)
-assigned_user_name (text)
-channel_partner_id (uuid)
-created_at (timestamp with time zone)
-current_status (text)
-customer_name (text)
-full_name (text)
-id (uuid)
-lead_id (uuid)
-lender_name (text)
-mobile (text)
-product (text)
-project_id (uuid)
-unique_lead_id (text)
-updated_at (timestamp with time zone)

4. leads
-age (text)
-assigned_to (uuid)
-attributed_builder_id (text)
-attributed_salesperson_id (text)
-auto_captured (boolean)
-browser (text)
-bulk_upload_id (uuid)
-channel_partner_id (uuid)
-cibil_score (text)
-city (text)
-cost_sheet_shared_count (integer)
-created_at (timestamp with time zone)
-created_by (uuid)
-current_emi (text)
-customer_source (text)
-device_type (text)
-digital_source (text)
-eligibility_score (integer)
-email (text)
-employment_type (text)
-follow_up_count (integer)
-follow_up_date (timestamp without time zone)
-full_name (text)
-id (uuid)
-integration_id (uuid)
-landing_page (text)
-language (text)
-last_followed_up_date (timestamp with time zone)
-last_followed_up_type (text)
-latest_cost_sheet_shared_date (timestamp with time zone)
-latest_quote_shared_date (timestamp with time zone)
-latest_site_visit_date (timestamp with time zone)
-lead_generation_method (text)
-lead_temperature (character varying)
-lead_type (text)
-loan_amount (numeric)
-loan_type (text)
-mobile (numeric)
-monthly_income (double precision)
-notes (text)
-os (text)
-priority (character varying)
-project_id (uuid)
-property_finalized (boolean)
-property_type (text)
-property_value (numeric)
-purchase_timeline (character varying)
-purpose (text)
-quote_shared_count (integer)
-referrer (text)
-remarks (text)
-sales_stage (text)
-screen_resolution (text)
-search_vector (tsvector)
-selected_bank (character varying)
-session_duration_seconds (integer)
-session_start_at (timestamp with time zone)
-site_visit_count (integer)
-source (text)
-source_ad_name (text)
-source_ad_set (text)
-source_campaign_name (text)
-source_metadata (jsonb)
-status (text)
-submitted_at (timestamp with time zone)
-timezone (text)
-unique_lead_id (text)
-unit_choices (jsonb)
-unit_number (character varying)
-unit_price (numeric)
-unit_type (character varying)
-updated_at (timestamp without time zone)
-user_agent (text)
-utm_campaign (text)
-utm_medium (text)
-utm_source (text)
-viewport_size (text)

Business rules:

- Every query must be filtered for the current logged-in user.
- If owner-specific data: customers.owner_user_id = :user_id
- If followup-specific data: followups.assigned_user_id = :user_id
- If builder-specific data: customers.builder_id = :builder_id
- Do not apply creation_user_id filter if role is builder :
    - For example : 
        question: "show me all the customers for whom loans are disbursed ?"
        role: builder
        In this case, do not apply filter creation_user_id = :user_id because builders can see all customers related to their builder_id, not just the ones they created.

        Sample SQL snippet for builder role:

        SELECT c.customer_id, c.customer_name, c.disbursed_applications, c.total_disbursed_amount, l.current_status FROM customer_360_view AS c 
        JOIN loan_application_status_212d7de7 AS l ON c.unique_lead_id = l.unique_lead_id WHERE l.current_status = 'Disbursed' AND c.builder_id = 'eefa948a-2299-4af2-81e5-8ea3dc3e016e' ;

- Only Apply creation_user_id filter if role is builder when builder asks :
    - For example : 
        question: "show me all the customers created by <specific_user> for whom loans are disbursed "
        role: builder
        In this case, do not apply filter creation_user_id = :user_id because builders can see all customers related to their builder_id, not just the ones they created.

        Sample SQL snippet for builder role:

        SELECT c.customer_id, c.customer_name, c.disbursed_applications, c.total_disbursed_amount, l.current_status FROM customer_360_view AS c 
        JOIN loan_application_status_212d7de7 AS l ON c.unique_lead_id = l.unique_lead_id WHERE l.current_status = 'Disbursed' AND c.builder_id = '<builder_id>' and creation_user_id = '<creation_user_id>' ;


- Below are the loan status values exactly use these values when filtering for loan status:

| current_status |
| -------------- |
| Bank Login     |
| Cancelled      |
| Disbursed      |
| Follow-up      |
| KYC Pending    |
| Lost           |
| New Lead       |
| Sanctioned     |
| Under review   |

- Use following joining id's while applying joins:
- customer_360_view.unique_lead_id = loan_application_status_212d7de7.unique_lead_id


- Only read-only queries are allowed.
- Never generate INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE.
"""

# =========================================================
# STATE
# =========================================================

class GraphState(TypedDict):
    user_question: str
    user_context: Dict[str, Any]
    schema_context: str
    agent1_plan: Dict[str, Any]
    sql_query: str
    sql_validation_notes: str
    sql_result: List[Dict[str, Any]]
    execution_error: str
    final_answer: str


# =========================================================
# STRUCTURED OUTPUT MODELS
# =========================================================

class QueryPlan(BaseModel):
    question_reframed: str = Field(description="Clear explanation of user question")
    intent: str = Field(description="Business intent of the question")
    tables_required: List[str] = Field(description="Tables required to answer the question")
    columns_required: List[str] = Field(description="Important columns required")
    metrics_required: List[str] = Field(description="Metrics needed")
    filters_required: List[str] = Field(description="Filters including date/user filters")
    joins_required: List[str] = Field(description="Join logic if needed")
    aggregation_required: Optional[str] = Field(default="", description="Any aggregation needed")
    sort_logic: Optional[str] = Field(default="", description="Sort logic")
    limit_logic: Optional[str] = Field(default="", description="Limit logic")
    access_control_logic: str = Field(description="How to filter records for current user")


class SQLGenerationOutput(BaseModel):
    sql_query: str
    validation_notes: str


class NLAnswerOutput(BaseModel):
    answer: str


# =========================================================
# AGENT 1: QUERY UNDERSTANDING + TABLE IDENTIFICATION
# =========================================================

def agent1_query_planner(state: GraphState) -> GraphState:
    structured_llm = llm.with_structured_output(QueryPlan)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are Agent 1 for TranchIQ chatbot.

Your job:
1. Understand the user's question
2. Identify which tables are needed
3. Identify metrics, filters, joins, date logic
4. Ensure user-specific access is included
5. Produce a detailed execution plan for Agent 2

Rules:
- Use only the provided schema
- Be explicit about joins
- Be explicit about filters
- Always include user access control logic
- Do not write SQL
"""
            ),
            (
                "user",
                """
Schema:
{schema_context}

User context:
{user_context}

User question:
{user_question}
"""
            )
        ]
    )

    result = structured_llm.invoke(
        prompt.format_messages(
            schema_context=state["schema_context"],
            user_context=json.dumps(state["user_context"], indent=2),
            user_question=state["user_question"],
        )
    )

    state["agent1_plan"] = result.model_dump()

    logger.info(f"agent1_query_planner Executed")
    return state


# =========================================================
# AGENT 2: SQL GENERATION + VALIDATION
# =========================================================

def is_safe_select_query(sql: str) -> bool:
    sql_clean = sql.strip().lower()
    if not sql_clean.startswith("select"):
        return False

    forbidden = [
        "insert ", "update ", "delete ", "drop ", "alter ", "truncate ",
        "grant ", "revoke ", "create ", "replace "
    ]
    return not any(word in sql_clean for word in forbidden)


def agent2_sql_writer(state: GraphState) -> GraphState:
    structured_llm = llm.with_structured_output(SQLGenerationOutput)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are Agent 2 for TranchIQ chatbot.

Your job:
1. Write a correct PostgreSQL SELECT query
2. Use only provided schema and plan
3. Include mandatory user scoping
4. Validate syntax and business correctness
5. Return only read-only SQL

Rules:
- PostgreSQL / Supabase syntax
- Only SELECT queries
- Always enforce user access filter
- Use explicit joins
- Use aliases when helpful
- Keep SQL production-safe
"""
            ),
            (
                "user",
                """
Schema:
{schema_context}

User context:
{user_context}

Agent 1 plan:
{agent1_plan}
"""
            )
        ]
    )

    result = structured_llm.invoke(
        prompt.format_messages(
            schema_context=state["schema_context"],
            user_context=json.dumps(state["user_context"], indent=2),
            agent1_plan=json.dumps(state["agent1_plan"], indent=2),
        )
    )

    sql_query = result.sql_query.strip()

    # basic hard validation
    if not is_safe_select_query(sql_query):
        raise ValueError(f"Unsafe or non-SELECT SQL generated: {sql_query}")

    # simple required filter checks
    user_id = state["user_context"].get("user_id")
    builder_id = state["user_context"].get("builder_id")

    if user_id and str(user_id) not in sql_query and ":user_id" not in sql_query:
        print("[WARN] Generated SQL does not visibly include user_id. Please verify schema/filtering logic.")

    if builder_id and str(builder_id) not in sql_query and ":builder_id" not in sql_query:
        print("[WARN] Generated SQL does not visibly include builder_id. Please verify schema/filtering logic.")

    state["sql_query"] = sql_query
    state["sql_validation_notes"] = result.validation_notes

    logger.info(f"agent2_sql_writer Executed")
    return state


# =========================================================
# AGENT 3: SQL EXECUTION
# =========================================================

def substitute_params(sql: str, user_context: Dict[str, Any]) -> str:
    """
    Simple substitution.
    For production, use proper SQL parameter binding and/or a query builder.
    """
    safe_sql = sql
    for key, value in user_context.items():
        placeholder = f":{key}"
        if placeholder in safe_sql:
            if value is None:
                replacement = "NULL"
            elif isinstance(value, (int, float)):
                replacement = str(value)
            else:
                escaped = str(value).replace("'", "''")
                replacement = f"'{escaped}'"
            safe_sql = safe_sql.replace(placeholder, replacement)
    return safe_sql


def agent3_sql_executor(state: GraphState) -> GraphState:
    try:
        final_sql = substitute_params(state["sql_query"], state["user_context"])

        with psycopg.connect(SUPABASE_DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(final_sql)
                columns = [desc[0] for desc in cur.description] if cur.description else []
                rows = cur.fetchall()

        result = [dict(zip(columns, row)) for row in rows]

        state["sql_result"] = result
        state["execution_error"] = ""

        logger.info(f"agent3_sql_executor Executed, rows returned: {len(result)}")
        return state

    except Exception as e:
        state["sql_result"] = []
        state["execution_error"] = str(e)
        logger.error(f"agent3_sql_executor Error: {str(e)}")
        return state


# =========================================================
# AGENT 4: RESULT TO NATURAL LANGUAGE
# =========================================================

def agent4_answer_generator(state: GraphState) -> GraphState:
    structured_llm = llm.with_structured_output(NLAnswerOutput)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are Agent 4 for TranchIQ chatbot.

Your job:
1. Answer the user's question using SQL results
2. Be concise but business-friendly
3. Ensure answer is aligned with the question
4. If no rows found, say so clearly
5. Do not invent facts not present in results
"""
            ),
            (
                "user",
                """
User question:
{user_question}

Agent 1 plan:
{agent1_plan}

SQL query:
{sql_query}

SQL result:
{sql_result}

Execution error:
{execution_error}
"""
            )
        ]
    )

    result = structured_llm.invoke(
        prompt.format_messages(
            user_question=state["user_question"],
            agent1_plan=json.dumps(state["agent1_plan"], indent=2),
            sql_query=state["sql_query"],
            sql_result=json.dumps(state["sql_result"], indent=2, default=str),
            execution_error=state["execution_error"],
        )
    )

    state["final_answer"] = result.answer
    logger.info(f"agent4_answer_generator Executed")
    return state


# =========================================================
# GRAPH
# =========================================================

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("agent1_query_planner", agent1_query_planner)
    graph.add_node("agent2_sql_writer", agent2_sql_writer)
    graph.add_node("agent3_sql_executor", agent3_sql_executor)
    graph.add_node("agent4_answer_generator", agent4_answer_generator)

    graph.set_entry_point("agent1_query_planner")
    graph.add_edge("agent1_query_planner", "agent2_sql_writer")
    graph.add_edge("agent2_sql_writer", "agent3_sql_executor")
    graph.add_edge("agent3_sql_executor", "agent4_answer_generator")
    graph.add_edge("agent4_answer_generator", END)

    return graph.compile()


# =========================================================
# RUNNER
# =========================================================

def ask_tranchiq_bot(
    user_question: str,
    user_context: Dict[str, Any]
) -> Dict[str, Any]:
    app = build_graph()

    initial_state: GraphState = {
        "user_question": user_question,
        "user_context": user_context,
        "schema_context": SCHEMA_CONTEXT,
        "agent1_plan": {},
        "sql_query": "",
        "sql_validation_notes": "",
        "sql_result": [],
        "execution_error": "",
        "final_answer": "",
    }

    result = app.invoke(initial_state)
    return result


# =========================================================
# EXAMPLE
# =========================================================

"""
if __name__ == "__main__":
    user_context = {
        "user_id": "11111111-1111-1111-1111-111111111111",
        "builder_id": "22222222-2222-2222-2222-222222222222",
        "role": "relationship_manager"
    }

    questions = [
        "Show top 10 customers to follow up for today",
        "Which customers have pending tranches?",
        "Which customers took action on quote yesterday?",
        "Show customers who opened quotes in the last 7 days"
    ]

    for q in questions:
        print("\n" + "=" * 80)
        print("QUESTION:", q)

        result = ask_tranchiq_bot(q, user_context)

        print("\nAGENT 1 PLAN:")
        print(json.dumps(result["agent1_plan"], indent=2))

        print("\nSQL QUERY:")
        print(result["sql_query"])

        print("\nSQL VALIDATION NOTES:")
        print(result["sql_validation_notes"])

        print("\nSQL RESULT:")
        print(json.dumps(result["sql_result"][:5], indent=2, default=str))

        print("\nFINAL ANSWER:")
        print(result["final_answer"])
"""