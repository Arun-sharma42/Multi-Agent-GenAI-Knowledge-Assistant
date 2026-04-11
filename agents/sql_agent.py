"""
agents/sql_agent.py
────────────────────
Text-to-SQL agent flow:
  1. Receive natural language query
  2. Build prompt with the database schema
  3. Ask LLM to write a safe SELECT query
  4. Validate the SQL (no destructive operations)
  5. Execute against SQLite
  6. Return formatted results

Interview talking point:
  "Security is critical here — I strip anything that isn't a SELECT
   statement before execution, and I use parameterised queries where
   possible. In production, you'd add a dedicated SQL firewall."
"""

import re
import sqlite3
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import BaseAgent, AgentResponse
from database.db_setup import get_connection, get_schema_description
from utils.llm_client import get_llm


SQL_SYSTEM_PROMPT = """
You are an expert SQLite query writer. Given a natural language question and a database schema, write a single valid SQLite SELECT query.

Rules:
1. Output ONLY the raw SQL query — no markdown, no backticks, no explanation.
2. Only write SELECT statements. Never INSERT, UPDATE, DELETE, DROP, or ALTER.
3. Use table aliases for clarity (e.g. s for students, r for results).
4. Use ROUND(value, 2) for decimal marks.
5. Always LIMIT results to 50 rows maximum unless the user asks for all.
6. If the question cannot be answered with the schema, output: UNSUPPORTED

{schema}
"""


class SQLAgent(BaseAgent):
    """
    Converts natural language to SQL, executes it, and formats results.
    """

    def __init__(self):
        super().__init__("SQLAgent")
        # Low temperature for precise, deterministic SQL generation
        self.llm    = get_llm(temperature=0.0)
        self.schema = get_schema_description()

    def run(self, query: str, context: str = "") -> AgentResponse:
        """
        Full Text-to-SQL pipeline.
        """
        # ── Step 1: Generate SQL ───────────────────────────────────────────
        sql = self._generate_sql(query)
        self.log.info(f"Generated SQL: {sql}")

        if sql == "UNSUPPORTED":
            return AgentResponse(
                answer=(
                    "❓ I couldn't translate your question into a database query.\n\n"
                    "Try asking something like:\n"
                    "- 'Show students with marks above 80'\n"
                    "- 'What is the average score in Mathematics?'\n"
                    "- 'List all Grade A students'"
                ),
                agent_name=self.name,
                metadata={"sql": sql},
            )

        # ── Step 2: Validate SQL (security check) ─────────────────────────
        validation_error = self._validate_sql(sql)
        if validation_error:
            return AgentResponse(
                answer=f"⛔ SQL validation failed: {validation_error}",
                agent_name=self.name,
                success=False,
                metadata={"sql": sql, "error": validation_error},
            )

        # ── Step 3: Execute ────────────────────────────────────────────────
        rows, columns, error = self._execute_sql(sql)

        if error:
            return AgentResponse(
                answer=f"❌ SQL execution error: {error}\n\nGenerated SQL:\n```sql\n{sql}\n```",
                agent_name=self.name,
                success=False,
                metadata={"sql": sql, "error": error},
            )

        # ── Step 4: Format results ─────────────────────────────────────────
        formatted = self._format_results(rows, columns)
        row_count = len(rows)

        answer = (
            f"{formatted}\n\n"
            f"*{row_count} row(s) returned*\n\n"
            f"**SQL used:**\n```sql\n{sql}\n```"
        )

        return AgentResponse(
            answer=answer,
            agent_name=self.name,
            metadata={
                "sql":        sql,
                "row_count":  row_count,
                "columns":    columns,
            },
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _generate_sql(self, query: str) -> str:
        """Ask the LLM to write SQL for the given natural language query."""
        system = SQL_SYSTEM_PROMPT.format(schema=self.schema)
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"Question: {query}"),
        ]
        response = self.llm.invoke(messages)
        # Strip any markdown fences the model might have added
        sql = response.content.strip()
        sql = re.sub(r"^```(?:sql)?", "", sql, flags=re.IGNORECASE).strip()
        sql = re.sub(r"```$", "", sql).strip()
        return sql

    def _validate_sql(self, sql: str) -> str | None:
        """
        Block any non-SELECT operations.
        Returns an error message if invalid, None if safe.
        """
        normalized = sql.strip().upper()
        dangerous  = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
                      "CREATE", "TRUNCATE", "REPLACE", "EXEC"]

        for keyword in dangerous:
            if re.search(rf"\b{keyword}\b", normalized):
                return f"Destructive keyword detected: '{keyword}'"

        if not normalized.startswith("SELECT"):
            return "Query must start with SELECT"

        return None

    def _execute_sql(
        self, sql: str
    ) -> tuple[list[dict], list[str], str | None]:
        """
        Execute the SQL and return (rows, columns, error).
        rows is a list of dicts keyed by column name.
        """
        try:
            conn = get_connection()
            cur  = conn.execute(sql)
            columns = [desc[0] for desc in cur.description] if cur.description else []
            rows    = [dict(row) for row in cur.fetchall()]
            conn.close()
            return rows, columns, None
        except sqlite3.Error as e:
            return [], [], str(e)

    def _format_results(
        self, rows: list[dict], columns: list[str]
    ) -> str:
        """
        Render query results as a Markdown table for clean display in the UI.
        """
        if not rows:
            return "✅ Query executed successfully — no rows returned."

        # Header
        header    = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"

        # Rows
        data_rows = []
        for row in rows:
            cells = [str(row.get(col, "")) for col in columns]
            data_rows.append("| " + " | ".join(cells) + " |")

        return "\n".join([header, separator] + data_rows)
