import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic

from wren_langchain import WrenToolkit
from langchain.agents import create_agent
import json
import sqlite3
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from dotenv import load_dotenv
from wren_api.forecasting_tools import load_forecasting_models

load_dotenv()

# Initialize SQLite connection once for list_chat_threads
conn = sqlite3.connect("chat_memory.db", check_same_thread=False)
checkpointer = None

class CustomWrenToolkit(WrenToolkit):
    def get_tools(self, **kwargs) -> list:
        tools = super().get_tools(**kwargs)
        tools = [t for t in tools if t.name != "wren_query"]
        
        from langchain_core.tools import tool
        import uuid
        import pandas as pd

        @tool("wren_query")
        def wren_query(sql: str, limit: int = 100) -> dict:
            """Execute SQL through the Wren semantic layer and return rows preview."""
            print(f"[TOOL wren_query] Invoked with SQL query:\n{sql}")
            try:
                table = self.query(sql, limit=limit)
                print(f"[TOOL wren_query] Successfully executed preview query. Retrieved PyArrow Table: {table.num_rows} rows, {table.num_columns} columns.")
                
                from wren_langchain._format import format_query_content
                from wren_langchain._envelope import make_success
                
                content, warnings = format_query_content(table, total_rows=table.num_rows)
                
                data = {
                    "columns": table.column_names,
                    "rows": table.to_pylist(),
                    "row_count": table.num_rows,
                    "content_truncated": bool(warnings),
                }
                return make_success(content=content, data=data, warnings=warnings)
            except Exception as e:
                print(f"[TOOL wren_query] ERROR: {str(e)}")
                from wren_langchain._envelope import make_error
                return make_error(e)

        @tool("wren_export_csv")
        def wren_export_csv(sql: str) -> str:
            """Execute SQL and export the entire dataset as a downloadable CSV file.
            Use this when the user explicitly requests to download, export, save, or retrieve the results as a CSV file.
            """
            print(f"[TOOL wren_export_csv] Invoked with SQL query:\n{sql}")
            try:
                table = self.query(sql)
                print(f"[TOOL wren_export_csv] Successfully executed query. Retrieved PyArrow Table: {table.num_rows} rows, {table.num_columns} columns.")
                df = table.to_pandas()
                os.makedirs("exports", exist_ok=True)
                filename = f"export_{uuid.uuid4().hex[:8]}.csv"
                filepath = os.path.join("exports", filename)
                df.to_csv(filepath, index=False)
                print(f"[TOOL wren_export_csv] Exported DataFrame converted and saved to disk at: {filepath}")
                return f"CSV successfully exported! Download link: /exports/{filename}"
            except Exception as e:
                print(f"[TOOL wren_export_csv] ERROR failed to export CSV: {str(e)}")
                return f"Error exporting CSV: {str(e)}"

        @tool("wren_create_bar_chart")
        def wren_create_bar_chart(sql: str, x_column: str, y_column: str, title: str) -> str:
            """Execute SQL and create a beautiful bar chart image saved on disk, returning the Markdown image link.
            Use this tool whenever the user asks for a chart, graph, bar chart, visualization, or visual representation of data.
            """
            print(f"[TOOL wren_create_bar_chart] Invoked with SQL query:\n{sql}\nAxes: X={x_column}, Y={y_column}")
            try:
                table = self.query(sql)
                print(f"[TOOL wren_create_bar_chart] Successfully executed query. Retrieved PyArrow Table: {table.num_rows} rows.")
                df = table.to_pandas()
                
                # Check columns case-insensitively to be extremely user-friendly and robust
                cols_lower = {col.lower(): col for col in df.columns}
                if x_column.lower() not in cols_lower:
                    return f"Error: X-axis column '{x_column}' not found. Available columns are: {', '.join(df.columns)}"
                if y_column.lower() not in cols_lower:
                    return f"Error: Y-axis column '{y_column}' not found. Available columns are: {', '.join(df.columns)}"
                
                # Map to exact case
                x_col_exact = cols_lower[x_column.lower()]
                y_col_exact = cols_lower[y_column.lower()]
                
                # Plotting configuration
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                plt.close('all')
                plt.clf()
                
                sns.set_theme(style="darkgrid", rc={
                    "axes.facecolor": "#1e1e2e",
                    "figure.facecolor": "#11111b",
                    "grid.color": "#313244",
                    "text.color": "#cdd6f4",
                    "axes.labelcolor": "#cdd6f4",
                    "xtick.color": "#a6adc8",
                    "ytick.color": "#a6adc8",
                })
                
                fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
                
                # Render bar chart
                sns.barplot(
                    data=df, 
                    x=x_col_exact, 
                    y=y_col_exact, 
                    ax=ax, 
                    palette="viridis", 
                    hue=x_col_exact, 
                    legend=False
                )
                
                # Title and Label formatting
                ax.set_title(title, fontsize=16, fontweight='bold', pad=15, color="#cdd6f4")
                ax.set_xlabel(x_col_exact.replace('_', ' ').title(), fontsize=12, fontweight='bold', labelpad=10)
                ax.set_ylabel(y_col_exact.replace('_', ' ').title(), fontsize=12, fontweight='bold', labelpad=10)
                
                # Rotate X ticks if too many or too long
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                os.makedirs("exports", exist_ok=True)
                filename = f"chart_{uuid.uuid4().hex[:8]}.png"
                filepath = os.path.join("exports", filename)
                
                plt.savefig(
                    filepath, 
                    format="png", 
                    bbox_inches='tight', 
                    facecolor=fig.get_facecolor(), 
                    edgecolor='none'
                )
                plt.close(fig)
                
                print(f"[TOOL wren_create_bar_chart] Successfully generated bar chart image at: {filepath}")
                return f"![{title}](/exports/{filename})\n\n*(Visualized: {title} via /exports/{filename})*"
            except Exception as e:
                print(f"[TOOL wren_create_bar_chart] ERROR failed to generate bar chart: {str(e)}")
                return f"Error generating bar chart: {str(e)}"

        @tool("global_forecast")
        def global_forecast(material_id: int) -> dict:
            """Forecast next week's total order quantity for a specific product across all locations.
            Use this tool when the user explicitly asks to predict, forecast, or estimate future sales
            for a product WITHOUT specifying a city or location.
            Input: material_id (integer, e.g., 20187)
            Returns: prediction with historical data and confidence interval.
            """
            print(f"[TOOL global_forecast] Invoked for material_id: {material_id}")
            from wren_api.forecasting_tools import global_forecast_impl
            return global_forecast_impl(material_id)

        @tool("location_forecast")
        def location_forecast(material_id: int, city: str) -> dict:
            """Forecast next week's order quantity for a specific product IN A SPECIFIC CITY.
            Use this tool when the user explicitly asks to predict, forecast, or estimate future sales
            for a product IN A SPECIFIC CITY or location.
            Input: material_id (integer, e.g., 20187) and city (string, e.g., 'SURAT' or 'AHMEDABAD').
            The city name will be automatically normalized (e.g., 'BARODA' -> 'VADODARA', 'THOR' -> 'THOL').
            Returns: prediction with historical data and confidence interval.
            """
            print(f"[TOOL location_forecast] Invoked for material_id: {material_id}, city: {city}")
            from wren_api.forecasting_tools import location_forecast_impl
            return location_forecast_impl(material_id, city)

        @tool("customer_product_forecast")
        def customer_product_forecast(customer_id: int, material_id: int) -> dict:
            """Forecast next week's order quantity for a specific customer and product combination.
            Use this tool when the user asks to predict, forecast, or estimate future sales
            for a specific CUSTOMER and PRODUCT together (e.g., "what will customer X buy of product Y?").
            Input: customer_id (integer) and material_id (integer).
            Returns: prediction with historical data and confidence interval.
            """
            print(f"[TOOL customer_product_forecast] Invoked for customer_id: {customer_id}, material_id: {material_id}")
            from wren_api.forecasting_tools import customer_product_forecast_impl
            return customer_product_forecast_impl(customer_id, material_id)

        @tool("customer_forecast")
        def customer_forecast(customer_id: int) -> dict:
            """Forecast next week's TOTAL order quantity for a specific customer across ALL products.
            Use this tool when the user asks to predict, forecast, or estimate total future volume
            for a specific CUSTOMER (e.g., "what will customer X order next week?").
            This aggregates all product-level predictions for that customer.
            Input: customer_id (integer).
            Returns: total predicted quantity with top products breakdown.
            """
            print(f"[TOOL customer_forecast] Invoked for customer_id: {customer_id}")
            from wren_api.forecasting_tools import customer_forecast_impl
            return customer_forecast_impl(customer_id)

        tools.extend([wren_query, wren_export_csv, wren_create_bar_chart, global_forecast, location_forecast, customer_product_forecast, customer_forecast])
        return tools

# Project configs
PROJECTS = {
    # "start_report": {
    #     "path": "/home/web-h-063/start_report_wren",
    #     "description": "Start Report Wren project",
    # },
    # "riddhi_gsp": {
    #     "path": "/home/web-h-063/riddhi-gsp",
    #     "description": "Riddhi GSP Wren project",
    # },
    # "riddhi-gsp-gemini": {
    #     "path": "/home/web-h-063/riddhi-gsp-test",
    #     "description": "Riddhi GSP Gemini project",
    # },
    "customer-sales": {
        "path": "/home/web-h-063/sales_wren",
        "description": "Customer Sales project",
    },
}

# Cache for initialized toolkits
_toolkit_cache: dict[str, WrenToolkit] = {}
_models = {}


def get_model(provider: str = "minimax", thinking_level: str | None = None):
    global _models
    provider = (provider or "minimax").lower()
    if provider == "gemini":
        provider = "gemini-2.5-flash"
    
    cache_key = (provider, thinking_level)
    if cache_key not in _models:
        if provider.startswith("gemini"):
            api_key = os.environ.get("GEMINI_API_KEY_")
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            kwargs = {
                "model": provider,
                "api_key": api_key,
                "include_thoughts": True,
            }
            if provider != "gemini-2.5-flash":               
                if thinking_level and thinking_level != "off":
                    kwargs["thinking_level"] = thinking_level
                
            _models[cache_key] = ChatGoogleGenerativeAI(**kwargs)
        else:
            _models[cache_key] = ChatAnthropic(
                model="MiniMax-M2.7",
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                base_url="https://api.minimax.io/anthropic",
            )
    return _models[cache_key]


def load_project_env(project_path: str):
    env_path = os.path.join(project_path, ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip("'\"")
                        os.environ[key] = val
            print(f"Loaded environment variables from {env_path}")
        except Exception as e:
            print(f"Error loading {env_path}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global checkpointer
    load_project_env(".")
    # Pre-initialize both projects at startup
    async with AsyncSqliteSaver.from_conn_string("chat_memory.db") as saver:
        checkpointer = saver
        for project_id, config in PROJECTS.items():
            try:
                load_project_env(config["path"])
                toolkit = CustomWrenToolkit.from_project(config["path"])
                _toolkit_cache[project_id] = {
                    "toolkit": toolkit,
                    "description": config["description"],
                    "path": config["path"],
                }
                print(f"Initialized project: {project_id}")
            except Exception as e:
                print(f"Failed to initialize {project_id}: {e}")

        # Load forecasting models at startup
        load_forecasting_models()
        print("Forecasting models loaded at startup")

        yield

    # Cleanup
    _toolkit_cache.clear()


from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title="Wren AI Agent API", lifespan=lifespan)

# Mount static exports folder to serve generated CSV files
from fastapi.staticfiles import StaticFiles
os.makedirs("exports", exist_ok=True)
app.mount("/exports", StaticFiles(directory="exports"), name="exports")

# Add CORS middleware to support standalone local HTML loads
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_custom_system_prompt(toolkit) -> str:
    prompt = (
        "You use Wren Engine as the semantic layer for data querying. SQL targets MDL model names (defined in `target/mdl.json`); the engine translates to the target database dialect.\n\n"
        "# Workflow for every data question\n\n"
        "Run these steps in order:\n\n"
        "1. Recall similar past NL→SQL pairs:\n"
        "   `wren_recall_queries(question=\"<user's question>\", limit=3)`\n"
        "   Use the results as few-shot examples. Empty results are fine — continue\n"
        "   to the next step. Do NOT skip this step on the grounds that the question\n"
        "   seems simple; past pairs may use better joins, filters, or column names\n"
        "   than you would write from scratch.\n\n"
        "2. Fetch schema and business context:\n"
        "   `wren_fetch_context(question=\"<user's question>\")`\n"
        "   Optionally narrow scope with `model=\"<name>\"` or\n"
        "   `item_type=\"model\" | \"column\" | \"relationship\" | \"view\"`.\n\n"
        "3. Compose SQL targeting Wren model names — NEVER raw database tables.\n\n"
        "4. (Complex queries only) Verify with `wren_dry_plan(sql=\"...\")` before\n"
        "   executing. \"Complex\" = subqueries, multi-step CTEs, or JOINs not\n"
        "   already defined as MDL relationships. Simple GROUP BY or\n"
        "   model-defined JOINs can skip this step.\n\n"
        "5. Execute: `wren_query(sql=\"...\", limit=100)`. Raise the limit only when you genuinely need more rows. For any query where a full downloadable spreadsheet would be useful to the user, you should additionally execute `wren_export_csv(sql=\"...\")` with the same or a corresponding detailed query to generate a downloadable CSV file.\n\n"
        "6. Persist the NL→SQL pair: `wren_store_query(nl=\"<user's original question>\", sql=\"<the SQL you ran>\", tags=[...])`.\n\n"
        "   Store BY DEFAULT after a successful query. Skip ONLY when:\n"
        "   - The query failed (`ok=false`).\n"
        "   - The user said the result is wrong.\n"
        "   - The SQL is exploratory (e.g. `SELECT * FROM x LIMIT 10` with no\n"
        "     analytical clauses).\n"
        "   - There is no natural-language question (e.g. the user pasted raw SQL).\n"
        "   - The user explicitly said don't save.\n\n"
        "   The `nl` value should be the user's original question, not a paraphrase.\n\n"
        "# Error recovery\n\n"
        "If a tool returns `ok=false`, inspect `error.phase` and `error.message`:\n\n"
        "- `SQL_PARSING` → SQL syntax error. Read the message, fix, and retry.\n"
        "- `METADATA_FETCHING` / `MDL_EXTRACTION` → wrong model or column name;\n"
        "  use `wren_fetch_context(question=\"<bad name>\", item_type=\"model\")` (or `item_type=\"column\"`) to find the correct one.\n"
        "- `SQL_EXECUTION` → database-side error. `error.metadata.dialect_sql` shows\n"
        "  the translated SQL — diagnose against the message (type mismatch, missing\n"
        "  function, permission, timeout). Add explicit `CAST` or simplify the query\n"
        "  if needed.\n\n"
        "Don't silently abandon. Either fix and retry, or report the failure to the\n"
        "user along with what you tried.\n\n"
        "# Things to avoid\n\n"
        "- Don't guess model or column names — call `wren_fetch_context` first.\n"
        "- Don't skip `wren_recall_queries` on questions that seem \"simple\" — past pairs are often the most accurate template.\n"
        "- Don't store failed queries, queries the user said are wrong, or exploratory queries.\n"
        "- Don't store SQL that has no clear natural-language question.\n"
        "- Don't write SQL against raw database tables — always use MDL model names.\n\n"
        "# Human-readable output (master data resolution)\n\n"
        "When presenting order summaries, line-item details, billing/AR records, forecasts, or any business-facing answer, prefer descriptive labels over raw foreign-key IDs.\n\n"
        "- Before answering, use `wren_fetch_context` to find master or denormalized models and relationships for the entities in the question.\n"
        "- In SQL, JOIN or select from models/views that expose names, descriptions, and other human-readable fields tied to each ID column you would otherwise show.\n"
        "- In tables and narrative summaries, show the descriptive value each ID represents.\n"
        "- Show a bare numeric or alphanumeric ID only when master data is unavailable, the lookup returns no row, or the user explicitly asks for the ID.\n"
        "- Document numbers that are themselves the business identifier the user asked about (order number, invoice number, etc.) may be shown as-is.\n"
        "- For coded fields (status, type, block reason), show a readable label or meaning when available in the data or schema context; include the code only if it adds clarity.\n\n"
        "## Available tools\n"
        "- `wren_dry_plan`: Plan SQL through MDL and return the expanded target-dialect SQL.\n"
        "- `wren_list_models`: List all models defined in this Wren project with column counts and descriptions.\n"
        "- `wren_fetch_context`: Fetch relevant schema and business context for an analytical question.\n"
        "- `wren_recall_queries`: Recall up to *limit* past NL→SQL pairs similar to *question*.\n"
        "- `wren_store_query`: Save a confirmed natural-language → SQL pair for future recall.\n"
        "- `wren_query`: Execute SQL through the Wren semantic layer and return a preview of the rows.\n"
        "- `wren_export_csv`: Execute SQL and export the entire dataset as a downloadable CSV file.\n"
        "- `wren_create_bar_chart`: Execute SQL and create a beautiful bar chart image saved on disk, returning the Markdown image link.\n\n"
        "- `global_forecast`: Forecast next week's total order quantity for a product across all locations. Use ONLY when user asks to predict/forecast sales WITHOUT specifying a city. Input: material_id (int).\n"
        "- `location_forecast`: Forecast next week's order quantity for a product in a specific city. Use ONLY when user asks to predict/forecast sales IN A SPECIFIC CITY or location. Input: material_id (int) and city (str).\n"
        "- `customer_product_forecast`: Forecast next week's order quantity for a specific CUSTOMER and PRODUCT. Use when user asks about a specific customer's demand for a specific product. Input: customer_id (int) and material_id (int).\n"
        "- `customer_forecast`: Forecast next week's TOTAL order quantity for a customer across ALL products. Use when user asks about total customer volume. Input: customer_id (int).\n\n"
        "CSV EXPORT RULE: The `wren_export_csv` tool executes a SQL query, automatically exports the entire dataset as a CSV file, and returns a download link. Whenever a user asks to retrieve, download, or export a detailed list, structured rows, or tabular data, you MUST call `wren_export_csv` and provide the resulting download link verbatim in your final response.\n\n"
        "PROACTIVE RECORD EXPORTING RULE: To deliver a premium and proactive experience, when the user asks an aggregate or count question about a specific subset of records (e.g., 'How many customers were created in the last 30 days?' or 'How many orders are pending?'), you should:\n"
        "1. Call `wren_query` with the aggregate COUNT SQL to retrieve the count value.\n"
        "2. Call `wren_export_csv` with a detailed `SELECT *` query targeting the underlying records (e.g. `SELECT * FROM customers WHERE created_at >= ...` or `SELECT * FROM orders WHERE status = ...`) to generate a downloadable CSV of the actual records.\n"
        "3. In your final response, display BOTH the count result clearly AND include the generated download link verbatim so the user has immediate access to the raw records.\n\n"
        "BAR CHART RULE: You should ONLY call `wren_create_bar_chart` to render a bar chart when the user EXPLICITLY asks for a chart, graph, visualization, or plot (e.g., 'Show me a chart of ...' or 'Plot the ...'). Do NOT generate a bar chart proactively if they only ask for comparison, grouping, or ranking questions in standard text.\n"
        "When a chart is explicitly requested, you MUST call `wren_create_bar_chart` with:\n"
        "1. A grouping SQL query that aggregates the data (e.g., `SELECT country, COUNT(*) as count FROM customers GROUP BY country`).\n"
        "2. The categorical axis column name (`x_column`).\n"
        "3. The numerical metric column name (`y_column`).\n"
        "4. A highly descriptive title (`title`).\n"
        "Include the returned markdown image syntax verbatim in your final response. Do NOT generate a bar chart for simple, single-metric queries with no groupings (e.g., 'How many total customers do we have?' should just be a simple text count, not a chart with a single bar).\n\n"
        "PRODUCT NAME RESOLUTION (before forecasting): When the user mentions a product by name, brand, or family "
        "and does NOT give an exact material ID or full SKU description:\n"
        "1. Resolve candidates with `wren_query` against `vw_material_revenue` (or `vw_sales_orders` if you need city-scoped variants):\n"
        "   - Match on `item_description` (case-insensitive) using the user's search terms\n"
        "   - Return `material`, `item_description`, and optionally `material_group`\n"
        "   - If the user named a city/region, prefer materials that appear in that city (filter via `vw_sales_orders` or forecast-relevant sales data).\n"
        "2. **0 matches** → Tell the user no product was found; suggest alternate spelling. Do NOT call forecasting tools.\n"
        "3. **Exactly 1 match** → Use that `material_id` and proceed with the appropriate forecast tool.\n"
        "4. **2+ matches (variants)** → Do NOT call `global_forecast`, `location_forecast`, or `customer_product_forecast` yet.\n"
        "   - Reply with a short clarification question asking which variant they mean.\n"
        "   - List each variant as: `material_id` — full `item_description`\n"
        "   - Wait for the user to pick one (by number, material ID, or full description).\n"
        "   - Only after they choose, call the forecast tool with that `material_id`.\n"
        "5. Never guess or default to the first row in the result set.\n\n"
        "FORECASTING RULE: Only call forecasting tools when the user EXPLICITLY asks to predict, forecast, or estimate future sales or demand. Do NOT call forecasting tools for historical data questions. Forecasting tools require a resolved `material_id` — follow PRODUCT NAME RESOLUTION first when the user gave only a product name or family.\n"
        "- If user asks \"what will be the sales next week for product X\" (single resolved material_id) -> call `global_forecast`\n"
        "- If user asks \"what will be the sales of product X in SURAT next week\" (single resolved material_id) -> call `location_forecast`\n"
        "- If user asks \"what will customer X buy of product Y next week\" (single resolved material_id) -> call `customer_product_forecast`\n"
        "- If user asks \"what will customer X order next week\" or \"customer X total volume\" -> call `customer_forecast`\n"
        "- If user asks \"what were the sales last week\" or any historical question -> use `wren_query` ONLY\n\n"
        "When presenting a forecast to the user, follow this structure:\n"
        "1. State the prediction clearly: \"Based on the past {N} weeks of sales data...\"\n"
        "2. Show the historical trend (last N weeks qty)\n"
        "3. State the prediction with confidence interval\n"
        "4. Explain model used (global or location-based)\n\n"
        "FORECASTING CONFIDENCE: Predictions include a confidence interval based on model RMSE. For global forecasts the interval is ±312,286 units, for location forecasts ±28,248 units, and for customer-product forecasts ±2,130 units. Wider intervals indicate higher uncertainty.\n\n"
        "CRITICAL: At the very end of your final response, you MUST always provide a clear 'Citations & SQL Explanation' section. In this section, list exactly which database tables and fields/columns were used to get these results, and briefly explain the logic of the SQL query that was executed."
    )
    
    from pathlib import Path
    instructions_file = Path(toolkit._project_path) / "instructions.md"
    if instructions_file.exists():
        body = instructions_file.read_text().strip()
        if body:
            prompt += f"\n\n## Project-specific instructions\n\n{body}"
            
    return prompt


@app.get("/")
def serve_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


class ChatRequest(BaseModel):
    project_id: str
    question: str
    thread_id: str | None = None
    provider: str | None = None
    thinking_level: str | None = None
    stream_mode: str | None = "updates"


@app.get("/projects")
def list_projects():
    """List all available projects and their tools."""
    result = []
    for project_id, data in _toolkit_cache.items():
        toolkit = data["toolkit"]
        tools = [t.name for t in toolkit.get_tools()]
        result.append({
            "project_id": project_id,
            "description": data["description"],
            "path": data["path"],
            "tools": tools,
            "system_prompt": get_custom_system_prompt(toolkit),
        })
    return result


@app.get("/projects/{project_id}")
def get_project(project_id: str):
    """Get details for a specific project."""
    if project_id not in _toolkit_cache:
        raise HTTPException(status_code=404, detail="Project not found")
    data = _toolkit_cache[project_id]
    toolkit = data["toolkit"]
    return {
        "project_id": project_id,
        "description": data["description"],
        "path": data["path"],
        "tools": [t.name for t in toolkit.get_tools()],
        "system_prompt": get_custom_system_prompt(toolkit),
    }


def format_messages_chunk(chunk):
    """Format a token-level messages chunk as SSE."""
    import json
    if isinstance(chunk, dict) and "data" in chunk:
        chunk = chunk["data"]
    if not isinstance(chunk, tuple) or len(chunk) < 1:
        return
    message = chunk[0]
    
    # Handle AIMessageChunk
    if message.__class__.__name__ == "AIMessageChunk":
        if hasattr(message, "content"):
            if isinstance(message.content, str) and message.content:
                yield f"event: text\ndata: {json.dumps({'text': message.content})}\n\n"
            elif isinstance(message.content, list):
                for block in message.content:
                    if isinstance(block, dict):
                        if block.get("type") == "thinking" and block.get("thinking"):
                            yield f"event: thinking\ndata: {json.dumps({'thinking': block.get('thinking')})}\n\n"
                        elif block.get("type") == "text" and block.get("text"):
                            yield f"event: text\ndata: {json.dumps({'text': block.get('text')})}\n\n"
        
        # Handle tool call chunks
        if hasattr(message, "tool_call_chunks") and message.tool_call_chunks:
            for tc in message.tool_call_chunks:
                if tc.get("name"):
                    yield f"event: tool_call\ndata: {json.dumps({'name': tc['name'], 'input': {}})}\n\n"

    # Handle ToolMessage
    elif message.__class__.__name__ == "ToolMessage":
        if hasattr(message, "content"):
            try:
                content = json.loads(message.content) if isinstance(message.content, str) else message.content
                yield f"event: tool_result\ndata: {json.dumps({'tool': message.name, 'content': content})}\n\n"
            except:
                yield f"event: tool_result\ndata: {json.dumps({'tool': message.name, 'content': str(message.content)})}\n\n"


def format_stream_event(event):
    """Format a streaming event as SSE."""
    import json
    if "model" in event:
        for message in event["model"]["messages"]:
            content = message.content if hasattr(message, "content") else str(message)
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tc in message.tool_calls:
                    yield f"event: tool_call\ndata: {json.dumps({'name': tc['name'], 'input': tc.get('args', {})})}\n\n"
            if hasattr(message, "content"):
                if isinstance(message.content, str):
                    if message.content:
                        yield f"event: text\ndata: {json.dumps({'text': message.content})}\n\n"
                elif isinstance(message.content, list):
                    for block in message.content:
                        if isinstance(block, dict):
                            if block.get("type") == "thinking":
                                yield f"event: thinking\ndata: {json.dumps({'thinking': block.get('thinking', '')})}\n\n"
                            elif block.get("type") == "text":
                                yield f"event: text\ndata: {json.dumps({'text': block.get('text', '')})}\n\n"
                            elif block.get("type") == "tool_use":
                                yield f"event: tool_call\ndata: {json.dumps({'name': block.get('name'), 'input': block.get('input')})}\n\n"
    elif "tools" in event:
        for message in event["tools"]["messages"]:
            if hasattr(message, "content"):
                try:
                    content = json.loads(message.content) if isinstance(message.content, str) else message.content
                    yield f"event: tool_result\ndata: {json.dumps({'tool': message.name, 'content': content})}\n\n"
                except:
                    yield f"event: tool_result\ndata: {json.dumps({'tool': message.name, 'content': str(message.content)})}\n\n"


@app.get("/chat/threads")
def list_chat_threads():
    """List distinct active thread sessions stored in SQLite."""
    try:
        cursor = conn.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY checkpoint_id DESC")
        threads = [row[0] for row in cursor.fetchall()]
        return {"threads": threads}
    except Exception as e:
        return {"threads": []}


@app.delete("/chat/threads/{thread_id}")
def delete_chat_thread(thread_id: str):
    """Delete a distinct thread session from SQLite checkpoints."""
    try:
        conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        conn.commit()
        return {"status": "success", "thread_id": thread_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/chat/history/{project_id}/{thread_id}")
async def get_chat_history(project_id: str, thread_id: str, provider: str | None = None, thinking_level: str | None = None):
    """Retrieve message history for a given session thread."""
    if project_id not in _toolkit_cache:
        raise HTTPException(status_code=404, detail="Project not found")
    
    toolkit_data = _toolkit_cache[project_id]
    toolkit = toolkit_data["toolkit"]
    model = get_model(provider, thinking_level)
    custom_system_prompt = get_custom_system_prompt(toolkit)
    
    agent = create_agent(
        model=model,
        tools=toolkit.get_tools(),
        checkpointer=checkpointer,
        system_prompt=custom_system_prompt,
    )
    
    config = {"configurable": {"thread_id": thread_id}}
    state = await agent.aget_state(config)
    
    messages = state.values.get("messages", [])
    serialized = []
    for msg in messages:
        # Filter out system prompts and intermediate tool execution outputs
        if msg.type in ("system", "tool"):
            continue
        role = "user" if msg.type == "human" else "assistant"
        
        # Handle lists of dict blocks (like text/thinking blocks from MiniMax) gracefully
        content_str = ""
        thinking_str = ""
        if isinstance(msg.content, str):
            content_str = msg.content
        elif isinstance(msg.content, list):
            text_parts = []
            thinking_parts = []
            for block in msg.content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "thinking":
                        thinking_parts.append(block.get("thinking", ""))
                    elif "text" in block:
                        text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            content_str = "".join(text_parts)
            thinking_str = "".join(thinking_parts)
        else:
            content_str = str(msg.content)

        serialized.append({
            "role": role,
            "content": content_str,
            "thinking": thinking_str,
            "type": msg.type
        })
    return {"thread_id": thread_id, "messages": serialized}


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Streaming chat endpoint - creates agent on the fly and streams events."""
    if req.project_id not in _toolkit_cache:
        raise HTTPException(status_code=404, detail="Project not found")

    import uuid
    thread_id = req.thread_id or f"session_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    toolkit_data = _toolkit_cache[req.project_id]
    load_project_env(toolkit_data["path"])
    toolkit = toolkit_data["toolkit"]
    model = get_model(req.provider, req.thinking_level)

    custom_system_prompt = get_custom_system_prompt(toolkit)

    agent = create_agent(
        model=model,
        tools=toolkit.get_tools(),
        checkpointer=checkpointer,
        system_prompt=custom_system_prompt,
    )
    stream_mode = req.stream_mode or "updates"

    async def event_generator():
        try:
            yield f"event: thread_id\ndata: {json.dumps({'thread_id': thread_id})}\n\n"
            if stream_mode == "messages":
                async for chunk in agent.astream(
                    {"messages": [{"role": "user", "content": req.question}]},
                    config=config,
                    stream_mode="messages",
                    version="v2",
                ):
                    for line in format_messages_chunk(chunk):
                        yield line
            else:
                async for event in agent.astream(
                    {"messages": [{"role": "user", "content": req.question}]},
                    config=config,
                ):
                    for line in format_stream_event(event):
                        yield line
            yield "event: done\ndata: {}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )