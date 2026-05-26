# Customer-Product Forecasting Tools Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two LangChain tools (`customer_product_forecast` and `customer_forecast`) to the Wren AI agent that predict next-week demand at the customer-product and customer-total levels respectively.

**Architecture:** The customer-product XGBoost model is loaded at startup alongside the existing global/location models. Two new `@tool` decorators are added inside `CustomWrenToolkit.get_tools()`. The system prompt is updated with rules for when to call these new tools. Customer ID is stored as VARCHAR in PostgreSQL, requiring string conversion in SQL queries.

**Tech Stack:** FastAPI, LangChain `@tool`, joblib, XGBoost (trained model), Wren AI semantic layer

---

## File Structure

```
wren_api/
├── main.py                                          # Modified: CustomWrenToolkit, system prompt
├── forecasting_tools.py                            # Modified: add customer-product functions
└── exports/                                        # Static files (already exists)
```

---

## Models Location

```
/home/web-h-063/Documents/sales/models/
├── customer_product_forecast_model.pkl    # NEW - Customer-Product model
├── customer_product_feature_cols.pkl      # NEW - Feature column names
├── weekly_forecast_model.pkl              # Existing - Global model
├── weekly_feature_cols.pkl                # Existing - Global features
├── location_forecast_model.pkl            # Existing - Location model
├── city_encoder.pkl                       # Existing - City LabelEncoder
└── feature_cols.pkl                       # Existing - Location features
```

---

## Model Performance Reference

### Customer-Product Model (Test Metrics)
- MAE: 2,130 units
- Feature columns: `['lag_1', 'lag_2', 'lag_3', 'lag_4', 'rolling_avg_3w', 'rolling_avg_4w', 'rolling_std_3w', 'week_number', 'month']`
- Primary signal: lag features + rolling averages (same as other models)

---

## Database Schema Notes

- `nk.sales_order_header.sold_to_party` is **VARCHAR** (not integer)
- Always pass `str(customer_id)` in SQL params, never raw integer
- Query pattern for customer+material:
```sql
WHERE h.sold_to_party = %s AND v.material = %s AND h.document_date >= '2025-12-01'
GROUP BY 1, 2, 3, 4, 5 ORDER BY 1 DESC LIMIT 4
```

---

## Task Decomposition

### Task 1: Add customer-product model globals and loader

**Files:**
- Modify: `/home/web-h-063/Documents/auto-data-intelligence/wren_api/forecasting_tools.py:14-34`

- [ ] **Step 1: Add customer-product model globals**

Add after existing global declarations (after line 20):

```python
# Customer-Product model globals (loaded at startup)
_cp_model = None
_cp_feature_cols = None
```

- [ ] **Step 2: Update load_forecasting_models() to load customer-product model**

Modify the `load_forecasting_models()` function to add:

```python
def load_forecasting_models():
    """Load all forecasting models into global variables. Called at startup."""
    global _global_model, _global_feature_cols
    global _location_model, _location_encoder, _location_feature_cols
    global _cp_model, _cp_feature_cols  # ADD THIS

    _global_model = joblib.load(os.path.join(MODELS_DIR, 'weekly_forecast_model.pkl'))
    _global_feature_cols = joblib.load(os.path.join(MODELS_DIR, 'weekly_feature_cols.pkl'))

    _location_model = joblib.load(os.path.join(MODELS_DIR, 'location_forecast_model.pkl'))
    _location_encoder = joblib.load(os.path.join(MODELS_DIR, 'city_encoder.pkl'))
    _location_feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_cols.pkl'))

    # NEW: Load customer-product model
    _cp_model = joblib.load(os.path.join(MODELS_DIR, 'customer_product_forecast_model.pkl'))
    _cp_feature_cols = joblib.load(os.path.join(MODELS_DIR, 'customer_product_feature_cols.pkl'))
```

- [ ] **Step 3: Commit**

```bash
git add wren_api/forecasting_tools.py
git commit -m "feat: add customer-product model globals and loader"
```

---

### Task 2: Add customer-product data fetch and feature compute functions

**Files:**
- Modify: `/home/web-h-063/Documents/auto-data-intelligence/wren_api/forecasting_tools.py`

- [ ] **Step 1: Add fetch function after existing fetch functions**

Add after `fetch_weekly_data_location()` (around line 86):

```python
def fetch_weekly_data_customer_product(customer_id: int, material_id: int) -> pd.DataFrame:
    """Fetch last 4 weeks of weekly data for a specific customer + product."""
    conn = get_db_connection()
    query = """
        SELECT DATE_TRUNC('week', h.document_date) AS week,
               h.sold_to_party AS customer_id,
               c.name AS customer_name,
               v.material,
               v.item_description,
               SUM(v.order_quantity) AS qty_ordered
        FROM nk.sales_order_items v
        JOIN nk.sales_order_header h ON h.sales_document = v.sales_document
        LEFT JOIN nk.customer_master c ON c.customer = h.sold_to_party
        WHERE h.sold_to_party = %s AND v.material = %s AND h.document_date >= '2025-12-01'
        GROUP BY 1, 2, 3, 4, 5
        ORDER BY 1 DESC LIMIT 4
    """
    df = pd.read_sql(query, conn, params=(str(customer_id), material_id))
    conn.close()
    return df.sort_values('week').reset_index(drop=True)
```

- [ ] **Step 2: Add compute function for customer-product features**

Add after `compute_location_features()` (around line 122):

```python
def compute_customer_product_features(qty_values: np.ndarray, week, feature_cols) -> pd.DataFrame:
    """Compute feature vector for customer-product model."""
    lag_1 = qty_values[-1] if len(qty_values) >= 1 else 0
    lag_2 = qty_values[-2] if len(qty_values) >= 2 else 0
    lag_3 = qty_values[-3] if len(qty_values) >= 3 else 0
    lag_4 = qty_values[-4] if len(qty_values) >= 4 else 0
    rolling_avg_3w = np.mean(qty_values[-3:]) if len(qty_values) >= 1 else 0
    rolling_avg_4w = np.mean(qty_values[-4:]) if len(qty_values) >= 1 else 0
    rolling_std_3w = np.std(qty_values[-3:]) if len(qty_values) >= 2 else 0

    features = [[
        lag_1, lag_2, lag_3, lag_4,
        rolling_avg_3w, rolling_avg_4w, rolling_std_3w,
        week.isocalendar()[1], week.month
    ]]
    return pd.DataFrame(features, columns=feature_cols).fillna(0)
```

- [ ] **Step 3: Commit**

```bash
git add wren_api/forecasting_tools.py
git commit -m "feat: add customer-product fetch and feature compute functions"
```

---

### Task 3: Add customer_product_forecast_impl function

**Files:**
- Modify: `/home/web-h-063/Documents/auto-data-intelligence/wren_api/forecasting_tools.py`

- [ ] **Step 1: Add customer_product_forecast_impl at end of file**

Add after `location_forecast_impl()` (after line 237):

```python
def customer_product_forecast_impl(customer_id: int, material_id: int) -> dict:
    """
    Implementation of customer_product_forecast tool.
    Returns prediction + historical data + confidence interval for a customer-product pair.
    NOTE: customer_id must be converted to string for SQL (sold_to_party is VARCHAR).
    """
    try:
        df = fetch_weekly_data_customer_product(customer_id, material_id)

        if len(df) == 0:
            return make_error(f"No data found for customer {customer_id} and product {material_id}")

        customer_name = df.iloc[0]['customer_name']
        description = df.iloc[0]['item_description']
        qty_values = df['qty_ordered'].values
        most_recent_week = df['week'].max()

        # Compute features
        features_df = compute_customer_product_features(qty_values, most_recent_week, _cp_feature_cols)

        # Predict
        prediction = _cp_model.predict(features_df)[0]

        # Confidence interval using RMSE (2,130 from model evaluation)
        rmse = 2130
        confidence_interval = f"± {int(rmse):,} units"

        # Historical weeks
        historical = [
            {"week": str(row['week'].date()), "qty": int(row['qty_ordered'])}
            for _, row in df.sort_values('week', ascending=False).iterrows()
        ]

        result_data = {
            "customer_id": int(customer_id),
            "customer_name": customer_name,
            "material": int(material_id),
            "description": description,
            "last_week_qty": int(qty_values[-1]),
            "last_week_date": str(most_recent_week.date()),
            "predicted_next_week": max(0, int(prediction)),
            "confidence_interval": confidence_interval,
            "historical_weeks": historical,
            "confidence": "low" if len(df) < 4 else "medium",
            "model_used": "customer_product_forecast",
            "base_rmse": rmse
        }

        return make_success(
            content=f"Customer-product forecast for {customer_name} - {description}: predicted {max(0, int(prediction)):,} units (± {rmse:,}) based on {len(df)} weeks of data",
            data=result_data
        )

    except Exception as e:
        return make_error(f"Customer-product forecast failed: {str(e)}")
```

- [ ] **Step 2: Commit**

```bash
git add wren_api/forecasting_tools.py
git commit -m "feat: add customer_product_forecast_impl function"
```

---

### Task 4: Add customer_forecast_impl (aggregate all products for a customer)

**Files:**
- Modify: `/home/web-h-063/Documents/auto-data-intelligence/wren_api/forecasting_tools.py`

- [ ] **Step 1: Add customer_forecast_impl function**

Add after `customer_product_forecast_impl()`:

```python
def customer_forecast_impl(customer_id: int) -> dict:
    """
    Implementation of customer_forecast tool.
    Returns aggregated predictions across ALL products for a customer.
    Sums individual product-level predictions for the total.
    NOTE: customer_id must be converted to string for SQL (sold_to_party is VARCHAR).
    """
    try:
        conn = get_db_connection()
        query = """
            SELECT DATE_TRUNC('week', h.document_date) AS week,
                   h.sold_to_party AS customer_id,
                   c.name AS customer_name,
                   v.material,
                   v.item_description,
                   SUM(v.order_quantity) AS qty_ordered
            FROM nk.sales_order_items v
            JOIN nk.sales_order_header h ON h.sales_document = v.sales_document
            LEFT JOIN nk.customer_master c ON c.customer = h.sold_to_party
            WHERE h.sold_to_party = %s AND h.document_date >= '2025-12-01'
            GROUP BY 1, 2, 3, 4, 5
            ORDER BY 1, 4
        """
        df = pd.read_sql(query, conn, params=(str(customer_id),))
        conn.close()

        if len(df) == 0:
            return make_error(f"No data found for customer {customer_id}")

        customer_name = df.iloc[0]['customer_name']

        # For each product, predict and sum
        product_predictions = []
        for material_id, group in df.groupby('material'):
            group = group.sort_values('week')
            qty_values = group['qty_ordered'].values[-4:]

            if len(qty_values) < 1:
                continue

            description = group.iloc[0]['item_description']
            most_recent_week = group['week'].max()

            # Build features
            lag_1 = qty_values[-1] if len(qty_values) >= 1 else 0
            lag_2 = qty_values[-2] if len(qty_values) >= 2 else 0
            lag_3 = qty_values[-3] if len(qty_values) >= 3 else 0
            lag_4 = qty_values[-4] if len(qty_values) >= 4 else 0
            rolling_avg_3w = np.mean(qty_values[-3:]) if len(qty_values) >= 1 else 0
            rolling_avg_4w = np.mean(qty_values[-4:]) if len(qty_values) >= 1 else 0
            rolling_std_3w = np.std(qty_values[-3:]) if len(qty_values) >= 2 else 0
            week_number = most_recent_week.isocalendar()[1]
            month = most_recent_week.month

            features = [[lag_1, lag_2, lag_3, lag_4, rolling_avg_3w, rolling_avg_4w,
                         rolling_std_3w, week_number, month]]
            features_df = pd.DataFrame(features, columns=_cp_feature_cols).fillna(0)
            prediction = max(0, _cp_model.predict(features_df)[0])

            product_predictions.append({
                "material": int(material_id),
                "description": description,
                "last_week_qty": int(lag_1),
                "predicted_qty": int(prediction)
            })

        # Sort by predicted qty desc
        product_predictions.sort(key=lambda x: x['predicted_qty'], reverse=True)
        total_predicted = sum(p['predicted_qty'] for p in product_predictions)

        result_data = {
            "customer_id": int(customer_id),
            "customer_name": customer_name,
            "num_products": len(product_predictions),
            "total_predicted_qty": total_predicted,
            "product_predictions": product_predictions[:10],  # Top 10
            "confidence": "medium",
            "model_used": "customer_forecast"
        }

        return make_success(
            content=f"Customer forecast for {customer_name}: predicted total {total_predicted:,} units across {len(product_predictions)} products based on recent purchase history",
            data=result_data
        )

    except Exception as e:
        return make_error(f"Customer forecast failed: {str(e)}")
```

- [ ] **Step 2: Commit**

```bash
git add wren_api/forecasting_tools.py
git commit -m "feat: add customer_forecast_impl function"
```

---

### Task 5: Add tools to CustomWrenToolkit in main.py

**Files:**
- Modify: `/home/web-h-063/Documents/auto-data-intelligence/wren_api/main.py:159-184`

- [ ] **Step 1: Add customer_product_forecast tool after location_forecast**

Add after the `location_forecast` tool definition (around line 182):

```python
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
```

- [ ] **Step 2: Add customer_forecast tool**

Add after `customer_product_forecast`:

```python
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
```

- [ ] **Step 3: Update tools.extend() line**

Modify the tools.extend line (around line 184) to include the new tools:

```python
        tools.extend([wren_query, wren_export_csv, wren_create_bar_chart, global_forecast, location_forecast, customer_product_forecast, customer_forecast])
```

- [ ] **Step 4: Commit**

```bash
git add wren_api/main.py
git commit -m "feat: add customer_product_forecast and customer_forecast tools"
```

---

### Task 6: Update system prompt with new tool instructions

**Files:**
- Modify: `/home/web-h-063/Documents/auto-data-intelligence/wren_api/main.py:312-393`

- [ ] **Step 1: Update FORECASTING RULE section**

Find the existing FORECASTING RULE section and add entries for the new tools:

Original (around line 382):
```python
"FORECASTING RULE: Only call `global_forecast` or `location_forecast` when the user EXPLICITLY asks to predict, forecast, or estimate future sales or demand. Do NOT call forecasting tools for historical data questions.\n"
"- If user asks \"what will be the sales next week for product X\" -> call `global_forecast`\n"
"- If user asks \"what will be the sales of product X in SURAT next week\" -> call `location_forecast`\n"
"- If user asks \"what were the sales last week\" or any historical question -> use `wren_query` ONLY\n"
```

Replace with:
```python
"FORECASTING RULE: Only call forecasting tools when the user EXPLICITLY asks to predict, forecast, or estimate future sales or demand. Do NOT call forecasting tools for historical data questions.\n"
"- If user asks \"what will be the sales next week for product X\" -> call `global_forecast`\n"
"- If user asks \"what will be the sales of product X in SURAT next week\" -> call `location_forecast`\n"
"- If user asks \"what will customer X buy of product Y next week\" -> call `customer_product_forecast`\n"
"- If user asks \"what will customer X order next week\" or \"customer X total volume\" -> call `customer_forecast`\n"
"- If user asks \"what were the sales last week\" or any historical question -> use `wren_query` ONLY\n"
```

- [ ] **Step 2: Update tool descriptions in same section**

Find line 368-369:
```python
"- `global_forecast`: Forecast next week's total order quantity for a product across all locations. Use ONLY when user asks to predict/forecast sales WITHOUT specifying a city. Input: material_id (int).\n"
"- `location_forecast`: Forecast next week's order quantity for a product in a specific city. Use ONLY when user asks to predict/forecast sales IN A SPECIFIC CITY or location. Input: material_id (int) and city (str).\n\n"
```

Replace with:
```python
"- `global_forecast`: Forecast next week's total order quantity for a product across all locations. Use ONLY when user asks to predict/forecast sales WITHOUT specifying a city. Input: material_id (int).\n"
"- `location_forecast`: Forecast next week's order quantity for a product in a specific city. Use ONLY when user asks to predict/forecast sales IN A SPECIFIC CITY or location. Input: material_id (int) and city (str).\n"
"- `customer_product_forecast`: Forecast next week's order quantity for a specific CUSTOMER and PRODUCT. Use when user asks about a specific customer's demand for a specific product. Input: customer_id (int) and material_id (int).\n"
"- `customer_forecast`: Forecast next week's TOTAL order quantity for a customer across ALL products. Use when user asks about total customer volume. Input: customer_id (int).\n\n"
```

- [ ] **Step 3: Add customer-forecast confidence info**

Find line 391:
```python
"FORECASTING CONFIDENCE: Predictions include a confidence interval based on model RMSE. For global forecasts the interval is ±312,286 units, for location forecasts ±28,248 units. Wider intervals indicate higher uncertainty.\n\n"
```

Replace with:
```python
"FORECASTING CONFIDENCE: Predictions include a confidence interval based on model RMSE. For global forecasts the interval is ±312,286 units, for location forecasts ±28,248 units, and for customer-product forecasts ±2,130 units. Wider intervals indicate higher uncertainty.\n\n"
```

- [ ] **Step 4: Commit**

```bash
git add wren_api/main.py
git commit -m "feat: update system prompt with new forecasting tools"
```

---

### Task 7: Test the integration

**Files:**
- Modify: `/home/web-h-063/Documents/sales/modeltesting.ipynb`

- [ ] **Step 1: Verify models load at startup**

Run the FastAPI app and check logs show:
```
Forecasting models loaded at startup
```

- [ ] **Step 2: Test customer_product_forecast with curl**

```bash
curl -X POST "http://localhost:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"project_id": "customer-sales", "question": "What will customer 100475 buy of product 20187 next week?"}'
```

Expected: Tool `customer_product_forecast` is called, returns prediction around 448 units (based on test output from modeltesting.ipynb).

- [ ] **Step 3: Test customer_forecast with curl**

```bash
curl -X POST "http://localhost:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"project_id": "customer-sales", "question": "What will customer 100475 order next week?"}'
```

Expected: Tool `customer_forecast` is called, returns aggregated total across all products.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/plans/2026-05-25-customer-product-forecasting-tools.md
git commit -m "docs: add customer-product forecasting implementation plan"
```

---

## Verification Checklist

After implementation, verify:
- [ ] `load_forecasting_models()` loads customer_product_forecast_model.pkl and customer_product_feature_cols.pkl
- [ ] `customer_product_forecast_impl(customer_id, material_id)` returns valid prediction
- [ ] `customer_forecast_impl(customer_id)` returns aggregated totals
- [ ] Both new `@tool` decorators are in `CustomWrenToolkit.get_tools()`
- [ ] System prompt includes rules for when to call `customer_product_forecast` and `customer_forecast`
- [ ] Agent correctly routes customer+product questions to `customer_product_forecast`
- [ ] Agent correctly routes customer-total questions to `customer_forecast`

---

## SQL Query Reference

### Customer-Product Query
```sql
SELECT DATE_TRUNC('week', h.document_date) AS week,
       h.sold_to_party AS customer_id,
       c.name AS customer_name,
       v.material,
       v.item_description,
       SUM(v.order_quantity) AS qty_ordered
FROM nk.sales_order_items v
JOIN nk.sales_order_header h ON h.sales_document = v.sales_document
LEFT JOIN nk.customer_master c ON c.customer = h.sold_to_party
WHERE h.sold_to_party = %s AND v.material = %s AND h.document_date >= '2025-12-01'
GROUP BY 1, 2, 3, 4, 5
ORDER BY 1 DESC LIMIT 4
```

### Customer-Total Query (all products)
```sql
SELECT DATE_TRUNC('week', h.document_date) AS week,
       h.sold_to_party AS customer_id,
       c.name AS customer_name,
       v.material,
       v.item_description,
       SUM(v.order_quantity) AS qty_ordered
FROM nk.sales_order_items v
JOIN nk.sales_order_header h ON h.sales_document = v.sales_document
LEFT JOIN nk.customer_master c ON c.customer = h.sold_to_party
WHERE h.sold_to_party = %s AND h.document_date >= '2025-12-01'
GROUP BY 1, 2, 3, 4, 5
ORDER BY 1, 4
```

**Important:** `sold_to_party` is VARCHAR — always use `str(customer_id)` in SQL params.

---

Plan complete and saved to `docs/superpowers/plans/2026-05-25-customer-product-forecasting-tools.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?