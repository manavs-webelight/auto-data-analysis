# Forecasting Tools Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two LangChain tools (`global_forecast` and `location_forecast`) to the Wren AI agent that call pre-trained XGBoost models to predict next-week product demand. The agent will only call these tools when the user explicitly asks to forecast or predict sales.

**Architecture:** Models are loaded at FastAPI startup via lifespan and stored in global variables. Two new `@tool` decorators are added inside `CustomWrenToolkit.get_tools()`. A system prompt section instructs the agent on when to use these tools and how to present results. Prediction output includes historical weeks data, confidence interval, and model explanation.

**Tech Stack:** FastAPI, LangChain `@tool`, joblib, XGBoost (trained models), Wren AI semantic layer

---

## File Structure

```
wren_api/
├── main.py                                          # Modified: lifespan, CustomWrenToolkit, system prompt
├── forecasting_tools.py                            # Create: forecasting tool functions
└── exports/                                         # Static files (already exists)
```

---

## Models Location
```
/home/web-h-063/Documents/sales/models/
├── weekly_forecast_model.pkl       # Global model
├── weekly_feature_cols.pkl         # Global model features
├── location_forecast_model.pkl     # Location-aware model
├── city_encoder.pkl                # City LabelEncoder for location model
└── feature_cols.pkl                 # Location model features
```

---

## Model Performance Reference

### Global Model (Test Metrics)
- MAE: 42,491 units
- RMSE: 312,286 units
- Primary signal: rolling_avg_3w (53%), rolling_avg_4w (31%)

### Location Model (Test Metrics)
- MAE: 3,441 units
- RMSE: 28,248 units
- Primary signal: lag_1 (57%), rolling_avg_3w (19%)

Confidence interval derived from RMSE: prediction ± RMSE

---

## Task Decomposition

### Task 1: Create `forecasting_tools.py` with tool functions

**Files:**
- Create: `/home/web-h-063/Documents/auto-data-intelligence/wren_api/forecasting_tools.py`

- [ ] **Step 1: Write the forecasting tool functions**

```python
# forecasting_tools.py
import joblib
import os
import numpy as np
import pandas as pd
import psycopg2
from typing import Optional
from wren_langchain._envelope import make_success, make_error


MODELS_DIR = '/home/web-h-063/Documents/sales/models'

# Global model globals (loaded at startup)
_global_model = None
_global_feature_cols = None

# Location model globals (loaded at startup)
_location_model = None
_location_encoder = None
_location_feature_cols = None


def load_forecasting_models():
    """Load all forecasting models into global variables. Called at startup."""
    global _global_model, _global_feature_cols
    global _location_model, _location_encoder, _location_feature_cols

    _global_model = joblib.load(os.path.join(MODELS_DIR, 'weekly_forecast_model.pkl'))
    _global_feature_cols = joblib.load(os.path.join(MODELS_DIR, 'weekly_feature_cols.pkl'))

    _location_model = joblib.load(os.path.join(MODELS_DIR, 'location_forecast_model.pkl'))
    _location_encoder = joblib.load(os.path.join(MODELS_DIR, 'city_encoder.pkl'))
    _location_feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_cols.pkl'))


def normalize_city(city_name: str) -> str:
    """Normalize city name: uppercase + trim + typo mappings."""
    mappings = {'BARODA': 'VADODARA', 'THOR': 'THOL'}
    return mappings.get(city_name.upper().strip(), city_name.upper().strip())


def get_db_connection():
    """Create a new PostgreSQL connection."""
    return psycopg2.connect(
        host='localhost',
        port=5438,
        database='customer_sales',
        user='postgres',
        password='postgres'
    )


def fetch_weekly_data_global(material_id: int) -> pd.DataFrame:
    """Fetch last 4 weeks of weekly data for a product (all locations combined)."""
    conn = get_db_connection()
    query = """
        SELECT DATE_TRUNC('week', h.document_date) AS week, v.material,
               v.item_description, SUM(v.order_quantity) AS qty_ordered
        FROM nk.sales_order_items v
        JOIN nk.sales_order_header h ON h.sales_document = v.sales_document
        WHERE v.material = %s AND h.document_date >= '2025-12-01'
        GROUP BY 1, 2, 3 ORDER BY 1 DESC LIMIT 4
    """
    df = pd.read_sql(query, conn, params=(material_id,))
    conn.close()
    return df.sort_values('week').reset_index(drop=True)


def fetch_weekly_data_location(material_id: int, city_normalized: str) -> pd.DataFrame:
    """Fetch last 4 weeks of weekly data for a product in a specific city."""
    conn = get_db_connection()
    query = """
        SELECT DATE_TRUNC('week', h.document_date) AS week, v.material,
               v.item_description, UPPER(TRIM(c.city)) AS customer_city,
               SUM(v.order_quantity) AS qty_ordered
        FROM nk.sales_order_items v
        JOIN nk.sales_order_header h ON h.sales_document = v.sales_document
        JOIN nk.customer_master c ON c.customer = h.sold_to_party
        WHERE v.material = %s AND UPPER(TRIM(c.city)) = %s
          AND h.document_date >= '2025-12-01'
        GROUP BY 1, 2, 3, 4 ORDER BY 1 DESC LIMIT 4
    """
    df = pd.read_sql(query, conn, params=(material_id, city_normalized))
    conn.close()
    return df.sort_values('week').reset_index(drop=True)


def compute_global_features(qty_values: np.ndarray, week, feature_cols) -> pd.DataFrame:
    """Compute feature vector for global model."""
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


def compute_location_features(qty_values: np.ndarray, week, city_encoded: int, feature_cols) -> pd.DataFrame:
    """Compute feature vector for location model."""
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
        week.isocalendar()[1], week.month, city_encoded
    ]]
    return pd.DataFrame(features, columns=feature_cols).fillna(0)


def global_forecast_impl(material_id: int) -> dict:
    """
    Implementation of global_forecast tool.
    Returns prediction + historical data + confidence interval.
    """
    try:
        df = fetch_weekly_data_global(material_id)

        if len(df) == 0:
            return make_error(f"Product {material_id} not found in data")

        description = df.iloc[0]['item_description']
        qty_values = df['qty_ordered'].values
        most_recent_week = df['week'].max()

        # Compute features
        features_df = compute_global_features(qty_values, most_recent_week, _global_feature_cols)

        # Predict
        prediction = _global_model.predict(features_df)[0]

        # Confidence interval using RMSE as approximate std
        rmse = 312286  # from model evaluation
        confidence_interval = f"± {int(rmse):,} units"

        # Historical weeks
        historical = [
            {"week": str(row['week'].date()), "qty": int(row['qty_ordered'])}
            for _, row in df.sort_values('week', ascending=False).iterrows()
        ]

        result_data = {
            "material": int(material_id),
            "description": description,
            "last_week_qty": int(qty_values[-1]),
            "last_week_date": str(most_recent_week.date()),
            "predicted_next_week": int(prediction),
            "confidence_interval": confidence_interval,
            "historical_weeks": historical,
            "confidence": "low" if len(df) < 4 else "medium",
            "model_used": "global_weekly_forecast",
            "base_rmse": rmse
        }

        return make_success(
            content=f"Global forecast for product {material_id}: predicted {int(prediction):,} units (± {rmse:,}) based on {len(df)} weeks of data",
            data=result_data
        )

    except Exception as e:
        return make_error(f"Global forecast failed: {str(e)}")


def location_forecast_impl(material_id: int, city: str) -> dict:
    """
    Implementation of location_forecast tool.
    Returns prediction + historical data + confidence interval for product in a city.
    """
    try:
        city_normalized = normalize_city(city)

        df = fetch_weekly_data_location(material_id, city_normalized)

        if len(df) == 0:
            return make_error(f"No data found for product {material_id} in city '{city_normalized}'")

        description = df.iloc[0]['item_description']
        qty_values = df['qty_ordered'].values
        most_recent_week = df['week'].max()

        # Encode city
        try:
            city_encoded = int(_location_encoder.transform([city_normalized])[0])
        except:
            return make_error(f"City '{city_normalized}' is not in the training data. The agent should query the database to find the correct city name.")

        # Compute features
        features_df = compute_location_features(qty_values, most_recent_week, city_encoded, _location_feature_cols)

        # Predict
        prediction = _location_model.predict(features_df)[0]

        # Confidence interval using RMSE
        rmse = 28248  # from model evaluation
        confidence_interval = f"± {int(rmse):,} units"

        # Historical weeks
        historical = [
            {"week": str(row['week'].date()), "qty": int(row['qty_ordered'])}
            for _, row in df.sort_values('week', ascending=False).iterrows()
        ]

        result_data = {
            "material": int(material_id),
            "city_original": city,
            "city_normalized": city_normalized,
            "description": description,
            "last_week_qty": int(qty_values[-1]),
            "last_week_date": str(most_recent_week.date()),
            "predicted_next_week": int(prediction),
            "confidence_interval": confidence_interval,
            "historical_weeks": historical,
            "confidence": "low" if len(df) < 4 else "medium",
            "model_used": "location_forecast",
            "base_rmse": rmse
        }

        return make_success(
            content=f"Location forecast for product {material_id} in {city_normalized}: predicted {int(prediction):,} units (± {rmse:,}) based on {len(df)} weeks of data",
            data=result_data
        )

    except Exception as e:
        return make_error(f"Location forecast failed: {str(e)}")
```

- [ ] **Step 2: Verify imports work**

Run: `cd /home/web-h-063/Documents/auto-data-intelligence/wren_api && python -c "from wren_langchain._envelope import make_success, make_error; print('envelope imports OK')"`
Expected: `envelope imports OK`

---

### Task 2: Modify `main.py` - Lifespan to load models at startup

**Files:**
- Modify: `/home/web-h-063/Documents/auto-data-intelligence/wren_api/main.py`

- [ ] **Step 1: Add import for forecasting_tools at top of main.py**

Find the existing imports section (after `from wren_langchain import WrenToolkit` or similar) and add:
```python
from forecasting_tools import load_forecasting_models
```

- [ ] **Step 2: Call `load_forecasting_models()` in the lifespan function**

In the `async with AsyncSqliteSaver.from_conn_string(...)` block inside `lifespan()`, after the toolkit cache initialization loop (around line 256), add:
```python
# Load forecasting models at startup
load_forecasting_models()
print("Forecasting models loaded at startup")
```

- [ ] **Step 3: Verify FastAPI starts without errors**

Run: `cd /home/web-h-063/Documents/auto-data-intelligence/wren_api && python -c "import main; print('main.py imports OK')"`
Expected: No import errors

---

### Task 3: Modify `main.py` - Add forecasting tools to CustomWrenToolkit

**Files:**
- Modify: `/home/web-h-063/Documents/auto-data-intelligence/wren_api/main.py:23-159`

- [ ] **Step 1: In `CustomWrenToolkit.get_tools()`, add the two new tool definitions**

Inside the `get_tools` method, after the `@tool("wren_create_bar_chart")` definition and before `tools.extend([...])`, add:

```python
        @tool("global_forecast")
        def global_forecast(material_id: int) -> dict:
            """Forecast next week's total order quantity for a specific product across all locations.
            Use this tool when the user explicitly asks to predict, forecast, or estimate future sales
            for a product WITHOUT specifying a city or location.
            Input: material_id (integer, e.g., 20187)
            Returns: prediction with historical data and confidence interval.
            """
            print(f"[TOOL global_forecast] Invoked for material_id: {material_id}")
            from forecasting_tools import global_forecast_impl
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
            from forecasting_tools import location_forecast_impl
            return location_forecast_impl(material_id, city)
```

- [ ] **Step 2: Update `tools.extend([...])` to include new tools**

Change line 158 from:
```python
tools.extend([wren_query, wren_export_csv, wren_create_bar_chart])
```
to:
```python
tools.extend([wren_query, wren_export_csv, wren_create_bar_chart, global_forecast, location_forecast])
```

---

### Task 4: Modify `main.py` - Update system prompt with forecasting instructions

**Files:**
- Modify: `/home/web-h-063/Documents/auto-data-intelligence/wren_api/main.py:329-350` (the "Available tools" section)

- [ ] **Step 1: Add forecasting tools documentation to system prompt**

In the `get_custom_system_prompt()` function, in the "Available tools" section (around line 337), add after the `wren_create_bar_chart` description:

```
"- `global_forecast`: Forecast next week's total order quantity for a product across all locations. Use ONLY when user asks to predict/forecast sales WITHOUT specifying a city. Input: material_id (int).
"- `location_forecast`: Forecast next week's order quantity for a product in a specific city. Use ONLY when user asks to predict/forecast sales IN A SPECIFIC CITY or location. Input: material_id (int) and city (str).\n\n"
```

- [ ] **Step 2: Add forecasting usage rules to system prompt**

After the "BAR CHART RULE" section (around line 349), add:

```
"FORECASTING RULE: Only call `global_forecast` or `location_forecast` when the user EXPLICITLY asks to predict, forecast, or estimate future sales or demand. Do NOT call forecasting tools for historical data questions.
- If user asks "what will be the sales next week for product X" -> call `global_forecast`
- If user asks "what will be the sales of product X in SURAT next week" -> call `location_forecast`
- If user asks "what were the sales last week" or any historical question -> use `wren_query` ONLY

When presenting a forecast to the user, follow this structure:
1. State the prediction clearly: "Based on the past {N} weeks of sales data..."
2. Show the historical trend (last N weeks qty)
3. State the prediction with confidence interval
4. Explain model used (global or location-based)

FORECASTING CONFIDENCE: Predictions include a confidence interval based on model RMSE. For global forecasts the interval is ±312,286 units, for location forecasts ±28,248 units. Wider intervals indicate higher uncertainty.\n\n"
```

---

### Task 5: Test the integration

**Files:**
- Test: `/home/web-h-063/Documents/auto-data-intelligence/wren_api/main.py`

- [ ] **Step 1: Verify models load at startup**

Run the FastAPI app briefly:
Run: `cd /home/web-h-063/Documents/auto-data-intelligence/wren_api && timeout 10 uvicorn main:app --host 0.0.0.0 --port 8000 2>&1 || true`
Expected: Should see "Forecasting models loaded at startup" and no import errors

- [ ] **Step 2: Test tool loading**

Run: `cd /home/web-h-063/Documents/auto-data-intelligence/wren_api && python -c "
from main import CustomWrenToolkit
# Check tools exist - don't need full init, just check the code path
print('CustomWrenToolkit class defined OK')
"`
Expected: No errors

- [ ] **Step 3: Verify forecasting_tools functions work with real data**

Run in the wren_api directory after models are exported:
Run: `cd /home/web-h-063/Documents/auto-data-intelligence/wren_api && python -c "
import sys
sys.path.insert(0, '.')
from forecasting_tools import load_forecasting_models, global_forecast_impl, location_forecast_impl
load_forecasting_models()
result = global_forecast_impl(20187)
print('global_forecast result:', result.get('data', result) if hasattr(result, 'get') else result)
result2 = location_forecast_impl(20187, 'SURAT')
print('location_forecast result:', result2.get('data', result2) if hasattr(result2, 'get') else result2)
"`
Expected: Both return success with prediction data

---

## System Prompt Additions Summary

The system prompt (in `get_custom_system_prompt()`) needs these additions:

### 1. Tool Descriptions (in Available tools list)
```
- `global_forecast`: Forecast next week's total order quantity for a product across all locations.
- `location_forecast`: Forecast next week's order quantity for a product in a specific city.
```

### 2. New Rule Section (after BAR CHART RULE)
```
FORECASTING RULE: ... (detailed rules as above)
```

---

## Implementation Sequence

1. Task 1: Create `forecasting_tools.py` (self-contained, no dependencies on main.py changes)
2. Task 2: Modify `main.py` lifespan to load models at startup
3. Task 3: Add tools to `CustomWrenToolkit`
4. Task 4: Update system prompt
5. Task 5: Test integration

---

## Verification Checklist

- [ ] `python -c "from forecasting_tools import load_forecasting_models; load_forecasting_models(); print('OK')"` works
- [ ] FastAPI starts without import errors
- [ ] `/projects` endpoint lists `global_forecast` and `location_forecast` in tools
- [ ] Tool descriptions appear in system prompt
- [ ] Forecasting rules appear in system prompt
- [ ] `global_forecast_impl(20187)` returns valid prediction with historical data
- [ ] `location_forecast_impl(20187, 'SURAT')` returns valid prediction with historical data
- [ ] Unknown city returns error with guidance to use SQL to find correct city

---

## Notes

- Models are loaded from `/home/web-h-063/Documents/sales/models/` - ensure this path is accessible from the wren_api directory
- The `make_success` and `make_error` envelope functions format the output consistently with other Wren tools
- Confidence intervals use model RMSE as approximate standard deviation (not a true prediction interval, but a reasonable approximation)
- City normalization handles BARODA→VADODARA, THOR→THOL as specified by user
- The agent is responsible for fetching material_id and city from the database using `wren_query` before calling forecasting tools if the user didn't provide them explicitly