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

# Customer-Product model globals (loaded at startup)
_cp_model = None
_cp_feature_cols = None


def load_forecasting_models():
    """Load all forecasting models into global variables. Called at startup."""
    global _global_model, _global_feature_cols
    global _location_model, _location_encoder, _location_feature_cols
    global _cp_model, _cp_feature_cols

    _global_model = joblib.load(os.path.join(MODELS_DIR, 'weekly_forecast_model.pkl'))
    _global_feature_cols = joblib.load(os.path.join(MODELS_DIR, 'weekly_feature_cols.pkl'))

    _location_model = joblib.load(os.path.join(MODELS_DIR, 'location_forecast_model.pkl'))
    _location_encoder = joblib.load(os.path.join(MODELS_DIR, 'city_encoder.pkl'))
    _location_feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_cols.pkl'))

    _cp_model = joblib.load(os.path.join(MODELS_DIR, 'customer_product_forecast_model.pkl'))
    _cp_feature_cols = joblib.load(os.path.join(MODELS_DIR, 'customer_product_feature_cols.pkl'))


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
