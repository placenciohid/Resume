# Data_pipeline_for_Event_Prediction.py

# Import necessary libraries
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
    DateType,
)
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
from pyspark.ml.feature import (
    StandardScaler,
    OneHotEncoder,
    StringIndexer,
    VectorAssembler,
    QuantileDiscretizer,
)

# Import DataFrames
df1 = spark.read.csv("data1.csv", header=True)
df2 = spark.read.csv("data2.csv", header=True)
df3 = spark.read.csv("data3.csv", header=True)
event_logs_df = spark.read.csv("event_logs.csv", header=True)

# 1. Remove Null Columns

def remove_null_columns(source_df):
    # Select columns that are not completely null
    non_null_columns = [
        column
        for column in source_df.columns
        if source_df.select(column).dropna().count() > 0
    ]
    # Return DataFrame with non-null columns
    return source_df.select(*non_null_columns)

# Apply the function to your DataFrames
df1_clean = remove_null_columns(df1)
df2_clean = remove_null_columns(df2)
df3_clean = remove_null_columns(df3)

# 2. Get Tables and Column Names

def get_table_columns(table_name: str, source_df: DataFrame):
    # Create a list of tuples containing table name and column names
    columns_info = [(table_name, col) for col in source_df.columns]
    # Create a DataFrame from the list of tuples
    return source_df.sql_ctx.createDataFrame(
        columns_info, ["table_name", "column_name"]
    )

# Extract column information from each DataFrame
df1_columns = get_table_columns("df1_clean", df1_clean)
df2_columns = get_table_columns("df2_clean", df2_clean)
df3_columns = get_table_columns("df3_clean", df3_clean)

# Combine the column information
combined_columns_df = df1_columns.union(df2_columns).union(df3_columns)

# 3. Select Relevant Columns

def select_columns(source_df, columns):
    return source_df.select(*columns)

# Specify columns to keep for each DataFrame
columns_df1 = ["column_a", "column_b", "column_c"]
columns_df2 = ["column_d", "column_e", "column_f"]
columns_df3 = ["column_g", "column_h", "column_i"]

# Select relevant columns
df1_selected = select_columns(df1_clean, columns_df1)
df2_selected = select_columns(df2_clean, columns_df2)
df3_selected = select_columns(df3_clean, columns_df3)

# 4. Filter Operations

# Define codes to filter
codes_to_filter = ["code1", "code2"]

# Filter DataFrames based on specific columns
df1_filtered = df1_selected.filter(F.col("operation_code").isin(codes_to_filter))
df2_filtered = df2_selected.filter(F.col("operation_code").isin(codes_to_filter))
df3_filtered = df3_selected.filter(F.col("operation_code").isin(codes_to_filter))

# 5. Create Auxiliary Table for Target Events

def create_target_events_table(event_logs):
    # Define conditions for identifying target events
    event_condition = (F.col("Table_Name") == "TableA") & (
        (F.col("Field_Name") == "TargetEvent")
        & (F.col("New_Value").isNotNull())
        & (F.col("New_Value") != "")
    )
    resolve_condition = (F.col("Table_Name") == "TableA") & (
        (F.col("Field_Name") == "TargetEvent")
        & (F.col("Old_Value").isNotNull())
        & (F.col("Old_Value") != "")
        & (
            F.col("New_Value").isNull()
            | (F.col("New_Value") == "")
        )
    )
    # Identify target events and resolutions
    event_df = event_logs.filter(event_condition).select(
        F.col("Document").alias("id"),
        F.col("Change_Date").alias("event_date"),
        F.lit("event").alias("event_type"),
    )
    resolve_df = event_logs.filter(resolve_condition).select(
        F.col("Document").alias("id"),
        F.col("Change_Date").alias("resolve_date"),
        F.lit("resolve").alias("event_type"),
    )
    # Combine and aggregate data
    combined_events_df = event_df.union(resolve_df)
    result_df = combined_events_df.groupBy("id").agg(
        F.min("event_date").alias("first_event_date"),
        F.max("resolve_date").alias("last_resolve_date"),
        F.count(F.when(F.col("event_type") == "event", True)).alias("event_count"),
    )
    return result_df

# Apply the function to your event logs DataFrame
target_events_table = create_target_events_table(event_logs_df)

# 6. Create Details Table

def create_details(df_main, df_auxiliary, target_events_df):
    # Join DataFrames on id
    details_df = df_main.join(
        df_auxiliary, df_main.id == df_auxiliary.id, "inner"
    ).select(
        df_main["*"],
        df_auxiliary["additional_column"],
    )
    # Add event information
    details_df = details_df.join(
        target_events_df, "id", "left"
    )
    # Add event flag
    details_df = details_df.withColumn(
        "event_flag",
        F.when(F.col("first_event_date").isNotNull(), 1).otherwise(0),
    )
    return details_df

# Create the details table
details = create_details(df1_filtered, df3_filtered, target_events_table)

# 7. Preprocess Training and Prediction Data

def preprocess_data(df):
    # Handle categorical variables
    categorical_columns = ["category_a", "category_b"]
    stages = []
    for column in categorical_columns:
        indexer = StringIndexer(
            inputCol=column, outputCol=f"{column}_indexed", handleInvalid="keep"
        )
        encoder = OneHotEncoder(
            inputCol=f"{column}_indexed", outputCol=f"{column}_encoded"
        )
        stages += [indexer, encoder]
    # Scale numerical features
    numerical_column = "numeric_feature"
    assembler = VectorAssembler(
        inputCols=[numerical_column], outputCol=f"{numerical_column}_vector"
    )
    scaler = StandardScaler(
        inputCol=f"{numerical_column}_vector",
        outputCol=f"scaled_{numerical_column}",
        withStd=True,
        withMean=True,
    )
    stages += [assembler, scaler]
    # Create and apply the pipeline
    pipeline = Pipeline(stages=stages)
    df = pipeline.fit(df).transform(df)
    # Extract scaled values
    df = df.withColumn(
        f"scaled_{numerical_column}",
        vector_to_array(F.col(f"scaled_{numerical_column}"))
        .getItem(0)
        .cast(DoubleType()),
    )
    # Drop intermediate columns
    columns_to_drop = categorical_columns + [f"{col}_indexed" for col in categorical_columns] + [f"{numerical_column}_vector"]
    df = df.drop(*columns_to_drop)
    return df

# Apply preprocessing to your data
preprocessed_df = preprocess_data(details)

# Split into training and prediction datasets based on date
training_data = preprocessed_df.filter(F.col("record_date") < F.lit("2022-01-01"))
prediction_data = preprocessed_df.filter(F.col("record_date") >= F.lit("2022-01-01"))

# 8. Calculate Match Percentages for Tables

def compute_match_percentages(df_a, df_b, key):
    total_count = df_a.count()
    match_count = df_a.join(df_b, key, "inner").count()
    match_percentage = (match_count / total_count) * 100
    return match_percentage

# Calculate match percentages between different tables
percentage_ab = compute_match_percentages(df1_filtered, df2_filtered, "id")
percentage_ac = compute_match_percentages(df1_filtered, df3_filtered, "id")

# 9. Prepare Data for Target Event Prediction Models

def prepare_target_event_data(df):
    # Create target variable for event prediction
    df = df.withColumn(
        "target_event_flag",
        F.when(
            (F.col("event_flag") == 1) & (F.col("status") != "Inactive"),
            1
        ).otherwise(0),
    )
    # Apply preprocessing steps similar to previous preprocessing function
    df = preprocess_data(df)
    return df

# Prepare data for target event prediction models
target_event_data = prepare_target_event_data(details)

# Split into training and validation datasets
train_target_event_data, val_target_event_data = target_event_data.randomSplit([0.8, 0.2], seed=42)
