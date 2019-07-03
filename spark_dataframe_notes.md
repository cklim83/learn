Basic DataFrame Operations
==========================
Last Edited: 12 June 19

### DataFrame Overview
- A DataFrame consists of a series of *Row* records and a number of *columns* that represent a computation expression that can be performed on each individual record in the Dataset.
- Partitioning defines the physical distribution of records across cluster as specified in the *partitioning schema*.

### Schemas
- Schema can be inferred, aka *schema-on-read*, or manually defined.
- **Schema-on-read** is usually fine for **ad-hoc analysis** but could be slow with plain-text formats like csv or json and could run into precision issues e.g. long type incorrectly interpreted as integer
- **Production ETL** is best served with **manual schema specification**.

~~~ Python
df = spark.read.format('json').load("/data/flight-data/json/2015-summary.json")
print(df.schema)
# StructType(List(StructField(DEST_COUNTRY_NAME,StringType,true),
# StructField(ORIGIN_COUNTRY_NAME,StringType,true),
# StructField(count,LongType,true)))
~~~

- A schema is StructType holding a List of StructFields. Each StructField contains a column name, a Spark datatype, and a boolean field to indicate if that column can contain null values. Users can *optionally* specify metadata to associate with column.
- If the types in the data (at runtime) do not match the schema, Spark will throw an error.

~~~ Python
# Manual Schema Definition
from pyspark.sql.types import StructType, StructField, StringType, LongType

my_schema = StructType([StructField('DEST_COUNTRY_NAME', StringType(), True),
      StructField('ORIGIN_COUNTRY_NAME', StringType(), True),
      StructField('count', LongType(), False), metadata={"my_key": "my_value"}])   
~~~

### Columns and Expressions
- We construct and refer to columns using _**col**_ or _**column**_ function.
- Columns are **not resolved** until its is compared to column names in *catalog* during *analyzer* phase. Hence we will not immediately get an error if column does not exist.
~~~ Python
from pyspark.sql.functions import col, column
col("myColumn") # Creating myColumn which is not resolved yet.
df.col("count") # referring to count column in df
~~~

- Transformations(select, manipulate, remove) on columns are termed _**expressions**_.
- Expressions can be created using the _**expr**_ function. The simplest expression is a column reference i.e. expr("someCol") is equivalent to col("someCol")
- expr function can **parse transformations and column references from a string** and its result could be passed into further transformation
  - **expr("someCol - 5")**, **col("someCol") - 5** and **expr("someCol") - 5** all compile to same logical tree
  ~~~ Python
  from pyspark.sql.functions import expr, col
  (((col("someCol") + 5) * 200) - 6) < col("otherCol") # Dataframe code
  expr("(((someCol + 5) * 200) -6) < otherCol") # SQL code using expr
  ~~~

  - This implies both SQL code and DataFrame code both compile to the same logical tree before execution, hence they have same performance

- You can access the list of columns of a DataFrame through its columns attribute
~~~ Python
df.columns
~~~

#### Records and Rows
- Each row in a DataFrame represents a single record
- Spark manipulates **Row** objects *using column expressions* to produce usable values.

  ##### Creating Rows
  ~~~ Python
  from pyspark.sql import Row
  myRow = Row("Hello", None, 1, False) # Creating a Row
  myRow[0] # access row values
  ~~~
  - Unlike DataFrame, Rows have no schema. To ensure Rows can be combined to form a DataFrame, the values of each record must be entered in the same order.

### DataFrame Transformations
- Core transformations are:
  - add rows or columns
  - remove rows or columns
  - transform row into column (or vice versa) i.e. pivot
  - Change order of rows based on values in column i.e. sort

#### Creating DataFrames
- from raw data sources
~~~ Python
df = spark.read.format("json").load("/data/flight-data/json/2015-summary.json")
~~~

- from rdd or pandas using SparkSession.createDataFrame()

  ~~~ python
  rdd = sc.fromText('mtcars.csv')
  df = spark.createDataFrame(rdd)

  import pandas as pd
  pandas_df = pd.DataFrame({'Name': ['John', 'David'], 'Age':[23, 25]})
  df = spark.createDataFrame(pandas_df)
  ~~~

- programmatically
~~~ Python
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType, LongType
mySchema = StructType([StructField("name", StringType(), True),
      StructField("nationality", StringType(), True),
      StructField("age", LongType(), False)])
data = [Row("John", null, 23), Row("David", "USA", 25)]
df = spark.createDataFrame(data, mySchema)
df.show()
~~~

#### select and selectExpr
- select() accepts string(s) representing column names (not in list)
~~~ Python
df.select("DEST_COUNTRY_NAME", "ORIGIN_COUNTY_NAME").show()
~~~
- We can also use **column references** in select()
~~~ Python
from pyspark.sql.functions import expr, col, column
df.select(expr("DEST_COUNTRY_NAME"), col("DEST_COUNTRY_NAME"),
      column("DEST_COUNTRY_NAME")).show()
~~~

- We can now also mix **Column reference/object** with strings
~~~ Python
df.select(col("DEST_COUNTRY_NAME"), "DEST_COUNTRY_NAME").show()
~~~

- **expr** is the most flexible reference.
~~~ Python
df.select(expr("DEST_COUNTRY_NAME AS destination")).show(2)
# SQL Equivalent
SELECT DEST_COUNTRY_NAME as destination FROM dfTable LIMIT 2
~~~
- **selectExpr**("Expression_1,...,Expression_n") allows us to insert a **series of expressions**, including **aggregations** over entire DataFrame, similar to those in SQL after the Select clause

~~~ Python
df.selectExpr("*", "(DEST_COUNTRY_NAME = ORIGIN_COUNTRY_NAME) as  
      withinCountry").show()
# SQL Equivalent
SELECT *, (DEST_COUNTRY_NAME = ORIGIN_COUNTRY_NAME) as withinCountry
FROM dfTable
LIMIT 2
Output:
+-----------------+-------------------+-----+-------------+
|DEST_COUNTRY_NAME|ORIGIN_COUNTRY_NAME|count|withinCountry|
+-----------------+-------------------+-----+-------------+
| United States   | Romania           | 15  | false       |
| United States   | Croatia           | 1   | false       |
+-----------------+-------------------+-----+-------------+

df.selectExpr("avg(count)", "count(distinct(DEST_COUNTRY_NAME))").show()
# SQL equivalent
SELECT AVG(count), COUNT(DISTINCT(DEST_COUNTRY_NAME))
FROM dfTable
Output:
+-----------+---------------------------------+
| avg(count)|count(DISTINCT DEST_COUNTRY_NAME)|
+-----------+---------------------------------+
|1770.765625| 132                             |
+-----------+---------------------------------+
~~~

#### Spark Literals
- We can insert explicit values as a *column* using lit(value).

~~~ Python
from pyspark.sql.functions import lit
df.select(expr("*"), lit(1).alias('One')).show(2)
# SQL Equivalent
SELECT *, 1 FROM dfTables LIMIT 2;
Output:
+-----------------+-------------------+-----+---+
|DEST_COUNTRY_NAME|ORIGIN_COUNTRY_NAME|count|One|
+-----------------+-------------------+-----+---+
| United States   | Romania           | 15  | 1 |
| United States   | Croatia           | 1   | 1 |
+-----------------+-------------------+-----+---+
~~~

#### Adding Columns
- We can add a column to our DataFrame using **withColumn(colName, col_expr)**
~~~ Python
df.withColumn("numOne", lit(1)).show(2)
~~~

#### Renaming Columns
- We can rename a column using **withColumnRenamed("orig_colName", "new_colName")**
~~~ python
df.withColumnRenamed("DEST_COUNTRY_NAME", "dest").columns
Output: dest, ORIGIN_COUNTRY_NAME, count
# We can also rename a column using withColumn at expense of creating new column
df.withColumn("dest", expr("DEST_COUNTRY_NAME")).columns
Output: DEST_COUNTRY_NAME, ORIGIN_COUNTRY_NAME, count, dest
~~~

#### Reserved Characters and Keywords
- Space and - are reserved keywords. For column references, any column name with these characters should be escape using backticks (`) else we will get a compilation error.
~~~ Python
dfWithLongColName.selectExpr(
"`This Long Column-Name`",
"`This Long Column-Name` as `new col`")\
.show(2)
~~~

#### Case Sensitivity
- Spark is case insensitive by default. To make it case sensitive, we can set the configuration
~~~ Python
set spark.sql.caseSensitive true
~~~

#### Removing Columns
- Besides using select, we can also remove columns using
**drop("colName_1", "colName_2")**
~~~ python
df.drop("ORIGIN_COUNTRY_NAME", "DEST_COUNTRY_NAME").columns
~~~

#### Changing Column's Type (cast)
- we can use the cast operator to change column type
~~~ Python
# Assume count is captured as string and we cast it to long
df.withColumn("count2", col("count").cast("long"))
# SQL Equivalent
SELECT *, CAST(count as long) AS count2 FROM dfTable
~~~

#### Filtering Rows
- To filter rows, we create expressions that evaluates to true or false and pass them as arguments to **filter** or **where** functions. **where is preferred** as it is SQL like.
~~~ Python
df.filter(col("count") < 2).show()
df.where("count < 2").show() # both are equivalent
# SQL equivalent
SELECT * FROM dfTable WHERE count < 2 LIMIT 2
Output:
+-----------------+-------------------+-----+
|DEST_COUNTRY_NAME|ORIGIN_COUNTRY_NAME|count|
+-----------------+-------------------+-----+
| United States   | Croatia           | 1   |
| United States   | Singapore         | 1   |
+-----------------+-------------------+-----+
~~~

- Although we can use multiple filters in same expression using AND, it is not helpful performance wise as Spark disregard their order and filter all simultaneously. Hence for multiple filters, we could just chain them sequentially and Spark will handle.
~~~ Python
df.where(col("count") < 2).where(col("ORIGIN_COUNTRY_NAME")!="Croatia").show()
# SQL equivalent
SELECT * FROM dfTable WHERE count < 2 AND ORIGIN_COUNTRY_NAME != "Croatia" LIMIT 2
Output:
+-----------------+-------------------+-----+
|DEST_COUNTRY_NAME|ORIGIN_COUNTRY_NAME|count|
+-----------------+-------------------+-----+
| United States   | Singapore         | 1   |
| Moldova         | United States     | 1   |
+-----------------+-------------------+-----+
~~~

#### Getting Unique Values/Counts
- **distinct()** is to de-duplicate any rows in a DataFrame.
~~~ Python
df.select("ORIGIN_COUNTRY_NAME").distinct().count() # Get num distinct entries
df.select"ORIGIN_COUNTRY_NAME").distinct() # Get distinct rows.
~~~

#### Getting Random Samples
- **sample(withReplacement, fraction, seed)** allows us to draw random samples from a DataFrame
~~~ Python
seed = 555
withReplacement = False
fraction = 0.5
df.sample(withReplacement, fraction, seed)
~~~
- **randomSplit([prop_1,...,prop_n], seed)** allows us to create randomized splits used to create training and validation sets in machine learning. Sum of proportions should sum to 1 or they will be normalized.
~~~ python
total_count = df.count()
print("df count:", total_count)
dataFrames = df.randomSplit([0.25, 0.25, 0.5], seed)
split_1_count = dataFrames[0].count()
split_2_count = dataFrames[1].count()
split_3_count = dataFrames[2].count()
print("Split_1 - count:{}, proportion: {}".format(split_1_count, split_1_count/total_count))
print("Split_2 - count:{}, proportion: {}".format(split_2_count, split_2_count/total_count))
print("Split_3 - count:{}, proportion: {}".format(split_3_count, split_3_count/total_count))
Output:
df count: 256
Split_1 - count:60, proportion: 0.234375
Split_2 - count:66, proportion: 0.2578125
Split_3 - count:130, proportion: 0.5078125
~~~

#### Concatenating and Appending Rows (Union)
- union(newDF) returns a new DataFrame with the rows concatenated.
~~~ Python
from pyspark.sql import Row
schema = df.schema
newRows = [Row("New Country", "Other Country", 50000),
Row("New Country 2", "Other Country 3", 10000)]
rdd = spark.sparkContext.parallelize(newRows)
newDF = spark.createDataFrame(rdd, schema)
df.union(newDF).where("count > 5000").show()
Output:
+-----------------+-------------------+------+
|DEST_COUNTRY_NAME|ORIGIN_COUNTRY_NAME| count|
+-----------------+-------------------+------+
|           Mexico|      United States|  7140|
|    United States|      United States|370002|
|           Canada|      United States|  8399|
|    United States|             Mexico|  7187|
|    United States|             Canada|  8483|
|      New Country|      Other Country| 50000|
|    New Country 2|    Other Country 3| 10000|
+-----------------+-------------------+------+
df.union(newDF).where("DEST_COUNTRY_NAME LIKE '%New Country%'").show()
Output:
+-----------------+-------------------+-----+
|DEST_COUNTRY_NAME|ORIGIN_COUNTRY_NAME|count|
+-----------------+-------------------+-----+
|      New Country|      Other Country|50000|
|    New Country 2|    Other Country 3|10000|
+-----------------+-------------------+-----+
~~~
- Make sure the schema and column orders match up as spark will not auto-arrange or raise an error. See example below.
~~~ Python
# CK check if order is different
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, LongType, StringType
newRows = [
  Row(50000, "New Country", "Other Country"),
  Row(10000, "New Country 2", "Other Country 3")
]
rdd = spark.sparkContext.parallelize(newRows)
schema = StructType([StructField("count", LongType(), False),
                     StructField("ORIGIN_COUNTRY_NAME", StringType(), True),
                     StructField("DEST_COUNTRY_NAME", StringType(), True)])
newDF = spark.createDataFrame(rdd, schema)
newDF.union(df).show(5)
Output:
+-------------+-------------------+-----------------+
|        count|ORIGIN_COUNTRY_NAME|DEST_COUNTRY_NAME|
+-------------+-------------------+-----------------+
|        50000|        New Country|    Other Country|
|        10000|      New Country 2|  Other Country 3|
|United States|            Romania|               15|
|United States|            Croatia|                1|
|United States|            Ireland|              344|
+-------------+-------------------+-----------------+
~~~

#### Sorting Rows
- **sort()** and **orderBy()** accepts strings, column references or expressions and sort by ascending order in default.
- we can chain **desc()** and **asc()** from sql.functions _**to column references**_ to be explict on descending/ascending sort order.
- Advanced: use **asc_nulls_first(), asc_nulls_last(), desc_nulls_first() and desc_nulls_last()** to control where to place null entries.
- Optimisation: It is sometime beneficial to sort within each partition first before another set of transformations. **sortWithinPartitions()** is used for that.
~~~ Python
df.sort("count", "ORIGIN_COUNTRY_NAME").show(3)
df.orderBy("count", "DEST_COUNTRY_NAME").show(3)
df.orderBy(expr("count + 1"), col("DEST_COUNTRY_NAME")).show(3)
# Explicit Sort
from pyspark.sql.functions import desc, asc
df.orderBy(expr("count desc")).show(2)
df.orderBy(col("count").desc(), col("DEST_COUNTRY_NAME").asc()).show(2)
# SQL Equivalent
SELECT * FROM dfTable ORDER BY count DESC, DEST_COUNTRY_NAME ASC LIMIT 2
# Sort Wihin Partition
spark.read.format("json").load("/data/flight-data/json/*-summary.json")\
      .sortWithinPartitions("count")
~~~

#### Limit
- use **limit(n)** to restrict number of rows extracted from a DataFrame.
~~~ Python
df.limit(5).show()
~~~

#### Repartition and Coalesce
- **repartition(), rdd.getNumPartitions()**
- optimize by partitioning data along frequently filtered column
- Repartition incur full data shuffle regardless whether one is required. Repartition only when future partitions is greater than current partition count or when looking to partition by a set of columns.
~~~ Python
df.repartition(5) # break into 5 partitions
df.repartition(col("DEST_COUNTRY_NAME")) # by frequently filtered column
df.repartition(5, col("DEST_COUNTRY_NAME"))
df.rdd.getNumPartitions() # 5
~~~

- **coalesce()** used to combine partitions and do not incur full shuffle.
~~~ Python
df.repartition(5, col("DEST_COUNTRY_NAME")).coalesce(2)
~~~

#### Collect Rows to Driver
- We usually want to collect transformed data to driver for local usage. Note data collected can crash driver and cause application state loss if too big.
- **collect()** gets data from entire DataFrame to spark driver
- **take(n)** selects first N rows to driver
- **show(n)** displays first N rows
- Optional: **toLocalIterator()** collects partitions to the driver as iterator, allows you to iterate over entire dataset partition-by-partition serially.


### Working with Different Types of Data

#### Where to Look for APIs
- PySpark SQL functions are available [here](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html)

#### Converting to Spark Types
- **lit()** is used to convert native language type (e.g. python) to Spark types.
~~~python
from pyspark.sql.functions import lit
df.select(lit(5), lit("five"), lit(5.0))
# SQL Equivalent
SEL 5, 'five', 5.0
~~~

#### Working with Booleans
- We build boolean expressions i.e. those that evaluates to True or False for filtering.
~~~ Python
from pyspark.sql.functions import col
df.where(col("InvoiceNo") != 536365) \
      .select("InvoiceNo", "Description") \
      .show(5, False)
# Using expressions in the form of a String
df.where("InvoiceNo != 536365")
~~~
- For compound boolean expressions involving **and** and **or**, spark has a preferred usage.
- **and** is best written as a chained sequence of individual **where** operations as all where operations are conducted in parallel in spark and the order does not matter.
- **or** is best used between two expressions
~~~ Python
from pyspark.sql.functions import instr
priceFilter = col("UnitPrice") > 600
# instr(str, substr) returns the index(1-based) of the first occurrence of substr in str
descripFilter = instr(df.Description, "POSTAGE") >= 1 # POSTAGE in Description
df.where(df.StockCode.isin("DOT")).where(priceFilter | descripFilter).show()
~~~
- We can also added boolean result to a column and use **where** on that column.
~~~ python
from pyspark.sql.functions import instr
DOTCodeFilter = col("StockCode") == "DOT"
priceFilter = col("UnitPrice") > 600
descripFilter = instr(col("Description"), "POSTAGE") >= 1
df.withColumn("isExpensive", DOTCodeFilter & (priceFilter | descripFilter))\
.where("isExpensive")\
.select("unitPrice", "isExpensive").show(5)
~~~
- WARNING: to perform **null safe comparison**, we could do the following.
~~~ Python
df.where(col("Description").eqNullSafe("hello")).show()
~~~

#### Working with Numbers
- Addition, Mulitplication, Power in expressions
~~~ python
from pyspark.sql.functions import expr, pow
fabricatedQuantity = pow(col("Quantity") * col("UnitPrice"), 2) + 5
df.select(expr("CustomerId"), fabricatedQuantity.alias("realQuantity")).show(2)
# Equivalent Using selectExpr
df.selectExpr(
"CustomerId",
"(POWER((Quantity * UnitPrice), 2.0) + 5) as realQuantity").show(2)
# SQL Equivalent
SELECT customerId, (POWER((Quantity * UnitPrice), 2.0) + 5) as realQuantity
FROM dfTable
~~~
- Rounding using **round(num, precision)** and **bround(num, precision)** [round down]
~~~ Python
from pyspark.sql.functions import lit, round, bround
df.select(round(lit("2.5"),1)), bround(lit("2.5")), 1).show(1) # to 1 decimal
# SQL Equivalent
SELECT round(2.5), bround(2.5)
Output:
+-------------+--------------+
|round(2.5, 0)|bround(2.5, 0)|
+-------------+--------------+
| 3.0         | 2.0          |
+-------------+--------------+
~~~

- **corr("col1", "col2")** to Compute correlation of two Columns
~~~ Python
from pyspark.sql.functions import corr
df.stat.corr("Quantity", "UnitPrice")
df.select(corr("Quantity", "UnitPrice")).show()
# SQL equivalent
SELECT corr(Quantity, UnitPrice) FROM dfTable
~~~
- **describe()** for summary stats of numeric columns
- **count(), mean(), stddev_pop(), min(), max()** are other commonly used numeric functions available under pyspark.sql.functions
- DataFrame also have some numeric methods such as **approxQuantitle(), crosstab(), freqItems()** in a stat module.
~~~ Python
colName = "UnitPrice"
quantileProbs = [0.5]
relError = 0.05
df.stat.approxQuantile("UnitPrice", quantileProbs, relError) # 2.51
df.stat.crosstab("StockCode", "Quantity").show()
df.stat.freqItems(["StockCode", "Quantity"]).show()
~~~

- We can use **monotonically_increasing_id()** under pyspark.sql.functions to generate index starting from 0. They are guaranteed to be monotonically increasing, unique but not neccessary consecutive.
~~~ Python
from pyspark.sql.functions import monotonically_increasing_id
df.select(monotonically_increasing_id(), *).show(2)


#### Working with Strings
- **initcap()** will capitalize every word in a string when word is separated by space.
~~~ python
from pyspark.sql.functions import initcap
df.select(initcap(col("Description"))).show()
-- in SQL
SELECT initcap(Description) FROM dfTable
Output:
+----------------------------------+
|initcap(Description) |
+----------------------------------+
|White Hanging Heart T-light Holder|
|White Metal Lantern |
+----------------------------------+
~~~

- **upper()** and **lower()** to case string to fully upper or lower cases.
~~~ Python
from pyspark.sql.functions import lower, upper
df.select(col("Description"),
      lower(col("Description")),
      upper(lower(col("Description")))).show(2)
Output:
+--------------------+--------------------+-------------------------+
| Description        | lower(Description) |upper(lower(Description))|
+--------------------+--------------------+-------------------------+
|WHITE HANGING HEA...|white hanging hea...| WHITE HANGING HEA...    |
| WHITE METAL LANTERN| white metal lantern| WHITE METAL LANTERN     |
+--------------------+--------------------+-------------------------+
~~~

- **lpad(), ltrim(), rpad(), rtrim() and trim()** availing to add or remove space around as string.
- if lpad or rpad takes a number less than string length, it will remove values from the right side of string.
~~~ Python
from pyspark.sql.functions import lit, ltrim, rtrim, rpad, lpad, trim
df.select(
      ltrim(lit(" HELLO ")).alias("ltrim"),
      rtrim(lit(" HELLO ")).alias("rtrim"),
      trim(lit(" HELLO ")).alias("trim"),
      lpad(lit("HELLO"), 3, " ").alias("lp"), # number less than string length
      rpad(lit("HELLO"), 10, " ").alias("rp")).show(2)
# SQL Equivalent
SELECT
    ltrim(' HELLLOOOO '),
    rtrim(' HELLLOOOO '),
    trim(' HELLLOOOO '),
    lpad('HELLOOOO ', 3, ' '),
    rpad('HELLOOOO ', 10, ' ')
FROM dfTable
Output:
+---------+---------+-----+---+----------+
| ltrim   | rtrim   | trim| lp| rp       |
+---------+---------+-----+---+----------+
|HELLO    | HELLO   |HELLO| HE|HELLO     |
|HELLO    | HELLO   |HELLO| HE|HELLO     |
+---------+---------+-----+---+----------+
~~~

- Regular Expressions
  - Uses Java regular expressions.
  - **regexp_extract()** and **regex_replace()** extract and replaces values respectively

~~~ Python
# in Python
from pyspark.sql.functions import regexp_replace
regex_string = "BLACK|WHITE|RED|GREEN|BLUE"
df.select(
    regexp_replace(col("Description"), regex_string,  
      "COLOR").alias("color_clean"),
    col("Description")).show(2)
    -- in SQL
SELECT
    regexp_replace(Description, 'BLACK|WHITE|RED|GREEN|BLUE', 'COLOR') as
    color_clean, Description
FROM dfTable
Output:
+--------------------+--------------------+
| color_clean        | Description        |
+--------------------+--------------------+
|COLOR HANGING HEA...|WHITE HANGING HEA...|
| COLOR METAL LANTERN| WHITE METAL LANTERN|
+--------------------+--------------------+
~~~

- character level replacement using translate()
~~~ python
from pyspark.sql.functions import translate
df.select(translate(col("Description"), "LET", "137"),col("Description"))\
.show(2)
#  SQL Equivalent
SELECT translate(Description, 'LET', '137'), Description FROM dfTable
Output:
+----------------------------------+--------------------+
|translate(Description, LET, 137)  | Description        |
+----------------------------------+--------------------+
| WHI73 HANGING H3A...             |WHITE HANGING HEA...|
| WHI73 M37A1 1AN73RN              | WHITE METAL LANTERN|
+----------------------------------+--------------------+
~~~

- extract first occurrence using **regexp_extract()**
~~~ python
from pyspark.sql.functions import regexp_extract
extract_str = "(BLACK|WHITE|RED|GREEN|BLUE)"
df.select(
      regexp_extract(col("Description"), extract_str, 1).alias("color_clean"),
      col("Description")).show(2)
-- in SQL
SELECT regexp_extract(Description, '(BLACK|WHITE|RED|GREEN|BLUE)', 1),
Description
FROM dfTable
Output:
+-------------+--------------------+
| color_clean | Description        |
+-------------+--------------------+
| WHITE       |WHITE HANGING HEA...|
| WHITE       | WHITE METAL LANTERN|
+-------------+--------------------+
~~~

- check if substring exist in string
  - **instr()** or **locate()** are functions that act on columns and returns index of first occurrence of substring in string. Index starts with 1 and not 0. If not found, 0 is return. The values can be casted to boolean for use in row filtering.
  - both functions are identical except the argument order are reversed. instr(col_ref, substr) vs locate(substr, col_ref)

  ~~~ python
  from pyspark.sql.functions import instr
  containsBlack = instr(col("Description"), "BLACK") >= 1
  containsWhite = instr(col("Description"), "WHITE") >= 1
  df.withColumn("hasSimpleColor", containsBlack | containsWhite)\
        .where("hasSimpleColor")\
        .select("Description").show(3, False)
  -- in SQL
  SELECT Description FROM dfTable
  WHERE instr(Description, 'BLACK') >= 1 OR instr(Description, 'WHITE') >= 1
  +----------------------------------+
  |Description                       |
  +----------------------------------+
  |WHITE HANGING HEART T-LIGHT HOLDER|
  |WHITE METAL LANTERN               |
  |RED WOOLLY HOTTIE WHITE HEART.    |
  +----------------------------------+
  ~~~

  - check presence of a variable list of substring by generating is_substringN columns programmatically that could be used for filtering.

  ~~~ python
  from pyspark.sql.functions import expr, locate
  simpleColors = ["black", "white", "red", "green", "blue"]
  def color_locator(column, color_string):
    return locate(color_string.upper(), column)\
          .cast("boolean")\
          .alias("is_" + c)

  selectedColumns = [color_locator(df.Description, c) for c in simpleColors]
  selectedColumns.append(expr("*")) # has to a be Column type
  df.select(*selectedColumns).where(expr("is_white OR is_red"))\
        .select("Description").show(3, False)
  ~~~

#### Working with Dates and Timestamps
- Spark supports Date and Timestamp types. We can access the date attribute from a timestamp using **.date** attribute.
- **TimestampType** support up to seconds precision. To support sub-second precision, we will need to represent and operate them as **longs**
- **current_date()** and **current_timestamp()** to get current UTC date and timestamp.
~~~ python
from pyspark.sql.functions import current_date, current_timestamp
dateDF = spark.range(10)\
      .withColumn("today", current_date())\
      .withColumn("now", current_timestamp())
dateDF.createOrReplaceTempView("dateTable")
dateDF.show(2, truncate=False)
Output:
+---+----------+----------------------+
|id |today     |now                   |
+---+----------+----------------------+
|0  |2019-06-14|2019-06-14 06:30:21.12|
|1  |2019-06-14|2019-06-14 06:30:21.12|
+---+----------+----------------------+
dateDF.printSchema()
root
|-- id: long (nullable = false)
|-- today: date (nullable = false)
|-- now: timestamp (nullable = false)
~~~

- date arithmetic
  - date_add(), date_sub() supports date addition and subtraction

~~~ python
from pyspark.sql.functions import date_add, date_sub
dateDF.select(date_sub(col("today"), 5),
      date_add(col("today"), 5)).show(1)
Output:
+------------------+------------------+
|date_sub(today, 5)|date_add(today, 5)|
+------------------+------------------+
|        2019-06-09|        2019-06-19|
+------------------+------------------+
~~~

- intervals between two dates
  - **datediff()** to find number of days between two dates
  - **months_between()** to find number of months between two dates

~~~ python
from pyspark.sql.functions import datediff, months_between, to_date
dateDF.withColumn("week_ago", date_sub(col("today"), 7))\
.select(datediff(col("week_ago"), col("today"))).show(1)
Output:
+-------------------------+
|datediff(week_ago, today)|
+-------------------------+
| -7                      |
+-------------------------+
dateDF.select(
to_date(lit("2016-01-01")).alias("start"),
to_date(lit("2017-05-22")).alias("end"))\
.select(months_between(col("start"), col("end"))).show(1)
Output:
+--------------------------+
|months_between(start, end)|
+--------------------------+
| -16.67741935             |
+--------------------------+
~~~

- convert string to date using **to_date()**. It optionally accepts a format in Java SimpleDateFormat. Note: spark will return null instead of raising an error if it cant parse a date (see below).

~~~ python
from pyspark.sql.functions import to_date, lit
dateDF.select(to_date(lit("2016-20-12")),to_date(lit("2017-12-11"))).show(1)
Output:
+-------------------+-------------------+
|to_date(2016-20-12)|to_date(2017-12-11)|
+-------------------+-------------------+
| null              | 2017-12-11        |
+-------------------+-------------------+
~~~

- to avoid these these issues, it is best to supply the format for the string.
- format are optional for to_date() but compulsory for to_timestamp().
~~~ python
from pyspark.sql.functions import to_date
dateFormat = "yyyy-dd-MM"
cleanDateDF = spark.range(1).select(
to_date(lit("2017-12-11"), dateFormat).alias("date"),
to_date(lit("2017-20-12"), dateFormat).alias("date2"))
cleanDateDF.createOrReplaceTempView("dateTable2")
Output:
+----------+----------+
| date     | date2    |
+----------+----------+
|2017-11-12|2017-12-20|
+----------+----------+
from pyspark.sql.functions import to_timestamp
cleanDateDF.select(to_timestamp(col("date"), dateFormat)).show()
+----------------------------------+
|to_timestamp(`date`, 'yyyy-dd-MM')|
+----------------------------------+
| 2017-11-12 00:00:00              |
+----------------------------------+
~~~
- casting between dates and timestamps
-- in SQL
SELECT cast(to_date("2017-01-01", "yyyy-dd-MM") as timestamp)

- comparing dates
  - We just need to be sure to either use a date/timestamp type or specify our string according to the right format of yyyy-MM-dd if we’re comparing a date
~~~ python
cleanDateDF.filter(col("date2") > lit("2017-12-12")).show()
cleanDateDF.filter(col("date2") > "'2017-12-12'").show()
~~~

#### Working with Nulls in Data
- Missing or empty data should be represented with nulls than empty string as spark can optimise the former better.
- In schema definition, fields indicate as not nullable is not enforced. It is simply a flag for spark to optimise handling that column. Hence there could still be null values there.
- **coalesce()** allows us to return the first none null value from a set of columns. If there are no null values, it just returns the value from the first column.

~~~ python
temp = df.select(lit(None).alias("myNullCol"), "StockCode")
temp.show(3)
Output:
+---------+---------+
|myNullCol|StockCode|
+---------+---------+
|     null|   85123A|
|     null|    71053|
|     null|   84406B|
+---------+---------+
temp.select(coalesce("myNullCol", "StockCode")).show(3)
Output:
+------------------------------+
|coalesce(myNullCol, StockCode)|
+------------------------------+
|                        85123A|
|                         71053|
|                        84406B|
+------------------------------+
~~~
- Value selection in tuples (potentially with null)
  - **ifnull or nvl**: Select the second value if first is null. Return first value if that is not null.
  - **nullif**: returns null if two values are equal, returns second value otherwise.
  - **nvl2**: returns second value if first is **not null**, return last value otherwise

~~~ SQL
SELECT
  ifnull(null, 'return_value'),
  nullif('value', 'value'),
  nvl(null, 'return_value'),
  nvl2('not_null', 'return_value', "else_value")
FROM dfTable LIMIT 1
Output:
+------------+----+------------+------------+
| a          | b  | c          | d          |
+------------+----+------------+------------+
|return_value|null|return_value|return_value|
+------------+----+------------+------------+
~~~

- **drop() and fill()** in DataFrame.na
  - df.na.drop() to drop any rows with missing value
  - df.na.drop('all') to drop rows where all values in that row are null.
  - df.na.fill() to fill missing values
  - df.na.fill(value, subset=["col1", "col2"]) to fill missing rows in col1 and col2 with value
  - df.na.fill(dict) where dict key is the column name and dict value is full value for missing rows of that column.


- **replace()**
  - Used to target non missing rows too. Value must be same type as what it replaced.
~~~ python
df.na.replace([""], ['UNKNOWN'], 'Description')
~~~
- **Ordering with null values**:
  - use asc_nulls_first(), desc_nulls_first(), asc_nulls_last(),
  desc_nulls_last() to determine where nulls appear post sorting.


#### Complex Types
- Structs
  - Are DataFrames within DataFrames
  ~~~ python
  # Creating a struct called complex containing Description and InvoiceNo
  df.selectExpr("(Description, InvoiceNo) as complex")
  df.selectExpr("struct(Description, InvoiceNo) as complex")
  complexDF = df.select(struct("Description", "InvoiceNo").alias("complex"))
  complexDF.createOrReplaceTempView("complexDF")
  ~~~
  - Can access individual columns in struct using either **dot syntax or getField**
  ~~~ python
  complexDF.select("complex.Description")
  complexDF.select(col("complex").getField("Description"))
  ~~~
  - Can query all values in struct using \*. This brings all columns to the top-level DataFrame
  ~~~ python
  complexDF.select("complex.*")
  # SQL
  SELECT complex.* FROM complexDF
  ~~~

- Arrays
  - Collection of items similar to list
  ~~~ python
  from pyspark.sql.functions import split
  df.select(split(col("Description"), " ")).show(2)
  +---------------------+
  |split(Description, ) |
  +---------------------+
  | [WHITE, HANGING, ...|
  | [WHITE, METAL, LA...|
  +---------------------+
  df.select(split(col("Description")," ").alias("array_col"))\
      .selectExpr("array_col[0]").show(2)
  +------------+
  |array_col[0]|
  +------------+
  | WHITE      |
  | WHITE      |
  +------------+
  # SQL
  SELECT split(Description, ' ')[0] FROM dfTable
  ~~~

  - **size()** to query number of elements in array
  ~~~ python
  from pyspark.sql.functions import size
  df.select(size(split(col("Description"), " "))).show(2) # Returns 5 and 3 for 1st 2 rows.
  ~~~

  - array_contains() to check if array contains a value
  ~~~ python
  from pyspark.sql.functions import array_contains
  df.select(array_contains(split(col("Description"), " "), "WHITE")).show(2)
  # SQL
  SELECT array_contains(split("Description", ' '), 'WHITE') from dfTable
  Output:
  +--------------------------------------------+
  |array_contains(split(Description, ), WHITE) |
  +--------------------------------------------+
  | true                                       |
  | true                                       |
  +--------------------------------------------+
  ~~~

  - explode create one row for each element in array with values in other columns duplicated.
  ~~~ python
  from pyspark.sql.functions import split, explode
  df.withColumn("splitted", split(col("Description"), " "))\
      .withColumn("exploded", explode(col("splitted")))\
      .select("Description", "InvoiceNo", "exploded").show(2)
  # SQL
  SELECT Description, InvoiceNo, exploded
  FROM (SELECT *, split(Description, " ") as splitted FROM dfTable)
  LATERAL VIEW explode(splitted) as exploded
  Output:
  +--------------------+---------+--------+
  | Description        |InvoiceNo|exploded|
  +--------------------+---------+--------+
  |WHITE HANGING HEA...| 536365  | WHITE  |
  |WHITE HANGING HEA...| 536365  | HANGING|
  +--------------------+---------+--------+
  ~~~

- Maps
  - created using map function and key-value pairs of columns
  ~~~ python
  from pyspark.sql.functions import create_map
  df.select(create_map(col("Description"), col("InvoiceNo"))\
        .alias("complex_map")).show(2)
  # SQL
  SELECT map(Description, InvoiceNo) as complex_map from dfTable
  WHERE Description IS NOT NULL
  Output:
  +--------------------+
  | complex_map        |
  +--------------------+
  |Map(WHITE HANGING...|
  |Map(WHITE METAL L...|
  +--------------------+
  ~~~

  - Can query a map with a key
  ~~~ python
  df.select(map(col("Description"), col("InvoiceNo")).alias("complex_map"))\
        .selectExpr("complex_map['WHITE METAL LANTERN']").show(2)
  Output:
  +--------------------------------+
  |complex_map[WHITE METAL LANTERN]|
  +--------------------------------+
  | null                           |
  | 536365                         |
  +--------------------------------+
  ~~~

  - Explode map to key and value columns
  ~~~ python
  df.select(map(col("Description"), col("InvoiceNo")).alias("complex_map"))\
        .selectExpr("explode(complex_map)").show(2)
  Output:
  +--------------------+------+
  | key                | value|
  +--------------------+------+
  |WHITE HANGING HEA...|536365|
  | WHITE METAL LANTERN|536365|
  +--------------------+------+
  ~~~

#### JSON
- extract json object from json formatted string.
~~~ python
jsonDF = spark.range(1).selectExpr("""
      '{"myJSONKey": {"myJSONValue": [1, 2, 3]}}' as jsonString""")
~~~

- **_get_json_object** allows inline query of JSON object, be it dictionary or
array
- **json_tuple** if object has only 1 level of nesting
~~~ python
from pyspark.sql.functions import get_json_object, json_tuple
jsonDF.select(
      get_json_object(col("jsonString"), "$.myJSONKey.myJSONValue[1]") as
      "column", json_tuple(col("jsonString"), "myJSONKey")).show(2)
# SQL
jsonDF.selectExpr(
      "json_tuple(jsonString, '$.myJSONKey.myJSONValue[1]') as column").show(2)
Output:
+------+--------------------+
|column| c0                 |
+------+--------------------+
| 2    |{"myJSONValue":[1...|
+------+--------------------+
~~~

- **to_json()** to convert StructType to JSON
~~~ python
from pyspark.sql.functions import to_json
df.selectExpr("(InvoiceNo, Description) as myStruct")\
      .select(to_json(col("myStruct")))
~~~

- from_json() to parse JSON objects. Would need a schema for parsing.
~~~ python
from pyspark.sql.functions import from_json
from pyspark.sql.types import *
parseSchema = StructType((
      StructField("InvoiceNo", StringType(), True),
      StructField("Description", StringType(), True)))
df.selectExpr("(InvoiceNo, Description) as myStruct")\
      .select(to_json(col("myStruct")).alias("newJSON"))\
      .select(from_json(col("newJSON"), parseSchema), col("newJSON")).show(2)
Output:
+----------------------+--------------------+
|jsontostructs(newJSON)| newJSON            |
+----------------------+--------------------+
| [536365,WHITE HAN... |{"InvoiceNo":"536...|
| [536365,WHITE MET... |{"InvoiceNo":"536...|
+----------------------+--------------------+
~~~

### User Defined Functions
- Custom transformations using various programming languages(Scale, Python,
Java) that operate on individual records. There are performance considerations.

~~~ python
# Scala
val udfExampleDF = spark.range(5).toDF("num")
def power3(number:Double):Double = number * number * number
power3(2.0)

# Python
udfExampleDF = spark.range(5).toDF("num")
def power3(double_value):
    return double_value ** 3

power3(2.0) # test functions behaves as expected
~~~

- After UDF creation, register it with Spark to allow them to be used on all
worker nodes. Spark will serialize the function on driver and transfer over
network to all executor processes.
- If function is written in Java or Scale, it can use directly in Java Virtual
Machine (JVM), hence no performance penalty except for the inability to use
code generation capabilities spark has for built-in functions. Could have
performance issues if used on lots of objects.
- For Python UDFs, spark will have to start a python process on worker
serialize all data to format (previously in JVM environment) so that python can
 understand, execute function row by row in Python process before returning
 results to JVM and Spark.
- The cost lies in serializing data to python. Also python process will
compete with JVM for memory on same machine, causing it to fail if resource
constrained.
- Recommend to write all UDFs in Scale or Java to avoid this deterioration.

~~~ python
# Scala
import org.apache.spark.sql.functions.udf
val power3udf = udf(power3(_:Double):Double) #register udf with spark
udfExampleDF.select(power3udf(col("num"))).show()

# Python
from pyspark.sql.functions import udf, col
power3udf = udf(power3)
udfExampleDF.select(power3udf(col("num"))).show()

Output:
+-----------+
|power3(num)|
+-----------+
| 0         |
| 1         |
+-----------+
~~~
- Above method only allow UDF to be used as DataFrame function (i.e. can only
be used on column expression and not on strings). To use them in SQL, we need
to register it with Spark SQL. We can register in either scala, but that will
also make function available in SQL in python.

~~~ python
# Register Scala
spark.udf.register("power3", power3(_:Double):Double)
udfExampleDF.selectExpr("power3(num)").show(2)
# Can use power3 register in scala on any python expressions too
udfExampleDF.selectExpr("power3(num)").show(2)

from pyspark.sql.types import IntegerType, DoubleType
spark.udf.register("power3py", power3, DoubleType()) # Registered via Python
# in Python
udfExampleDF.selectExpr("power3py(num)").show(2)
# power3py also available in any scala segment.
~~~

- Best practice to specify return type, even though its optional, to ensure
functions are working. This is because Spark type is not exactly the same as
python type.
- If type of return value doesnt match specified, Spark **will not throw error
but will simply return null**.
~~~ python
# in Python
from pyspark.sql.types import IntegerType, DoubleType
spark.udf.register("power3py", power3, DoubleType())
# in Python
udfExampleDF.selectExpr("power3py(num)").show(2)
# registered via Python
-- in SQL
SELECT power3(12), power3py(12) -- doesn't work because of return type
~~~
- In above example, num form from range() are integers.
power of an integer results in another integer in python. Hence specifying
Spark's DoubleType (equivalent to float in Python) results in null due to type
mismatch. Can remedy this to ensure python function returns a float instead of
integer.
- To optionally return a value from UDF, use **None (python) or Option
(Scala)**
- Can also use UDFs in Hive syntax. To do that, enable Hive support via
SparkSession.builder().enableHiveSupport(), then register UDFs in SQL. This is
only supported with precompiled Scala and Java packages so need to specify them
 as dependency as follows:
 ~~~ SQL
 CREATE TEMPORARY FUNCTION myFunc AS 'com.organization.hive.udf.FunctionName'
 ~~~
 - Can make above function permanent by removing TEMPORARY keyword.

correctly.
<details>
  <Summary>This is my summary</Summary>
    Details here....
    More Details
</details>

<details>
  <summary><strong>2-This is my summary</summary>
    Hellow World here....
    More Details
</details>
