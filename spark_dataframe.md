**J**ust **E**nough PySpark DataFrame
=====================================
*Last Edited: v1.0 dated 11 June 2019*

## Table of Contents
- [Overview of PySpark Dataframe](#overview-of-pyspark-dataframe)
- [Libraries and Spark Instantiation](#libaries-and-spark-instantiation)
- [Creating DataFrames](#creating-dataframes)
- [Inspecting DataFrames](#inspecting-dataframes)
- [Handling Missing & Erroneous Values](#handling-missing-&-erroneous-values)
- [Selecting Data](#selecting-data)
- [Adding, Renaming & Dropping Columns](#adding,-renaming-&-dropping-columns)
- [Removing Duplicates](#removing-duplicates)
- [Grouping Data](#grouping-data)
- [Sorting Data](#sorting-data)
- [Repartitioning Data](#repartitioning-data)
- [Registering Views for SQL Type Queries](#registering-views-for-sql-type-queries)
- [Converting to Other Data Format](#converting-to-other-data-format)
- [Writing to Files](#writing-to-files)
- [Stopping Spark](#stop-spark)


#### Overview of PySpark DataFrame
Beginning version 2.0, Spark has moved to a DataFrame API, a higher level abstraction built on top of RDD, the underlying data structure in Spark. Although a Spark DataFrame shares great similarity to R or Pandas Dataframes, there are some fundamental differences:
- **Immutable in nature**: A dataframe once formed, cannot be changed. Every transformation to an existing dataframe actually returns a newly created one with changes incorporated.
- **Lazy Evaluation**: A transformation is not performed until an action is called.
- **Distributed**: Data are stored in distributed manner on multiple virtual compute units.

#### Libraries and Spark Instantiation
~~~ Python
from pyspark.sql import SparkSession, functions as F, Row
from pyspark.sql.types import *

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext # Get underlying spark context
~~~

#### Creating DataFrames
- Reading from Data Sources
~~~ Python
df_csv = spark.read.csv('/path/to/yourcsv.csv', header=True,
      inferSchema=True, sep=',', encoding='UTF-8')
df_par = spark.read.parquet('/path/to/parquet/folders')
~~~

- Reading From RDD/Pandas Dataframe using **SparkSession.createDataFrame()**
  ~~~ python
  rdd = sc.parallelize([Row(name='Tom', age=23),
        Row(name='David', age=40)])
  df_rdd = spark.createDataFrame(rdd)

  import pandas as pd
  pandasDF = pd.DataFrame({'name'= ['Tom', 'David'], 'age': [23, 40]})
  df_pd = spark.createDataFrame(pandasDF)
  ~~~

- Programmatically using **SparkSession.createDataFrame()**
  ~~~ python
  my_list = [['Tom', 23], ['David', 40]]
  df = spark.createDataFrame(my_list, ['name', 'age'])
  ~~~

#### Inspecting DataFrames
- Display Results
  - **df.show(n, truncate=True)** : Display content of df. Optional parameters n and truncate determine the number of rows and whether long values e.g. strings gets truncated
  - **df.printSchema()** : Display schema of df
  - **df.describe().show()** : Display summary statistics (mean, std, min, max, count) of df.
  - **df.explain()** : Print logical and physical plans of df


- Get Values
  - **df.count()** : Return row count of df
  - **df.columns** : Return a list of column names of df
  Note: Unlike pandas, there is no shape function in spark. df.count() and len(df.columns) achieves the same effect  
  - **df.take(n)** : Return first n rows of df
  - **df.select('colname').distinct().count()** : Return count of distinct values in column 'colname'
  - **df.dtypes** : Return a list of column types of df

#### Handling Missing & Erroneous Values
- **df.dropna(how='any', threshold=None, subset=None)** : returns new dataframe omitting rows with null values.
  - how - 'any' or 'all'. If 'any', drop a row if it contains any nulls. If 'all', drop a row only if all its values are null.
  - thresh – int, default None If specified, drop rows that have less than thresh non-null values. This overwrites the how parameter.
  - subset – optional list of column names to consider.


- **df.fillna(value, subset=None) alias na.fill()** : returns new dataframe with missing values filled.
  - value - int, long, float, string, bool or dict. Value to replace null values with.
    - If dictionary is provided, subset is ignored. Key points to the column and value is the replacement value for null rows in that column.
  - subset: optional list of column names to perform fillna operation.


- **df.replace(to_replace, value, subset=None)** : returns a new dataframe replacing a value with another value.
  - to_replace – bool, int, long, float, string, list or dict. Value to be replaced. If the value is a dict, then value is ignored or can be omitted, and to_replace must be a mapping between a value and a replacement.
  - value – bool, int, long, float, string, list or None. The replacement value must be a bool, int, long, float, string or None. If value is a list, value should be of the same length and type as to_replace. If value is a scalar and to_replace is a sequence, then value is used as a replacement for each item in to_replace.
  - subset – optional list of column names to consider.


#### Selecting Data
  - **filter(condition) alias where()** : returns new dataframe where rows with False condition are omitted.
    - Condition is a column of BooleanType. Column expressions or lambda functions are frequently used to generate the boolean values. Only rows with True values are retained.

  ~~~ python
  df.filter(df['age']>24).show() # retain only rows where age is >24.
  df.filter(df['age'].isNotNull()) # Select only rows which are not null.
  ~~~


 - **select(*cols)** : returns columns in cols in new dataframe.
   - cols - list of column names or expressions (derived columns).

   ~~~ python
   df.select("firstName", "lastName").show() : select columns
   # Select with column expression
   df.select(df['firstName'], (df['age'] + 1).alias('adj_age'))
   ~~~

 - **when(condition, value)** : *Column* function that returns value for rows where condition is True. If *Column*.otherwise(alternate_value) is not invoked, None is returned for False rows.
   - condition - a boolean *Column* expression
   - value - a literal value or *Column* expression

   ~~~ python
   # Select column firstName and derived column with 1s if age is >30 and 0s otherwise.
   df.select("firstName", F.when(df['age'] > 30, 1).otherwise(0)).show()
   ~~~

 - **between(lowerBound, upperBound)** : *Column* function that returns boolean *Column*. True if value in that column is between lower and upper bound and False otherwise.
 ~~~ python
 df.select(df.name, df['age'].between(22, 24)).show()
 ~~~

 - **startswith(other)** : *Column* function that returns boolean *Column*. True if start of string equals other.
   - other - string at start of line (do not use regex ^)
 - **endswith(other)** : *Column* function that returns boolean *Column*. True if end of string equals other.
   - other - string at end of line (do not use regex $)

  ~~~ python
  df.select("firstName", df['lastName'].startswith("Sm")).show()
  df.select(df['lastName'].endswith("th")).show()
  ~~~

 - **substring(startPos, length)** : Return a *Column* which is a substring of the column.
   - startPos - (int) start position. Begins with 1 and not 0.
   - length - (int) length of substring to extract

 ~~~ python
 df.select(df['firstName'].substr(1, 3).alias("name")).collect()
 ~~~

 - **like(other)** : Returns a boolean *Column* based on SQL LIKE match
   - other - SQL LIKE pattern

 ~~~ python
 df.select("firstName", df["lastName"].like("Smith")).show()
 ~~~

 - **rlike(other)** : Returns boolean *Column* based on regex match using SQL LIKE with regex.
 ~~~ python
 df.select("firstName", df['lastName'].rlike("^Smith")).show()
 ~~~

 - **isin(*cols)** : Returns boolean *Column*. For rows where element is in cols, returns True, False otherwise.
    - cols - List of values of various types for comparison.

 ~~~ python
 df[df["firstName"].isin(['Jane', 'Boris'])].collect()
 ~~~


#### Adding, Renaming & Dropping Columns
- **withColumn(colName, col)** : Return new DataFrame by adding column or replacing column with same name
  - colName - (string) name of new column
  - col - (*Column* expression() new column
- **withColumnRenamed(existing, new)** : Returns new DataFrame by renaming existing column.
  - existing - (string) current name of column
  - new - (string) new name of column
- **drop(*cols)** : Returns a new DataFrame that drops the specified column(s)
  - cols - (string, list of strings) name(s) of columns to drop

  ~~~ Python
  # Add Column
  df = df.withColumn('city', df.address.city) \
        .withColumn('postalCode', df.address.postcode) \
        .withColumn('state', df.address.state) \
        .withColumn('telePhoneNumber', explode(df.phoneNumber.number)) \
        .withColumn('telephoneType', explode(df.phoneNumber.type))
  # Rename Column
  df = df.withColumnRenamed('telePhoneNumber', 'phoneNumber')
  # Remove Column
  df = df.drop("address", "phoneNumber")
  ~~~

#### Removing Duplicates
- **dropDuplicates(subset=None)** : Return a new DataFrame with duplicate rows removed. Streaming data have different treatment controlled by withWatermark() function
  - subset - (string or list of strings) columns to drop duplicate.

  ~~~ python
  # Get unique ('Age', 'Gender') tuples
  df.select('Age', 'Gender').dropDuplicates().show()
  ~~~

#### Grouping Data
- **groupBy(*cols)** : Return a GroupData object to run aggregation function on them.
  - cols - (list of strings) column names to group by

  ~~~ python
  df.groupBy("age").count().show() # row counts for each age group
  ~~~

#### Sorting Data
- __sort(*cols, **kwargs)__ alias __orderBy__ : Returns new DataFrame sorted by specified column(s)
  - cols - (list of Column or column names) to sort by
  - ascending - (boolean or list of boolean) sort by ascending(True) or descending(False). Default True.

  ~~~ python
  df.sort("age", ascending=True).collect()
  df.orderBy(['age', 'city'], ascending=[True, False]).collect()
  ~~~


#### Repartitioning Data
- **coalesace(numPartitions)** : Returns new DataFrame with exactly numPartitions partitions
  - numPartitions - (int) target number of partitions
- **repartition(numPartitions, \*cols)** : Returns a new DataFrame partitioned by the given partitioning expressions.
  - numPartitions - (int or Column). If int, will represent the target number of partitions. If column, it will be used as first partioning column.

  ~~~ python
  df.coalesce(1).rdd.getNumPartitions() # Merge to 1 partition
  df.repartition(10).rdd.getNumPartitions() # Split to 10 partitions
  ~~~

#### Registering Views for SQL Type Queries
- Register DataFrame as Views
~~~ python
df.createGlobalTempView("people")
df.createTempView("customer")
df.createOrReplaceTempView("customer")
~~~
- Query Views
~~~ python
df = spark.sql("SELECT * FROM customer").show()
peopledf = spark.sql("SELECT * from global_temp.people").show()
~~~

#### Converting to Other Data Format
~~~ python
my_rdd = df.rdd # Convert to RDD.
myjson = df.toJSON() # Convert to JSON
pandas_df = df.toPandas() # convert to pandas dataframe
~~~

#### Writing to Files
~~~ python
df.write.save("nameAndCity.parquet")
df.select("firstName", "age").write.save("namesAndAges.json", format="json")
~~~

#### Stopping Spark
- Exception could arise when you attempt to create a new spark instance when there is a pre-existing ones. To avoid that, we could do the following:

~~~ python
try:
    spark.stop()
except:
    pass
~~~
