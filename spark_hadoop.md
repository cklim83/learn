### What is Apache Spark
- General purpose data processing engine for large datasets
- Written in Scala, a Java Virtual Machine functional programming language
- Two ways to run spark code: Spark Shell or Script
- Spark applications can be written in Python, Scala or Java
- Apache Spark consist of Core Spark and a collection of APIs on top of it.
  - Core Spark API provides Sparks underlying data abstraction:
    resilient distributed datasets (RDDs)
  - Spark SQL provide functions to work with structured data
  - MLib provide functions for scalable machine learning
  - Spark Streaming provides API for applications that handles processing and
    analysis of live datastreams in real-time
  - GraphX library is used for graph data and graph-parallel computation
    (Not mature yet so not supported by Cloudera for production use)

#### SPARK SQL
- Library for working with structured data
- Provide DataFrame and DataSet APIs
- DataFrame and DataSets are abstractions representing structured data in
  tabular form
- Can use these APIs to query and transform data in SQL-like operations
- Also includes Catalyst Optimizer which helps optimize Spark SQL data
  transformations and queries on Spark's distributed architecture
- Catalyst optimized SQL operations can be hundreds of times faster than
  manually coded transformations using core Spark
- General purpose SQL - includes SQL command line interface with a
  Thrift JDBC/ODBC server

#### Spark Shell
- Provides interactive Spark environment (Read/Evaluate/Print/Loop)
- Useful for learning, testing and ad-hoc analysis
- Two versions: Scala and Python
- Typically run on a gateway node that is part of Hadoop Cluster
- To start spark2 shell:
  - pyspark2 for Python
  - spark2-shell for Scala
  - 2 is required as Cloudera support both Spark 1.0 and 2.0 in same environment
- Spark Shell Options
  - master: cluster to connect to
  - jars: additional non-spark libraries required by application (Scala version)
  - py-files: additional non-spark libraries required by application (python version)
  - name: name of spark application (defaults to PySparkShell or Spark shell)
  - help: show all available shell options
  - e.g. pyspark2 --name "My application"

#### Spark Cluster options
- Three types of clusters:
  - Apache Hadoop YARN (Yet Another Resource Negotiator)
  - Apache Mesos
  - Spark Standalone
- Can also run locally
- Defaults to YARN for cluster configured by Cloudera Manager
- Specify the type or url of cluster using master option
- pyspark2 --master yarn[spark|mesos://masternode:port] or
  pyspark2 --master local [n] to run locally on n threads


#### Spark Session
- All spark applications starts with a SparkSession object
- SparkSession object is automatically created and assign to variable 'spark'
- when spark shell is launched.
- Attributes of a SparkSession object:
  - sql: execute spark SQL query
  - cataloq: entry point for Cataloq API for managing tables
  - read: function to read data from file or other data sources
  - conf: object to manage Spark configuration settings
  - sparkContext: returns sparkContext object as entry to core Spark API
- Spark uses Apache Log4j for logging
- Log4j allows applications to log messages at different levels
  [TRACE, DEBUG, INFO, WARN, ERROR, FATAL, OFF] in ascending levels
  - setting at WARN will capture all levels higher or equal to WARN.
  - Defaults to INFO for spark applications and WARN for spark shell
  - spark.sparkContext.setLogLevel("INFO") # To change logging level
  - Can also use setLogLevel function but is not ideal since it
    hardcodes the setting.

#### DataFrames and Datasets
- DataFrames are ordered collection of **Row** objects that can contain any
  values of various types (e.g. strings, float, int, arrays)
- DataFrames have schemas that map row values to named columns of specific types
- Datasets are similar to DataFrames except that instead of Row objects,
  they contain objects whose types are known at compile time
- Hence Datasets can enforce type consistency while DataFrame could not
- Datasets are only defined in Java and Scala
- Python doesnt use datasets since it is a loosely-typed language
- In Scala, there is actually no DataFrame class. Dataframe is alias for
  Dataset[Row] i.e. a Dataset with Row objects
- Supported JSON format in Spark is JSON Lines i.e. each line is a single
  JSON record

'''python
userDF = spark.read.json("filename.json")
userDF.printSchema()
userDF.show()  # defaults by printing 1st 20 rows
'''

#### Dataframe Operations
- Transformations:
  - Returns a new DataFrame based on existing ones
    - Hence we can chain transformations
    - DataFrames are immutable -> Transformations never modify original
  - **Does not** return data to driver, data remains distributed across
    application's executors
  -  Common ones are:
    - **select**: specify columns to keep
    - **where**: retain rows where expression is true (synonym as filter)
    - **orderBy**: rows are sorted by specified column(s)
    - **join**: Merge two dataframes on specified column(s)
    - **limit(n)**: returns new DataFrame with first n rows.


- Actions: Output values from DataFrame either saved to file or return to
  application driver process
  - **count**: return number of rows
  - **first**: returns first row (synonym to head)
  - **take(n)**: returns first n rows as array (synonym to head(n))
  - **show(n)**: display first n rows (default n = 20)
  - **collect**: returns all rows as array (Not for large dataset uses
    large bandwidth to transfer data from distributed executors and may
    exceed available memory on driver)
  - **write**: save data to file or other data source


- Queries: A sequence of transformations followed by an action

'''Python
usersDF= spark.read.json("users.json")
usersDF.select("name", "age").where("age > 20").show() # chain transformations
'''

#### Working With DataFrames and Schemas
- Spark SQL supported data types
  - text files (csv, json, txt)
  - binary format (Apache Parquet, Apache ORC)
  - tables (Hive metastore, JDBC)
- Parquet:
  - Optimized binary store for large datasets
  - Metadata embedded in file
  - Supported by Hadoop ecosystem tools such as Spark, MapReduce, Hive
    and Impala
  - Use parquet-tools to view head and schema of file

'''
$ parquet-tools head myfile.parquet # display first few records
$ parquet-tools schema myfile.parquet # display schema of file
'''

#### Creating DataFrame from Data Source
- spark.read returns a DataFrameReader object with following Attributes
  - format: csv, json, parquet (default) etc
  - option: key:value pair e.g. "Header": "True"
  - schema: specify a specific schema for data instead of inferring
- load/table: Chained function to load data from file/Hive table
'''
df = spark.read.format("csv").option("header", "true").load("/path.csv")
df = spark.read.csv("/path.csv") # Shortcut to combine format & load
mytable = spark.read.table("my_table")
'''

- file paths could be single file, list of files, a directory or wildcard
'''
spark.read.json("myfile.json")
spark.read.json("mydata/")
spark.read.json("mydata/*.json")
spark.read.json("myfile1.json", "myfile2.json")
'''
- path can be relative (to default file system) or absolute
  - myfile.json
  - hdfs://nnhost/loudacre/myfile.json
  - file://home/training/myfile.json

- Create DataFrame from Data in Memory
'''
mydata = List(("Josiah", "Bartlett"),
              ("Harry", "Porter"))
df = spark.createDataFrame(mydata)
df.show()
'''

#### Saving DataFrames to Data Source
- DataFrame.write returns a DataFrameWrite with these attributes:
  - format: csv, json, parquet (default) etc
  - mode: behavior when file/table already exist
    - error (default), overwrite, append, ignore
  - partitionBy: stores data in partitioned directories in form colum=value
  - option: specifies properties in "key", "value" pair for target source
  - save/saveAsTable: similar to load/table in read. Provide **directory** to
    save output.

'''
'# Save in mydata folder'
df.write.mode("append").option("path", "/loudacre/mydata").saveAsTable("my_table")
df.write.save("mydata") # save as parquet files in mydata relative directory
df.write.json("mydata") # shortcut to save a json in mydata relative directory
'''

#### DataFrame Schemas
- define the names and types of each column
- defined upon DataFrame creation and immutable thereafter
- When creating a new DataFrame from data source, schema is either inferred
  or specified programmatically
- When DataFrame is created via transformation, Spark calculates new schema
  based on query
- Spark can infer schema from structured data
  - Parquet files (schema metadata is stored in file)
  - Hive tables (schema stored in Hive metastore)
  - Parent DataFrames
- Spark can also attempts to infer schema from semi-structured sources such
  as json and csv.

'''
'# Infer schema + header present'
spark.read.option("inferSchema", "true").option("header", "true").csv("people.csv")
printSchema()
'''

Disadvantages with auto schema inferrence
- Inference relies on file scan, which may be slow for large files
- Schema may be incorrect for use case. Example a postcode inferred as integer
  when arithmatic doesnt make sense and is best treated as a string.

Manually Define Schema
- Schema is a **StructType** Object containing list of **StructField** objects
- StructField specify column name, data type, whether data is nullable (default to true)

'''
from pyspark.sql.types import *
columnsList = [
  StructField("pcode", StringType()),
  StructField("lastname", StringType()),
  StructField("firstname", StringType()),
  StructField("age", IntegerType())
]

peopleSchema = StructType(columnsList)
spark.read.option("header", "true").schema(peopleSchema).csv("people.csv")
'''

#### Eager and Lazy Execution
- Operations are **Eager** when executed when statement is reached
- **Lazy** when execution occurs only when results is referenced
- Spark queries are executed both lazily and eagerly
  - DataFrame schemas are determined eagerly at creation
  - Transformations (including data loading from source) are executed lazily,
    and only triggered when action is called

### Analyzing Data with DataFrame Queries
#### Querying using Column Expressions
- Simple column selection: df.select("col1", "col3")
- Column selection using reference: df.select(df['col1']) or df.select(df.col1)
  allows us to derive new columns using column expressions renamed using alias()

'''
df.select("lastName", (df.age*10).alias("age_10")).show()
df.where(df.firstName.startswith("A")).show()
'''

#### Grouping and Aggregation Queries
- groupBy takes one or more column names/references and returns a GroupedData object
- Returned object provides aggregation functions such as
  - count, max, min, mean/avg, sum, pivot, agg (aggregate using additional
    aggregation functions such as: first/last, countDistinct,
    approx_count_distinct (faster than full count), stddev,
    var_sample/var_pop (variance of sample or population),
    covar_samp/covar_pop (covariance of sample or population),
    corr (correlation)

'''
df.groupBy("pcode").count().show()

import pyspark.sql.functions as functions
df.groupBy("pcode").agg(functions.stddev("age")).show()
'''

#### Joining DataFrames
- **join** transformation to join two DataFrames
- Join types: inner (default), outer, left_outer, right_outer, leftsemi, crossJoin

'''
df.join(df2, "pcode", "left_outer").show() # left join where colnames are same
df.join(df2, df1.pcode==df2.zip).show() # inner join where colnames are different
'''

### Resilient Distributed Datasets(RDDs)
- RDDs are part of core Spark
- Resilient: If data is lost in memory, it can be recreated
- Distributed: Processed across cluster
- Dataset: Initial data can be from source (e.g. file) or created in memory
- RDDs are unstructured: no schema, not table-like (cannot be queried with sql
  transformation like where and select), uses lambda functions for RDD
  transformation.
- RDDs can contain any object types vs only Row objects for Dataframe and Row,
  case class objects and primitive types for Datasets
- RDDs not optimised by Catalyst optimizer, manually coded ones tends to be
  less efficient that DataFrames
- RDDs interconvertable with DataFrame and Datasets

#### RDD Data Types
- Primitive types such as int, char, booleans
- collections such strings, list, dictionaries and their nested variants
- Scala and Java Objects (if serializable)
- Mixed Types

#### RDD Data Sources
- Files (text and other formats)
- Data in Memory
- Other RDDs
- Datasets or DataFrames

#### Create RDDs from Files
- uses SparkContext rather than SparkSession
- SparkContext is part of core Spark library but SparkSession in part of Spark SQL library
- One Spark context per application
- SparkSession.sparkContext for accesss, variable sc in spark shell

#### Create RDDs from Text Files
- SparkContext.textFile reads '\n' terminated text files
- Each line = each RDD element
- Accepts a single file, directory, wildcard list of files, or comma
  separated list of files

'''
myRDD = spark.sparkContext.textFile("mydatadir/")
'''

Multi-line Input files e.g. JSON and XML
- Use wholeTextFiles
- Each file = each RDD element
- Only for small files as each element must fit in Memory

'''
userRDD = spark.sparkContext.wholeTextFiles("userFiles")
'''

#### RDDs from Collections (In Memory)
- Use SparkContext.parallelize(collection)

'''
data = ['Alice', 'Carlos', 'Frank']
myRDD = spark.sparkContext.parallelize(data)
'''

#### Saving RDDs
- Use RDD.saveAsTextFile to save as plain text files
- Use RDD.saveAsHadoopFile or saveAsNewAPIHadoopFile with specified
  Hadoop OutputFormat to save to other formats

#### RDD Operations
- Action (Returns data to spark driver by triggering transformation to be completed)
- Common Actions:
  - count : returns number of elements
  - first : returns first element
  - take(n) : returns list of first n elements
  - collect : return list of all elements
  - saveAsTextFile(dir) : save to text files
- Transformations (Performed Lazily)
- Create New RDDs from existing ones since RDDs are immutable
- Common transformation operations include:
  - RDD.distinct() : create new RDDs with duplicate elements removed
  - RDD.union(rdd) : create new RDD by appending rdd to caller RDD
  - RDD.map(function) : create new RDD by applying func to each element of caller RDD
  - RDD.filter(func) : create new RDD by applying func to each element of caller RDD


  ### Transforming Data with RDDs
  #### Functional Programming in Spark
  - Functional programming is based on input and outputs only with no state or
    side effects
  - Functions passed as parameters to other functions known as procedural
    parameters
  - Sparks architecture allows passed functions to be executed in multiple
    executors in parallel

  #### RDD Transformation Procedures
  - RDD transformations transform elements of RDD to new elements on executors
  - Some functions e.g. **distinct** and **union** implement their own
    transformation logic
  - Most transformations require caller to pass a function to be applied
    e.g. map(function_to_apply) and filter(function to yield boolean for selection)
  - Passed functions can be named or anonymous (e.g. lambda function in python)
    e.g. upperRDD = myRDD.map(lambda line: line.upper())

  #### RDD Execution
  - RDD queries are executed lazily as they have no schema. Hence they do not
    need to scan data and transformations executed only when action is called
  - DataFrame/Dataset first scan data to determine schema eagerly before
    data loading.
  - Transformation create new children RDDs from base RDD
  - We can use **toDebugString** to trace the lineage (transformation sequence)
    from source to its current form.
    e.g. print myFilteredRDD.toDebugString()
  - Where possible, Spark will complete all chained transformations and action
    for each element before moving to the next one to avoid the need to store
    intermediate results.

  #### Converting between RDDs and DataFrames
  - We can create a DataFrame via an RDD on unstructured/semi-structured data
    e.g. text using SparkSession.createDataFrame(myRDD, schema)
  - Can return underlying RDD of a dataframe using .rdd attribute

  '''
  from pyspark.sql.types import *
  mySchema = StructType([StructField("pcode", StringType()),
                         StructField("lastName", StringType()),
                         StructField("firstName", StringType()),
                         StructField("age", IntegerType())])
  myRDD = sc.textFile("people.txt"). \
    map(lambda line: line.split(",")). \
    map(lambda values: [values[0], values[1], values[2], int(values[3])])
  myDF = spark.createDataFrame(myRDD, mySchema)
  myDF.show(2)

  myRDD2 = myDF.rdd
  for row in myRDD2.take(2):
    print row
  '''

  ### Aggregating Data with Pair RDDs
  - paired RDDs are special as each element is a key-value pair (key, value)
  - keys and values can be of any data type
  - paired RDDs are commonly used with map-reduce and related tasks
    (e.g. sorting, joining, grouping and counting)

  #### Creating Paired RDDs
  - Functions to create key value pairs
    - map
    - flatMap/ flatMapValues
    - keyBy

'''
**# Example: Key Web log by User IDs**
56.38.234.188 - 99788 "GET /KBDOC-00157.html http/1.0" …
56.38.234.188 - 99788 "GET /theme.css http/1.0" …
203.146.17.59 - 25254 "GET /KBDOC-00230.html http/1.0" …

sc.textFile("weblogs/"). \
  keyBy(lambda line: line.split(' ')[2])

(key, value)
(99788, 56.38.234.188 - 99788 "GET /KBDOC-00157.html…)
(99788, 56.38.234.188 - 99788 "GET /theme.css…)
(25254, 203.146.17.59 - 25254 "GET /KBDOC-00230.html…)
'''

'''
**# Pairs with complex values**
00210\t43.00589\t-71.01320
01014\t42.17073\t-72.60484
01062\t42.32423\t-72.67915
01263\t42.3929\t-73.22848

sc.textFile("latlon.tsv"). \
  map(lambda line: line.split('\t')). \
  map(lambda values: (values[0], (float(values[1]), float(values[2]))))

("00210",(43.00589,-71.01320))
("01014",(42.17073,-72.60484))
("01062",(42.32423,-72.67915))
("01263",(42.3929,-73.22848))
'''

'''
**# FlatMapValues** splits a value field into multiple elements
"00001 sku010:sku933:sku022"
"00002 sku912:sku331"
"00003 sku888:sku022:sku010:sku594"
"00004 sku411"

orderRDD = sc.textFile("orderskus.txt"). \
  map(lambda line: line.split(' ')). \
  map(lambda values: (values[0], values[1])). \
  flatMapValues(lambda skus: skus.split(":"))

("00001","sku010")
("00001","sku933")
("00001","sku022")
("00002","sku912")
("00002","sku331")
("00003","sku888")
'''

#### MapReduce
- common programming model to process distributed datasets
- Hadoop mapreduce jobs only have single map and reduce phase
- Spark's implementation more flexible, allows map and reduce jobs to be
  interspersed via chaining
- Works on paired RDDs.
- Map phase: map each record to 0 or more records using functions such as
  map, filter, flatMap etc
- Reduce phase: works on map output and consolidates multiple using functions
  such as reduceByKey, sortByKey, mean etc.

'''
**Word count Example**

countsRDD = sc.textFile("catsat.txt"). \
  flatMap(lambda line: line.split(' ')). \
  map(lambda word: (word, 1)). \
  reduceByKey(lambda v1, v2: v1 + v2) # function combines values with same key

Note: As function to reduceByKey may be called in any other, it has to be
  1) commutative: x + y = y + x
  2) associative: (x+y) + z = x + (y+z)

the       (the, 1)        (on, 2)
cat       (cat, 1)        (sofa, 1)
sat       (sat, 1)        (mat, 1)
on        (on, 1)         (aardvark, 1)
the       (the, 1)        (the, 3)
mat       (mat, 1)        (cat, 1)
the       (the, 1)
aardvark  (aardvark, 1)
'''

#### Other Pair RDD Operations
- **countByKey** returns the count of occurrence of each key
- **groupByKey** groups all values for each key in RDD
- **sortByKey** sorts in ascending or descending order
- **join** returns RDDs containing all pairs with matching keys from 2 RDDs
- **leftOuterJoin, rightOuterJoin , fullOuterJoin** join two RDDs, including keys
  defined in the left, right, or both RDDs respectively
- **mapValues, flatMapValues** execute a function on just the values, keeping the
  key the same
- **lookup(key)** returns the value(s) for a key as a list


### Querying Tables and Views with APACHE SPARK SQL
