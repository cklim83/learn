### Impala SQL
- Fast data query suitable for ad-hoc and interactive analysis
- SQL like syntax e.g. SELECT * FROM tablename LIKE 'An%';

### Hadoop HDFS Commands
#### List contents
hdfs dfs -ls /          -> List root directory on HDFS (HaDoop FileSystem)
hdfs dfs -ls /user      -> List user directory no HDFS

#### Make and Remove Directories
hdfs dfs -mkdir /mydir  -> Create mydir directory at root
hdfs dfs -rm -r /mydir  -> Recursive removal of contents of mydir

#### Upload and Download files
hdfs dfs -put kb /mydir -> Upload kb folder from current directory of local
                           system to /mydir in hdfs
hdfs dfs -get src dest  -> Download from src in hdfs to dest on local drive

#### Print file contents using cat
hdfs dfs -cat /path/file | head -n 20 -> Cat (i.e. print to stdout) contents
                                         of file restricted to first 20 lines
                                         (pipe with head -n 20)
hdfs dfs -cat /path/file | less       -> Print content in pages, right arrow to
                                         go next page. 'q' to quit.
hdfs dfs -cat /path/file | more       -> Print line by line. Enter or right
                                         arrow to go next line. 'q' to quit.
hdfs dfs -tail /path/file             -> Print the end section of file
hdfs dfs -cat /path/file | tail -n 5  -> Cat last 5 lines of file.

#### Help
hdfs dfs                -> List all common hdfs commands
Note:
- Home directory in hdfs defaults to /user/username
- No concept of current/working directory in HDFS (i.e. pwd not applicable).
- Relative paths are all relative to home directory

### Submit SPARK job to YARN Cluster
spark2-submit local_src hdfs_target

### SPARK
- Refer to http://spark.apache.org/docs/2.2.0/ for SPARK documentation
- From the Programming Guides menu, select the DataFrames, Datasets and SQL.
- If you are viewing the Scala API, notice that the package names are displayed on the left. Use the search box or scroll down to find the org.apache.spark.sql package. DataFrames are simply an alias for Datasets of Row objects.
- If you are viewing the Python API, locate the pyspark.sql module.

pyspark2          -> start python spark2 interactive shell
spark2-shell      -> start scala shell
*In both cases, Spark will return a SparkSession object called spark
<pyspark.sql.session.SparkSession at address> (pyspark)
org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@address (scala)

spark.<TAB>       -> tab completion to see SparkSession methods available
sys.exit          -> scala exit of spark shell
Ctrl + D or exit  -> python exit of spark shell

#### Read and Display Json File
devDF = spark.read.json("/hdfsfile.json")   ->create dataframe. File not read yet, only its schema is inferred,

**SPARK Actions (Values will be returned to Spark Driver)**
devDF.printSchema()         -> print dataframe Schema
devDF.show(n)               -> Show first n rows. Defaults to 20 if n not provided
rows = devDF.take(n)        -> Take returns a list(Python)/array(Scala) of Row objects
devDF.count()               -> Returns number of rows in DataFrame

**SPARK Transformations. Return a new Dataframe but No Value**
makeModelDF = devDF.select("col1", "col2")  -> Select column1 and 2
Note: Since transformations return new dataframe, we can chain them together.

**Spark Query (Chained Transformations with Action)**
devDF.select("col1", "col3").where("make='Ronin'").show()
