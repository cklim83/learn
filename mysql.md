### Commands
**PRAGMA TABLE_INFO**(tablename)  - List type of each column in table

**Having** statement  - Use to select aggregated rows AFTER group by operation.
                        This filtering cannot be done by WHERE statements which
                        works on individual rows.

**CAST** (Colname As Type) - Casting in SQL. Type can be FLOAT etc.

### Joining 2 Tables
A right join, as the name indicates, is exactly the opposite of a left join.
Where the left join includes all rows in the table before the JOIN clause,
the right join includes all rows in the new table in the JOIN clause.
The main reason a right join would be used is when you are joining more than
two tables. In these cases, using a right join is preferable because it can
avoid restructuring your whole query to join one table. Outside of this,
right joins are used reasonably rarely, so for simple joins it's better to use
a left join than a right as it will be easier for your query to be read and
understood by others.

The other join type not supported by SQLite is a full outer join. A full
outer join will include all rows from the tables on both sides of the join.
Like right joins, full outer joins are reasonably uncommon, and similar results
can be achieved using a union clause (which we will teach in the next mission)

There is a handy shortcut we can use in our queries which lets us skip the
column names, and instead use the order in which the columns appear in the
SELECT clause. In this instance, migration_rate is the second column in our
SELECT clause so we can just use 2 instead of the column name:

'''
SELECT name, migration_rate FROM FACTS
ORDER BY 2 desc;
'''
You can use this shortcut in either the ORDER BY or GROUP BY clauses.
Be mindful that you want to ensure your queries are still readable, so
typing the full column name may be better for more complex queries.

The important thing to remember is that the result of any subqueries are always
calculated first, so we read from the inside out.

'''
select c.name as capital_city, f.name as country,
c.population
from facts f
inner join (
            select * from cities
            where capital=1 and population > 10000000
           ) c on f.id = c.facts_id
order by 3 desc;
'''

When you're writing complex queries with joins and subqueries, it helps to
follow this process:

- Think about what data you need in your final output
- Work out which tables you'll need to join, and whether you will need to join
  to a subquery.
- If you need to join to a subquery, write the subquery first.
- Then start writing your SELECT clause, followed by the join and any other
  clauses you will need.
- Don't be afraid to write your query in steps, running it as you go— for
  instance you can run your subquery as a 'stand alone' query first to make
  sure it looks like you want before writing the outer query.

'''
/* Putting all together */
select
    f.name as country,
    c.urban_pop,
    f.population as total_pop,
    cast(c.urban_pop as float) / cast(f.population as float) as urban_pct
from facts as f
INNER JOIN (
            select facts_id, sum(population) as urban_pop
            from cities
            group by 1
            ) c on f.id = c.facts_id
where urban_pct > 0.5
order by 4 asc;
'''

### Joining 3 or more tables
Process: Use a subquery to generate an intermediate solution subset before
joining with the main table

'''
/* Example Select top 5 albums by quantity of tracks purchased */
select
    aa.album_title as album,
    aa.artist_name as artist,
    SUM(il.quantity) as tracks_purchased
From invoice_line as il
INNER JOIN track as t ON il.track_id = t.track_id
INNER JOIN (Select
                album_id,
                title as album_title,
                name as artist_name
                From album as alb
            INNER JOIN artist a ON alb.artist_id = a.artist_id
            ) aa ON t.album_id = aa.album_id
Group by 1
Order by 3 desc
LIMIT 5;
'''

**Recursive Join**
In some cases, there can be a relation between two columns within the same
table. We can see that in our employee table, where there is a reports_to
column that has a relation to the employee_id column within the same table.

The reports_to column identifies each employee's supervisor. If we wanted
to create a report of each employee and their supervisor's name, we would
need some way of joining a table to itself. Doing this is called a recursive
join.

'''
SELECT
    e1.employee_id,
    e2.employee_id supervisor_id
FROM employee e1
INNER JOIN employee e2 on e1.reports_to = e2.employee_id
LIMIT 4;
'''

**Concatenate Operator**
SELECT ("this" || "is" || "my" || "string")
returns 'thisismystring'


**LIKE**
Syntax column LIKE '%pattern%' - where % is a wildcard match
Note: To ensure case sensitivity do not affect match use
LOWER(column) LIKE '%lower_case_patt%'

**CASE**
- Used to create custom categorisation
- There can be 1 or more WHEN lines, and the ELSE line is optional— without it,
  rows that don't match any WHEN will be assigned a null value
- We can nested case in the then statement if required.
'''
CASE
    WHEN [comparison_1] THEN [value_1]
    WHEN [comparison_2] THEN [value_2]
    ELSE [value_3]
    END
    AS [new_column_name]
'''

### **WITH** For Temporary Subqueries
'''
WITH
    [alias_name] AS ([subquery]),
    [alias_name_2] AS ([subquery_2]),
    [alias_name_n] AS ([subquery_n])

SELECT [main_query]

Example
WITH
    usa AS
        (
        SELECT * FROM customer
        WHERE country = "USA"
        ),
    last_name_g AS
        (
         SELECT * FROM usa
         WHERE last_name LIKE "G%"
        ),
    state_ca AS
        (
        SELECT * FROM last_name_g
        WHERE state = "CA"
        )

SELECT
    first_name,
    last_name,
    country,
    state
FROM state_ca
'''
Notes
- Instead of nesting subqueries with main query, which are difficult to read,
  we can use WITH to support **1 or more** subqueries
- Subqueries logic a bounded by () and separated by commas
- Each subquery can utilise all subqueries defined earlier

### **VIEWS** For Permanent Intermediate Result
'''
CREATE VIEW databasename.tablename AS
  SELECT ...;
'''

- A **VIEW** is a permanent data construct derived from a table that can
  be used by future queries whereas **WITH** constructs are temporary and
  exist only in memory.
- A view, once created, cannot be modified or redefined. Doing do will result
  in error indicating view already exist.
- To edit, we can **drop** it (similar to table) then recreate.

'''
CREATE VIEW chinook.customer_2 AS
    SELECT * FROM chinook.customer;

CREATE VIEW chinook.customer_2 AS
    SELECT
      customer_id,
      first_name || last_name name,
      phone,
      email,
      support_rep_id
    FROM chinook.customer;

Error: table customer_2 already exists

DROP VIEW chinook.customer_2;
'''

### Row Operations (Filter/Append) via Set Operators **UNION, INTERSECT, EXCEPT**
- **UNION** is similar to OR operation and appends two selections
- **INTERSECT** is similar to AND operation and select common rows
- **EXCEPT** is similar to AND NOT and selects rows that occur in the first
  selection but not in the second selection
- All 3 operations requires the two selections to have same number of columns,
  in the same order, and have compatible types (e.g. int and float and
  not str and int)
- Because of commonality requirement, they are usually used on different
  subselections of the same table

'''
SELECT * from customer_usa
UNION
SELECT * from customer_gt_90_dollars;

SELECT * from customer_usa
INTERSECT
SELECT * from customer_gt_90_dollars;

SELECT * from customer_usa
EXCEPT
SELECT * from customer_gt_90_dollars;
'''

### Tips for Complex Queries
- Write your query in stages, and run it as you go to make sure at each stage
  it's producing the output you expect.

- If something isn't behaving as you expect, break parts of the query out
  into their own, separate queries to make sure there's not an inner logic
  error.

- Don't be afraid to write separate queries to check the underlying data,
  for instance you might write a query that you can use to manually check
  a calculation and give yourself confidence that the output you're seeing
  is correct.
