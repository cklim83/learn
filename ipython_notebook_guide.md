### Help and Documentation
func_name? + shift+<Tab>    - display function docstring
func_name??   +shift+<Tab>  - display function source code. If object is not
                              implemented in Python but in C or some compiled extension, ?? will return same info as ?.


'''Python
In [3]: L = [1, 2, 3]
In [4]: L.insert?
Type:        builtin_function_or_method
String form: <built-in method insert of list object at 0x1024b8ea8>
Docstring:   L.insert(index, object) -- insert object before index
'''

''' Python
In [8]: square??
Type:        function
String form: <function square at 0x103713cb0>
Definition:  square(a)
Source:
def square(a):
    "Return the square of a"
    return a ** 2
'''

## Explore Modules with Tab Completion
### Tab-completion of object contents
obj.<Tab>     - will list all methods/attributes associated with object.
obj.co<Tab>   - will list all methods/attributes of object beginning with co.
'''Python
In [10]: L.<TAB>
L.append   L.copy     L.extend   L.insert   L.remove   L.sort
L.clear    L.count    L.index    L.pop      L.reverse

In [10]: L.c<TAB>
L.clear  L.copy   L.count

In [10]: L.co<TAB>
L.copy   L.count
'''

### Tab completion when importing
from pkg import co<Tab> - list all imports in package beginning with co

'''Python
In [10]: from itertools import co<TAB>
combinations                   compress
combinations_with_replacement  count

In [10]: import <TAB>
Display all 399 possibilities? (y or n)
Crypto              dis                 py_compile
Cython              distutils           pyclbr
...                 ...                 ...
difflib             pwd                 zmq

In [10]: import h<TAB>
hashlib             hmac                http
heapq               html                husl
'''

### Wildcard Matching
Tab completion is useful if you know the first few characters of the object or attribute you're looking for, but is little help if you'd like to match characters at the middle or end of the word. For this use-case, IPython provides a means of wildcard matching for names using the * character, which matches any string, including empty string.

'''IPython
In [10]: *Warning?
BytesWarning                  RuntimeWarning
DeprecationWarning            SyntaxWarning
FutureWarning                 UnicodeWarning
ImportWarning                 UserWarning
PendingDeprecationWarning     Warning
ResourceWarning

In [10]: str.*find*?
str.find
str.rfind
'''

## Magic Commands
%   - Single prefix for line magics
%%  - Double prefix for cell magics

%run filename.py  - run an external file. After running, all functions, objects
                    defined are available for use. %run? for available options.
%timeit ...       - time single line code
%%timeit ...      - time multiline code in cell
%magic            - access magic function Documentation
%lsmagic          - list all magic functions

'''
In [8]: %timeit L = [n ** 2 for n in range(1000)]
1000 loops, best of 3: 325 µs per loop

In [9]: %%timeit
   ...: L = []
   ...: for n in range(1000):
   ...:     L.append(n ** 2)
   ...:
1000 loops, best of 3: 373 µs per loop
'''

## Input and Output History
- All inputs and outputs in IPython are saved.
- Inputs are saved in a list while output in dictionary with key
  the index of the input
- Outputs allow us to reuse results from very expensive computations
  not saved in variable
- to suppress output of computation and omit them from output variable,
  add a ; to the statement e.g. 3+4;

print(In)       - Show all inputs
In[X]           - Access input at index X
%history -n 1-4 - List 1st 4 inputs. Type %history? for more options.

print(Out)      - Show all outputs
_               - Single _ to access last output
__              - Double _ to access second-to-last output
___             - Three _ to access third-from-last output
_X              - Shorthand for Out[X]

%rerun          - rerun some command history
%save           - save some command history to file
