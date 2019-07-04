### Commands
**echo [string]**   - print string to stdout <br>
**man [command]**   - display manual page of command <br>
**clear**           - clear screen <br>
**exit**            - exit shell e.g. python interactive shell <br>
**sleep [x]**       - let process sleep for x secs <br>

**echo [string] > filename.txt**  - redirect string to filename.txt <br>
**echo [string] >> filename.txt** - redirect string and append to filename.txt <br>
**cat file.txt**                  - print content of file to standard output <br>
** cat file1.txt file2.txt**       - concate and print content of file1 and file2 to stdout <br>
**diff file1.txt file2.txt**      - compare difference between two files <br>
**cat file1.txt file2.txt > file3.txt** - concat content in file1 and file 2 and redirect to file3.txt <br>

#### List files and Directories
**ls**          - list (non-hidden) files and directories in pwd <br>
**ls *.txt**    - list all files ending with .txt in pwd <br>
**ls -l *.txt** - list all files ending with .txt in long form (file           permissions, datetime last modification, file size) <br>
**ls -rtl**     - list all files and subdirectories in long form in reversed time
                  order. Most recently modified at the bottom.
                  **ls -r -t -l** or **ls -lrt** all works the same. <br>
**ls -a**       - a option to list all files and directories, including
                  hidden ones that are preceded by . e.g. .gitignore <br>
**ls -al**      - list all files in long form <br>

#### Move/Rename, Copy and Remove Files Commands
**mv test test.txt**        - rename file from test to test.txt <br>
**cp test1.txt test2.txt**  - create copy called test2.txt from test1.txt <br>
**rm test1.txt**    - remove filename test1.txt. rm -i test1.txt is run
                      implicit which seeks confirmation prior to deletion <br>
**rm -f *.txt**     - f option overrides default -i option and remove
                      all .txt files without confirmation <br>
**rmdir [mydir]**   - remove mydir <br>


#### Files Inspection Commands
**head filename.txt** - list first 10 lines of file to stdout <br>
**tail filename.txt** - list last 10 lines of file to stdout <br>
**wc filename.txt** - list the lines, words and bytes count of a file <br>
**head filename.txt | wc** - pipe result of head filename.txt for counting by wc <br>

#### "Less" Command
**less filename.txt** - list a large file by pages. <br>
**up & down arrow** keys	- Move up or down one line <br>
**spacebar**	- Move forward one page	<br>
**⌃F**	- Move forward one page	<br>
**⌃B**	- Move back one page <br>
**G**	- Move to end of file	<br>
**1G**	- Move to beginning of file <br>
**/[string]**	- Search file for string	<br>
**n**	- Move to next search result <br>
**N**	- Move to previous search result <br>
**q**	- Quit less <br>

#### "Grep" Command
Grep stands for **G**lobally search a **r**egular **e**xpression and **p**rint. <br>
**grep [string] [file]**	- Find string in file	e.g. grep foo bar.txt <br>
**grep -i [string] [file]**	- Find case-insensitively	e.g. grep -i foo bar.txt <br>

#### Searching and Killing Processes
**ps**	- Show processes	e.g. $ ps aux <br>
**top** - 	Show processes (sorted)	e.g. $ top <br>
**kill -[level] pid** - Kill a process	$ kill -15 24601 <br>
**pkill -[level] -f [name]**	- Kill matching processes	$ pkill -15 -f spring <br>

#### Utility Commands
**which [command]** - Show the path of the binary be executed <br>
**curl --help** - Help files for curl, a utility to download files from url <br>
**!!**  - run previous command <br>
**!curl** - run the last executed curl command <br>


### Keyboard Shortcut
**Ctrl + c**        - terminate current command <br>
**Ctr + a**         - move cursor to start of line <br>
**Ctrl + e**        - move cursor to end of line <br>
**Ctrl + u**        - delete line <br>
**Ctrl + l**        - clear screen. Same as clear command <br>
**Ctrl + D**        - same as exit command <br>
**Up / Down arrow** - browse to previous and current commands <br>
**option + click**  - move cursor to location clicked <br>
**tab**             - autocompletion based on character matching <br>
**Ctrl + r**        - list previous executed commands recursively <br>


### Bash Scripting
**cat #! which bash > mybash.sh** - Use which bash to get path to bash bin <br>
**tar -zcf output_path input_path** - create a tar.gz file <br>

#### Variables
**myvar=value** - assign value (strings or numeric). No space allowed. <br>
**user=$(whoami)** - command substitution. Substitute result of whoami system command for assignment to user variable. <br>
**$user** - variable-value substitution
**output=/tmp/${user}_home** - ${user} is called parameter expansion and similar to variable-value substitution. We cannot use $user instead as there is more character i.e. _home after it. <br>

#### Input, Output Value Redirection
**stdout_str > stdout.txt** - redirect stdout to file <br>
**stderr_str 2> stderr.txt** - redirect stderr msg <br>
**str_output &> stderr_stdout.txt** - redirect both stderr and stdout to file <br>
**stderr_str 2> /dev/null** - redirect stderr msg to null sink.
**cat > file1.txt** - redirect input from stdin using cat + > command to file. <br>
**cat < file1.txt** - redirect file content from file1.txt to stdout using cat + <  command.

#### Functions
''' bash
function my_function {
    ...
}

my_function # function call has to be after function definition
'''

#### String and Numeric Comparison
- less than: **-lt** (number), **<** (string) <br>
- greater than: **-gt** (number), **>** (string) <br>
- equal: **-eq** (number), **=** (string) <br>
- not equal: **-ne** (number), **!=** (string) <br>
- less or equal: **-le** (number), NA for string
- greater or equal: **-ge** (number), NA for string

'''bash
a=1
b=2
[$a -eq $b] # numeric comparison, evaluates to 1 if false, 0 if true.
echo $? # print result of exit status of the most recently executed foreground pipeline/evaluation .i.e. 1
['apple' = 'apple'] # string comparison
echo $? # print 0
'''

**Note**: Comparing strings with integers using numeric comparison operators
will result in the error: integer expression expected

#### Conditional Statements
'''
if test-commands; then # where test-commands can be [ comparison ]
  consequent-commands;
[elif more-test-commands; then
  more-consequents;]
[else alternate-consequents;]
fi
'''

#### Positional Arguments
**$1** - first argument supplied to script/function <br>
**$2** - second argument supplied to script/function <br>
**$#** - count of arguments supplied to script/function <br>
**$*** - all arguments supplied to script/function <br>


#### Bash Loops
- For Loops
'''
**for** i **in** 1 2 3; **do**
    echo $i
**done**

**for** i **in** $( cat items.txt ); **do** echo -n $i | wc -c; **done** # char count of each word in items.txt
'''
- while Loops (repeat while condition is true)
'''
counter=0
**while** [ $counter -lt 3 ]; **do**
    let counter+=1
    echo $counter
**done**
'''

- until Loop (repeat until condition is true)
'''
counter=6
**until** [ $counter -lt 3 ]; **do**
    let counter-=1
    echo $counter
**done**
'''

#### Bash Arithmetic
- Arithmetic expansion: enclose any mathematical expression inside double parentheses
'''
a=$(( 4 + 5))
echo $(( 100 - 1 ))
echo $(( 3 * 11 ))
echo $(( 100 / 9 ))
'''

- expr command: perform an arithmetic operation without enclosing mathematical expression within brackets or quotes. However, need \\* for multiplication to avoid expr: syntax error

'''
a=expr 2+2  # 2+2
echo expr 2 - 3  # 2-3
echo expr 2 \\* 3  # 2*3
echo expr 9 / 3  # 9/3
'''

- let command: similar to expr command, can evaluates a mathematical expression and stores its result into a variable

'''
let x=3+2
let y=4*($x + 1)
let y++ # increment by 1
let y-- # decrement by 1
'''

- bc command. For evaluating float results with varying decimal places.

'''
echo '8.5 / 2.3' | bc # returns 3
echo 'scale=2; 8.5/2.3' | bc  # returns 3.69
echo 'scale=5; 8.5/2.3' | bc # returns 3.69565
squareroot=$( echo 'scale=50;sqrt(50)' | bc )
echo $squareroot # returns 7.07106781186547524400844362104849039284835937688474
'''
