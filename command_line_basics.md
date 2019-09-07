## Introduction to Command Line
Source: https://launchschool.com/books/command_line/read/introduction

### Common Commands
- cd to change directory
- pwd: present working directory
- mv: move or rename directory
- man: command manual
- tar: archive managers
- less
- more
- cat
- echo
  - echo "Message": prints "Message" to std::out
  - echo "Hello" > path/myfile.txt: Redirect "Hello" to first line in myfile.txt. Overwrites first line if one exist.
  - echo "More messages" >> path/myfile.txt: Redirect "More messages" by appending myfile.txt
- head
- tail
- Use of *
- Ctrl + c to terminate command


### Relative vs Absolute Path
- folder/subfolder: relative from current directory
- /folder/subfolder: folder exists in the root

### Environment
- type env to show the values values of environment variables
- To make permanent changes to command line environment, we need to modify .bashrc or .bash_profile in home directory
- commands in terminals are just executable files
- The PATH variable determines which directories are searched when a command is entered
- PATH is an ordered, colon-delimited, list of directories that contain executables
- The order of the directories in the PATH variable is first-found-first-execute
- If you use a /, ., or ~ before your command, the command line will interpret that as an actual path to a file, and will not use the PATH variable
- You can add to PATH to make more commands available without having to memorize their exact path
- Modifications to PATH, or any environment variable, on the fly will not be permanent
e.g. PATH=$PATH:/path/to/my/file.exe is only valid in current terminal session and lost once i start a new one.
- Permanent modifications should be done in an environment file, like .bashrc

'''
echo 'export PATH=$PATH:/path/to/my.exe' >> ~/.bashrc # add to tail of PATH
source ~/.bashrc  # refresh terminal with updated bashrc content.
'''

Editing hidden files
- nano ~/.bashrc
- paste PS1="[your custom prompt goes here]" add end of files
- type Ctrl + o then enter to save. Then Ctrl + x to quit.
