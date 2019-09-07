#!/bin/bash

# Basics exercise
echo hello, world
echo "hello, world"
echo 'Use "man echo"'

sleep 5 # Sleep 5 seconds


# Manipulating Files
echo "From fairest creatures we desire increase," > line_1.txt
echo "That thereby beauty's rose might never die," >> line_2.txt

# concatenate two files to stdout and redirect to sonnet_1.txt
cat line_1.txt line_2.txt > sonnet_1.txt
echo "from fairest creatures we desire increase," > line_1_lowercap.txt
diff line_1.txt line_1_lowercap.txt  # compare two files and print difference

# Exercises
# Use the echo command and the redirect operator > to make a file called foo.txt
# containing the text “hello, world”. Then, using the cp command, make a copy of
# foo.txt called bar.txt. Using the diff command, confirm that the contents of
# both files are the same.

echo "hello, world" > foo.txt
cp foo.txt bar.txt
diff foo.txt bar.txt


# By combining the cat command and the redirect operator >, create a copy of
# foo.txt called baz.txt without using the cp command.

cat foo.txt > baz.txt


# Create a file called quux.txt containing the contents of foo.txt followed
# by the contents of bar.txt. Hint: As noted in Section 2.1.1, cat can take
# multiple arguments.

cat foo.txt bar.txt > quux.txt


# How do rm nonexistent and rm -f nonexistent differ for a nonexistent file?

# Answer: rm nonexistant will result in "cannot remove nonexistant": no such file
# or directory while rm -f nonexistant will remain silent with no error warning.
