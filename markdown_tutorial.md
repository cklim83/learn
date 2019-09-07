# Markdown Tutorial
This is a simple tutorial covering commonly used markdown syntax.

### Tools
Try Marked 2 for preview. Seems to work better than free online preview. 

###  Headers
Headers using # runs from H1 to H6. Markdown also automatically use headers to generate a table of contents, building links to navigate to that section directly. An alternative way to create a H1 header is to have === below header 1 text. To have header 2 text, use --- below header text instead.

### Examples of Headers
#####  This is a H4 header (shortkey: Ctr + 4)
###### This is the smallest H6 header (shortkey + 6)

Alternate Header 1
===

Alternate Header 2
---


# Paragraphs
To create a separate paragraph, markdown requires full empty line in between before it will create a new line. Otherwise the text will all align within the same paragraph. 


Single Line:
one
two
three

Multiple Lines:

one

two

three


# Italics, Bold and Strikethroughs
_Italics_ using single _ wrap

*Italics* using single * wrap

__Bold__ using double __ wrap

**Bold** using double ** wrap

~~Strikethrough~~ using double ~~ wrap

# Links
**Method 1**: wrap url within <>

**Method 2**: wrap link text with [],followed by url wrapped in (). You can include hover text as second parameter within quotes in parenthesis

**Method 3**: footnote style

### Examples

<http://github.com>

[Github](http://github.com)

[Hackeryou.com](http://hackeryou.com "This is where wes teaches")

Make sure you check out [Wes'][1] Site. 
If you want to lean React, you can learn at React for beginners.com - [Wes][1] did a great job on this.

[Wes][1] also teaches at [HackerYou][hack] where you can come and learn with him in person. Check out their site for more info,


[1]: http://wesbro.com
[hack]: http://hackeryou.com

# Images
! [] () without spaces in between is the syntax for inserting images. [] will house the **optional alt-text** used by search engines. () will house the url followed by optional hover text in "" as second parameter.

Footnote method similar to links also work for images.

### Picture 1
![Wow great prc!!](http://unsplash.it/500/500?random "This is tooltip")

### Picture 2
![][pic-1]

Random text after picture.

[pic-1]: http://unsplash.it/500/500?random

### Thumbnail Link To Full Image
[![](http://unsplash.it/50/50?random)](http://unsplash.it/500/500?random)

### Using img src html tag
[<img src="http://unsplash.it/50/50?random">](http://unsplah.it/500/500?random)

<img src="dog.jpg" width="500" height="500" alt="">

### Using CSS to style
<style>
	img {
		width:560px;
	}
</style>

# Lists

### Unordered List
Can use either '*', '-'' or '+'' for unordered list.

+ item 1
+ item 2
+ item 3

### Ordered List
Use 1. in front of all list items. Their position in sequence will automatically give them the right number.

1. item 1
	* After item 1, check this.
		
		This is inline.

		![](http://unsplash.it/500/500?random "Random photo")

		```python
			def my_function():
				pass
		```
		
1. item 2
1. item 3


###  Line Breaks, Horizontal Rules and Block Quotes

**Line Breaks**

Wes <br>
West2

West3ff

**Horizontal Rule**

---

===

>Block Quote
>
> - **Winston Churchill**



### Code Blocks & Syntax Highlighting

Here is my code:

Using indentation

	var x = 100;
	const dog = 'snickers'

```php
$age = 50;
$name = "wes";
echo strtoupper($name);
```

Wrapping code in back ticks in reply.

Hey did you try `var x =100;`?

```diff
var x = 100;
- var y = 200;
+ var y = 300;
```

### Tables
':' placement determines text alignment. Left for left align, right for right align, both ends for center align.

|Dog's Name| Age|
|:--------:|----:|
|Snickers|2|
|Prudence|8|


### Create Checkboxes

* [ ] Get Milk
* [x] Crack Eggs
* [ ] Cook Bacon











